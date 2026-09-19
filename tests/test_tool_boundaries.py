"""Regression coverage for tool outcomes, host validation and native read scope."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from techpilot.chat.executor import RepositoryToolExecutor
from techpilot.chat.permissions import ChatPermissionPolicy
from techpilot.chat.session import _tool_call_details, _tool_status
from techpilot.chat.tui import TechPilotTui
from techpilot.engine.agent import Agent
from techpilot.engine.events import CallbackEventSink, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision, PermissionManager
from techpilot.engine.tool_results import ToolResult, ToolStatus
from techpilot.engine.tools import get_tool
from techpilot.engine.trusted_diff import FileWriteProposal, TrustedDiffError
from techpilot.runtime.sessions import SessionEventSink, SessionStore


@pytest.mark.parametrize("name,arguments", [
    ("read_file", {"file_path": 123}),
    ("read_file", {"file_path": " "}),
    ("read_file", {"file_path": "x\0.py"}),
    ("read_file", {"file_path": "x", "offset": 0}),
    ("read_file", {"file_path": "x", "limit": -1}),
    ("read_file", {"file_path": "x", "limit": True}),
    ("read_file", {"file_path": "x", "offset": 1.5}),
    ("read_file", {"file_path": "x", "extra": 1}),
    ("read_file", {}),
    ("edit_file", {"file_path": "x", "old_string": "", "new_string": "x"}),
    ("edit_file", {"file_path": "x", "old_string": 123, "new_string": "x"}),
    ("write_file", {"file_path": "x", "content": None}),
    ("bash", {"command": " "}),
    ("bash", {"command": "echo x", "timeout": 0}),
    ("bash", {"command": "echo x", "timeout": True}),
    ("bash", {"command": "echo x", "timeout": "5"}),
    ("glob", {"pattern": ""}),
    ("glob", {"pattern": "*", "path": []}),
    ("grep", {"pattern": 123}),
    ("grep", {"pattern": "x", "include": []}),
])
def test_bad_arguments_fail_before_io_or_prompt(tmp_path, monkeypatch, name, arguments):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid request reached IO or permission prompt")

    monkeypatch.setattr(Path, "read_bytes", unexpected)
    monkeypatch.setattr(Path, "read_text", unexpected)
    monkeypatch.setattr(Path, "write_bytes", unexpected)
    monkeypatch.setattr(Path, "write_text", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    manager = PermissionManager(ChatPermissionPolicy(), SimpleNamespace(decide=unexpected))
    executor = RepositoryToolExecutor(tmp_path, manager)
    tool = get_tool(name)
    for result in (tool.execute(**arguments), executor.execute(tool, arguments)):
        assert result.status is ToolStatus.ERROR
        assert "bad arguments" in result
    agent = Agent(llm=None, tools=[tool], tool_executor=executor)
    assert agent._exec_tool(ToolCall("bad", name, arguments)).status is ToolStatus.ERROR


def test_empty_replacement_and_empty_new_file_remain_valid(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("remove\nkeep\n", encoding="utf-8")
    assert get_tool("edit_file").execute(str(target), "remove\n", "").status is ToolStatus.COMPLETED
    assert target.read_text() == "keep\n"
    assert get_tool("write_file").execute(str(tmp_path / "empty"), "").status is ToolStatus.COMPLETED
    with pytest.raises(TrustedDiffError, match="non-empty"):
        FileWriteProposal.for_edit(target, display_path="sample.py", old_string="", new_string="x")


@pytest.mark.parametrize("name", [".env", ".env.local", ".ENV.PRODUCTION"])
def test_sensitive_reads_and_searches_share_policy(tmp_path, name):
    secret = tmp_path / name
    secret.write_text("PRIVATE_TOKEN_123", encoding="utf-8")
    (tmp_path / "app.py").write_text("PUBLIC_VALUE", encoding="utf-8")
    executor = RepositoryToolExecutor(tmp_path)
    assert executor.execute(get_tool("read_file"), {"file_path": name}).status is ToolStatus.DENIED
    assert executor.execute(get_tool("grep"), {"path": name, "pattern": "."}).status is ToolStatus.DENIED
    assert get_tool("read_file").execute(str(secret)).status is ToolStatus.DENIED
    assert get_tool("grep").execute(".", str(secret)).status is ToolStatus.DENIED
    for tool_name, args in [("glob", {"pattern": "**/*"}), ("grep", {"pattern": "."})]:
        result = executor.execute(get_tool(tool_name), args)
        assert name not in result
        assert "PRIVATE_TOKEN_123" not in result
        assert "app.py" in result


@pytest.mark.parametrize("name", [".env.example", ".env.sample", ".env.template"])
def test_safe_templates_remain_visible(tmp_path, name):
    (tmp_path / name).write_text("EXAMPLE=placeholder", encoding="utf-8")
    executor = RepositoryToolExecutor(tmp_path)
    assert "placeholder" in executor.execute(get_tool("read_file"), {"file_path": name})
    assert name in executor.execute(get_tool("glob"), {"pattern": "**/*"})
    assert "placeholder" in executor.execute(get_tool("grep"), {"pattern": "EXAMPLE"})


def test_searches_do_not_follow_outside_or_sensitive_file_links(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_CONTENT", encoding="utf-8")
    secret = repo / ".env"
    secret.write_text("SECRET_CONTENT", encoding="utf-8")
    try:
        (repo / "external.txt").symlink_to(outside)
        (repo / "alias.txt").symlink_to(secret)
    except OSError:
        pytest.skip("creating symlinks requires platform privileges")
    executor = RepositoryToolExecutor(repo)
    for name, args in [("glob", {"pattern": "**/*"}), ("grep", {"pattern": "."})]:
        result = executor.execute(get_tool(name), args)
        assert "external.txt" not in result and "alias.txt" not in result
        assert "OUTSIDE_CONTENT" not in result and "SECRET_CONTENT" not in result
    assert executor.execute(get_tool("read_file"), {"file_path": "alias.txt"}).status is ToolStatus.DENIED


def test_bash_status_is_not_inferred_from_stdout(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="Error: example\n[exit code: 9]\ntimed out", stderr="",
    ))
    result = get_tool("bash").execute_in("echo example", cwd=str(tmp_path))
    assert result.status is ToolStatus.COMPLETED
    assert _tool_status(result) == "completed"
    assert _tool_status(str(result), "completed") == "completed"


def test_timeout_after_effect_is_unknown(tmp_path, monkeypatch):
    target = tmp_path / "effect.txt"

    def timeout(*args, **kwargs):
        target.write_text("already happened", encoding="utf-8")
        raise subprocess.TimeoutExpired("command", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = get_tool("bash").execute_in("echo test", cwd=str(tmp_path), timeout=1)
    assert result.status is ToolStatus.EFFECT_UNKNOWN
    assert "Inspect effects before retrying" in result
    assert target.exists()


def test_partial_write_failure_is_unknown(tmp_path, monkeypatch):
    target = tmp_path / "x.py"
    target.write_text("before", encoding="utf-8")

    def partial(proposal):
        target.write_text("partial", encoding="utf-8")
        raise OSError("disk failure")

    monkeypatch.setattr(FileWriteProposal, "apply", partial)
    manager = PermissionManager(ChatPermissionPolicy(), SimpleNamespace(decide=lambda request: PermissionDecision.allow("yes")))
    result = RepositoryToolExecutor(tmp_path, manager).execute(get_tool("edit_file"), {
        "file_path": "x.py", "old_string": "before", "new_string": "after",
    })
    assert result.status is ToolStatus.EFFECT_UNKNOWN
    assert target.read_text() == "partial"


@pytest.mark.parametrize("parallel", [False, True])
def test_status_survives_events_session_replay_and_ui(tmp_path, monkeypatch, parallel):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="Error: this is successful command output", stderr="",
    ))
    calls = [ToolCall("ok", "bash", {"command": "echo example"})]
    if parallel:
        calls.append(ToolCall("bad", "read_file", {"file_path": "x", "limit": False}))
    responses = iter([LLMResponse(tool_calls=calls), LLMResponse(content="done")])
    llm = SimpleNamespace(chat=lambda *args, **kwargs: next(responses))
    events = []
    store = SessionStore(tmp_path / "sessions")
    store.create("s1", repository_root=tmp_path, model="fake")
    sink = SessionEventSink(store, CallbackEventSink(events.append))
    agent = Agent(llm=llm, tools=[get_tool("bash"), get_tool("read_file")], event_sink=sink, session_id="s1")
    assert agent.chat("test") == "done"
    completions = [e for e in events if e.event_type is RuntimeEventType.TOOL_COMPLETED]
    assert completions[0].payload["tool_status"] == "completed"
    if parallel:
        assert completions[1].payload["tool_status"] == "error"
    projection = store.replay("s1")
    details = _tool_call_details(projection.events)
    assert details[0].status == "completed"
    tui = TechPilotTui()
    for event in events:
        if event.event_type in {RuntimeEventType.TOOL_REQUESTED, RuntimeEventType.TOOL_COMPLETED}:
            tui._apply_event(event)
    assert tui._tool_activities["ok"].status == "completed"
    assert json.loads(json.dumps(agent.messages)) == agent.messages


def test_legacy_result_compatibility_and_explicit_unknown():
    assert _tool_status("Permission denied edit_file") == "denied"
    assert _tool_status("Invalid regex: invalid pattern") == "error"
    assert _tool_status("text\n[exit code: 0]") == "completed"
    assert _tool_status("text\n[exit code: 3]") == "error"
    assert _tool_status(ToolResult("ambiguous", ToolStatus.EFFECT_UNKNOWN)) == "effect unknown"
    assert _tool_status("[cancelled]") == "effect unknown"
    assert _tool_status("[cancelled before execution]") == "not executed"
    assert _tool_status("[not executed: an earlier call stopped this turn]") == "not executed"


def test_non_object_arguments_remain_recoverable_in_multi_call_round(tmp_path):
    responses = iter([
        LLMResponse(tool_calls=[
            ToolCall("bad", "read_file", []),
            ToolCall("good", "now", {}),
        ]),
        LLMResponse(content="corrected"),
    ])
    events = []
    agent = Agent(
        llm=SimpleNamespace(chat=lambda *a, **k: next(responses)),
        tools=[get_tool("read_file"), get_tool("now")],
        tool_executor=RepositoryToolExecutor(tmp_path),
        event_sink=CallbackEventSink(events.append),
    )
    assert agent.chat("test") == "corrected"
    completed = [e for e in events if e.event_type is RuntimeEventType.TOOL_COMPLETED]
    assert [e.payload["tool_status"] for e in completed] == ["error", "completed"]


def test_unknown_effect_is_persisted_and_not_rendered_as_success(tmp_path):
    class UncertainTool:
        name = "uncertain"
        description = "Return an explicitly unknown effect for replay testing."
        parameters = {"properties": {}}

        def schema(self):
            return {"type": "function", "function": {"name": self.name, "parameters": self.parameters}}

        def execute(self):
            return ToolResult("[effect unknown] inspect before retry", ToolStatus.EFFECT_UNKNOWN)

    responses = iter([LLMResponse(tool_calls=[ToolCall("u1", "uncertain", {})]), LLMResponse(content="inspect")])
    store = SessionStore(tmp_path / "sessions")
    store.create("unknown", repository_root=tmp_path, model="fake")
    events = []
    agent = Agent(
        llm=SimpleNamespace(chat=lambda *a, **k: next(responses)), tools=[UncertainTool()], session_id="unknown",
        event_sink=SessionEventSink(store, CallbackEventSink(events.append)),
    )
    agent.chat("test")
    assert _tool_call_details(store.replay("unknown").events)[0].status == "effect unknown"
    tui = TechPilotTui()
    for event in events:
        if event.event_type in {RuntimeEventType.TOOL_REQUESTED, RuntimeEventType.TOOL_COMPLETED}:
            tui._apply_event(event)
    assert tui._tool_activities["u1"].status == "effect unknown"


def test_sensitive_alias_filter_with_resolved_target(tmp_path, monkeypatch):
    from techpilot.safety.paths import is_safe_search_path, is_sensitive_path

    secret = tmp_path / ".env"
    secret.write_text("private", encoding="utf-8")
    alias = tmp_path / "ordinary.txt"
    alias.write_text("placeholder", encoding="utf-8")
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        return secret if path == alias else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert is_sensitive_path(alias)
    assert not is_safe_search_path(alias, tmp_path)
