"""C3 permission protocol, Trusted Diff, and Chat policy tests."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from techpilot.chat.executor import RepositoryToolExecutor
from techpilot.chat.permissions import (
    ChatPermissionPolicy,
    TerminalPermissionPrompt,
    command_effect,
    command_prefix,
    command_tokens,
    is_concurrency_safe_read_command,
)
from techpilot.engine.permissions import (
    PermissionAction,
    PermissionDecision,
    PermissionEffect,
    PermissionGrantScope,
    PermissionManager,
    PermissionRequest,
)
from techpilot.engine.tools.bash import BashTool
from techpilot.engine.tools.edit import EditFileTool
from techpilot.engine.tools.write import WriteFileTool


class AskPolicy:
    def decide(self, request):
        del request
        return PermissionDecision.ask("test asks")


class RecordingPrompt:
    def __init__(self, *decisions, on_decide=None):
        self.decisions = iter(decisions)
        self.requests = []
        self.on_decide = on_decide

    def decide(self, request):
        self.requests.append(request)
        if self.on_decide:
            self.on_decide(request, len(self.requests))
        return next(self.decisions)


class CapturingBash(BashTool):
    def __init__(self):
        self.commands = []

    def execute_in(self, command: str, *, cwd: str, timeout: int = 120) -> str:
        self.commands.append((command, cwd, timeout))
        return f"ran: {command}"


def _request(
    *,
    tool_name="write_file",
    effect=PermissionEffect.WRITE,
    scope="notes.txt",
    arguments=None,
    tokens=(),
    prefix=(),
):
    return PermissionRequest(
        tool_call_id="call-1",
        tool_name=tool_name,
        effect=effect,
        normalized_arguments=arguments or {"file_path": scope, "content": "safe"},
        reason="test",
        scope=scope,
        command_tokens=tokens,
        command_prefix=prefix,
    )


def test_permission_request_keeps_runtime_arguments_immutable():
    arguments = {"file_path": "notes.txt", "content": "approved", "metadata": {"safe": True}}
    request = _request(arguments=arguments)
    arguments["content"] = "changed outside"

    assert request.normalized_arguments["content"] == "approved"
    with pytest.raises(TypeError):
        request.normalized_arguments["content"] = "prompt replacement"
    with pytest.raises(TypeError):
        request.normalized_arguments["metadata"]["safe"] = False


def test_permission_manager_reuses_session_and_prefix_grants_but_force_reprompts():
    prompt = RecordingPrompt(
        PermissionDecision.allow("session", PermissionGrantScope.SESSION),
        PermissionDecision.allow("prefix", PermissionGrantScope.PREFIX),
        PermissionDecision.deny("snapshot changed"),
    )
    manager = PermissionManager(AskPolicy(), prompt)
    file_request = _request()

    assert manager.authorize(file_request).action is PermissionAction.ALLOW
    assert manager.authorize(file_request).grant_scope is PermissionGrantScope.SESSION
    assert len(prompt.requests) == 1

    tokens = command_tokens("echo first")
    command_request = _request(
        tool_name="bash",
        effect=PermissionEffect.EXECUTE,
        scope="echo first",
        arguments={"command": "echo first"},
        tokens=tokens,
        prefix=command_prefix(tokens),
    )
    assert manager.authorize(command_request).action is PermissionAction.ALLOW
    followup_tokens = command_tokens("echo second")
    followup = _request(
        tool_name="bash",
        effect=PermissionEffect.EXECUTE,
        scope="echo second",
        arguments={"command": "echo second"},
        tokens=followup_tokens,
        prefix=command_prefix(followup_tokens),
    )
    assert manager.authorize(followup).grant_scope is PermissionGrantScope.PREFIX
    assert len(prompt.requests) == 2

    assert manager.authorize(file_request, force_prompt=True).action is PermissionAction.DENY
    assert len(prompt.requests) == 3


def test_write_is_denied_without_interactive_prompt_and_never_reaches_disk(tmp_path):
    executor = RepositoryToolExecutor(tmp_path)
    target = tmp_path / "new.txt"

    result = executor.execute(WriteFileTool(), {"file_path": "new.txt", "content": "candidate\n"})

    assert result.startswith("Permission denied write_file")
    assert not target.exists()
    assert executor.consume_turn_stop_message() is None


def test_unregistered_network_or_unknown_tools_are_not_misclassified_as_reads(tmp_path):
    executor = RepositoryToolExecutor(tmp_path)

    class NeverRun:
        name = "fetch_url"

        def execute(self, **kwargs):
            raise AssertionError("disabled network tool must not execute")

    result = executor.execute(NeverRun(), {"url": "https://example.com"})

    assert result.startswith("Permission denied fetch_url")


def test_new_empty_file_still_has_a_reviewable_trusted_diff(tmp_path):
    prompt = RecordingPrompt(PermissionDecision.deny("reviewed"))
    executor = RepositoryToolExecutor(
        tmp_path,
        PermissionManager(ChatPermissionPolicy(), prompt),
    )

    result = executor.execute(
        WriteFileTool(),
        {"file_path": "empty.txt", "content": ""},
    )

    assert result == "Permission denied write_file: reviewed"
    assert prompt.requests[0].trusted_preview == "--- /dev/null\n+++ b/empty.txt\n"
    assert not (tmp_path / "empty.txt").exists()


def test_trusted_diff_is_built_from_real_content_before_allowing_edit(tmp_path):
    target = tmp_path / "app.py"
    target.write_bytes(b"VALUE = 'old'\r\n")

    def assert_pre_write(request, call_number):
        assert call_number == 1
        assert target.read_bytes() == b"VALUE = 'old'\r\n"
        assert "-VALUE = 'old'" in request.trusted_preview
        assert "+VALUE = 'new'" in request.trusted_preview
        assert request.source_snapshot

    prompt = RecordingPrompt(PermissionDecision.allow("approved"), on_decide=assert_pre_write)
    manager = PermissionManager(ChatPermissionPolicy(), prompt)
    executor = RepositoryToolExecutor(tmp_path, manager)

    result = executor.execute(
        EditFileTool(),
        {"file_path": "app.py", "old_string": "old", "new_string": "new"},
    )

    assert result.startswith("Edited app.py")
    assert target.read_bytes() == b"VALUE = 'new'\r\n"
    assert len(prompt.requests) == 1


def test_snapshot_change_regenerates_diff_and_requires_new_approval(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("VALUE = old\nMARKER = base\n", encoding="utf-8")

    def mutate_after_first_preview(request, call_number):
        if call_number == 1:
            assert "MARKER = base" in request.trusted_preview
            target.write_text("VALUE = old\nMARKER = external\n", encoding="utf-8")

    prompt = RecordingPrompt(
        PermissionDecision.allow("first approval"),
        PermissionDecision.deny("reject regenerated diff"),
        on_decide=mutate_after_first_preview,
    )
    executor = RepositoryToolExecutor(
        tmp_path,
        PermissionManager(ChatPermissionPolicy(), prompt),
    )

    result = executor.execute(
        EditFileTool(),
        {"file_path": "app.py", "old_string": "old", "new_string": "new"},
    )

    assert result == "Permission denied edit_file: reject regenerated diff"
    assert target.read_text(encoding="utf-8") == "VALUE = old\nMARKER = external\n"
    assert len(prompt.requests) == 2
    assert "MARKER = external" in prompt.requests[1].trusted_preview
    assert "Source changed" in prompt.requests[1].reason


def test_file_session_grant_allows_later_write_to_same_path(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("one\n", encoding="utf-8")
    prompt = RecordingPrompt(
        PermissionDecision.allow("session", PermissionGrantScope.SESSION),
    )
    executor = RepositoryToolExecutor(
        tmp_path,
        PermissionManager(ChatPermissionPolicy(), prompt),
    )

    first = executor.execute(
        EditFileTool(),
        {"file_path": "notes.txt", "old_string": "one", "new_string": "two"},
    )
    second = executor.execute(
        EditFileTool(),
        {"file_path": "notes.txt", "old_string": "two", "new_string": "three"},
    )

    assert first.startswith("Edited") and second.startswith("Edited")
    assert target.read_text(encoding="utf-8") == "three\n"
    assert len(prompt.requests) == 1


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status --short", PermissionAction.ALLOW),
        ("dir", PermissionAction.ALLOW),
        ("ls -la", PermissionAction.ALLOW),
        ("git diff --output=patch.txt", PermissionAction.ASK),
        ("python -m pytest -q", PermissionAction.ALLOW),
        ("python -m ruff check .", PermissionAction.ALLOW),
        ("python -m pip install example", PermissionAction.ASK),
        ("curl https://example.com", PermissionAction.ASK),
        ("Remove-Item notes.txt", PermissionAction.ASK),
        ("rm -rf ./build", PermissionAction.DENY),
        ("Remove-Item ./build -Recurse -Force", PermissionAction.DENY),
        ("del /s /q C:\\important", PermissionAction.DENY),
        ("git reset --hard", PermissionAction.DENY),
    ],
)
def test_chat_command_policy_is_code_driven(command, expected):
    tokens = command_tokens(command)
    request = _request(
        tool_name="bash",
        effect=command_effect(command),
        scope=command,
        arguments={"command": command},
        tokens=tokens,
        prefix=command_prefix(tokens),
    )

    assert ChatPermissionPolicy().decide(request).action is expected


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git status --short --branch",
        "git diff --stat --cached",
        "git rev-parse --show-toplevel",
        "git branch --show-current",
    ],
)
def test_concurrency_safe_command_classifier_accepts_only_literal_git_inspection(command):
    assert is_concurrency_safe_read_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "pytest",
        "rg tool_execution",
        "git status src",
        "git branch -Dfeature",
        "git diff --no-index left right",
        "git status && whoami",
        "git status > status.txt",
        "git status %COMSPEC%",
    ],
)
def test_concurrency_safe_command_classifier_fails_closed(command):
    assert not is_concurrency_safe_read_command(command)


def test_command_prefix_grant_executes_later_matching_command_without_prompt(tmp_path):
    prompt = RecordingPrompt(
        PermissionDecision.allow("prefix", PermissionGrantScope.PREFIX),
    )
    executor = RepositoryToolExecutor(
        tmp_path,
        PermissionManager(ChatPermissionPolicy(), prompt),
    )
    bash = CapturingBash()

    assert executor.execute(bash, {"command": "echo first"}) == "ran: echo first"
    assert executor.execute(bash, {"command": "echo second"}) == "ran: echo second"
    assert [item[0] for item in bash.commands] == ["echo first", "echo second"]
    assert len(prompt.requests) == 1


def test_command_prefix_grant_never_reaches_a_later_blocked_compound_command(tmp_path):
    prompt = RecordingPrompt(
        PermissionDecision.allow("prefix", PermissionGrantScope.PREFIX),
    )
    executor = RepositoryToolExecutor(
        tmp_path,
        PermissionManager(ChatPermissionPolicy(), prompt),
    )
    bash = CapturingBash()

    assert executor.execute(bash, {"command": "echo first"}) == "ran: echo first"
    result = executor.execute(bash, {"command": "echo second && del notes.txt"})

    assert result.startswith("Policy denied bash: 该操作已被阻断，未执行")
    assert [item[0] for item in bash.commands] == ["echo first"]
    assert len(prompt.requests) == 1


def test_terminal_permission_prompt_is_independently_testable():
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    prompt = TerminalPermissionPrompt(console, input_fn=lambda _: "y")
    request = PermissionRequest(
        tool_call_id="write-1",
        tool_name="write_file",
        effect=PermissionEffect.WRITE,
        normalized_arguments={"file_path": "notes.txt", "content": "new\n"},
        reason="review write",
        scope="notes.txt",
        trusted_preview="--- a/notes.txt\n+++ b/notes.txt\n",
        source_snapshot="snapshot",
    )

    decision = prompt.decide(request)

    assert decision.action is PermissionAction.ALLOW
    rendered = output.getvalue()
    assert "需要你的确认" in rendered
    assert "--- a/notes.txt" in rendered
    assert "1. 仅允许这一次" in rendered
    assert "2. 本会话内允许此范围" in rendered
    assert "0. 拒绝（默认；直接回车也拒绝）" in rendered


def test_terminal_permission_prompt_accepts_numbered_session_and_prefix_choices():
    write_request = PermissionRequest(
        tool_call_id="write-1",
        tool_name="write_file",
        effect=PermissionEffect.WRITE,
        normalized_arguments={"file_path": "notes.txt", "content": "new\n"},
        reason="review write",
        scope="notes.txt",
        source_snapshot="snapshot",
    )
    command_request = PermissionRequest(
        tool_call_id="bash-1",
        tool_name="bash",
        effect=PermissionEffect.EXECUTE,
        normalized_arguments={"command": "python -m pip install example"},
        reason="review command",
        scope="python -m pip install example",
        command_tokens=("python", "-m", "pip", "install", "example"),
        command_prefix=("python", "-m", "pip", "install"),
    )

    session = TerminalPermissionPrompt(input_fn=lambda _: "2").decide(write_request)
    prefix = TerminalPermissionPrompt(input_fn=lambda _: "3").decide(command_request)

    assert session.grant_scope is PermissionGrantScope.SESSION
    assert prefix.grant_scope is PermissionGrantScope.PREFIX


def test_managed_policy_exposes_the_same_permission_decision_protocol():
    from techpilot.execution import PolicyDecision, ToolEffect

    decision = PolicyDecision(PermissionAction.DENY, "managed denial", ToolEffect.NETWORK)

    assert decision.to_permission_decision() == PermissionDecision.deny(decision.reason)
