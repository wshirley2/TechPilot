"""G1 acceptance checks for durable Goal plans and recovery views."""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from techpilot.chat.session import ChatSession, TerminalEventSink
from techpilot.engine.llm import LLMResponse
from techpilot.runtime import (
    CompletionCriterionKind,
    LongTaskEvent,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskStore,
    RuntimeBootstrap,
    RuntimeBootstrapInput,
)
from techpilot.runtime.sessions import SessionStore


def _step(step_id: str, dependencies: list[str] | None = None) -> dict:
    return {
        "id": step_id,
        "goal": f"Finish {step_id}",
        "dependencies": dependencies or [],
        "expected_output": f"{step_id} result",
        "work_scope": ["README.md"],
        "evidence": f"{step_id} evidence",
        "check": f"Review {step_id} evidence",
        "next_action": f"Work on {step_id}",
    }


def _document(*steps: dict, reason: str = "Initial stage split") -> dict:
    return {"reason": reason, "steps": list(steps)}


def _store(root) -> LongTaskStore:
    SessionStore.for_repository(root).create("chat-session", repository_root=root, model="fake-model")
    store = LongTaskStore.for_repository(root)
    store.create(
        "goal-1",
        goal="Complete the fixture",
        repository_root=root,
        session_id="chat-session",
        initial_status=LongTaskStatus.SETUP,
    )
    return store


def test_goal_plan_versions_and_selected_stage_survive_replay(tmp_path):
    store = _store(tmp_path)
    first = store.record_goal_plan("goal-1", _document(_step("inspect"), _step("fix", ["inspect"])))
    assert first.plan.version == 1
    assert first.ready_step_ids == ("inspect",)

    with pytest.raises(LongTaskStateError, match="unverified dependencies"):
        store.select_goal_step("goal-1", "fix")
    store.record_completion_criterion(
        "goal-1", kind=CompletionCriterionKind.FILE_EXISTS, path="README.md"
    )
    with pytest.raises(LongTaskStateError, match="Select a ready"):
        store.start("goal-1", reason="start before selecting")

    store.select_goal_step("goal-1", "inspect")
    revised = store.record_goal_plan(
        "goal-1", _document(_step("inspect"), _step("fix", ["inspect"]), _step("report", ["fix"]), reason="Add handoff")
    )
    assert [plan.version for plan in revised.plans] == [1, 2]
    assert revised.selected_step_id == "inspect"
    assert revised.plan.reason == "Add handoff"

    changed = _step("inspect")
    changed["goal"] = "Silently replace selected work"
    with pytest.raises(LongTaskStateError, match="retain the selected step unchanged"):
        store.record_goal_plan("goal-1", _document(changed, _step("fix", ["inspect"]), reason="Bad revision"))

    replayed = LongTaskStore.for_repository(tmp_path).replay("goal-1")
    assert [event.event_type for event in replayed.events if event.event_type.startswith("plan_")] == [
        "plan_recorded", "plan_revised"
    ]
    assert replayed.plan.version == 2
    assert replayed.selected_step_id == "inspect"
    assert replayed.ready_step_ids == ()


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        ([_step("inspect", ["missing"])], "missing dependencies"),
        ([_step("one", ["two"]), _step("two", ["one"])], "cycle"),
        ([_step("same"), _step("same")], "unique"),
        ([_step(f"s{index}") for index in range(13)], "1–12 steps"),
    ],
)
def test_goal_plan_rejects_invalid_dependency_graph_without_appending(tmp_path, steps, error):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match=error):
        store.record_goal_plan("goal-1", _document(*steps))
    assert [event.event_type for event in store.replay("goal-1").events] == ["task_created"]


def test_goal_plan_checkpoint_requires_evaluation_and_bad_event_fails_closed(tmp_path):
    store = _store(tmp_path)
    store.record_goal_plan("goal-1", _document(_step("inspect"), _step("parallel"), _step("fix", ["inspect"])))
    store.select_goal_step("goal-1", "inspect")
    store.record_completion_criterion("goal-1", kind=CompletionCriterionKind.FILE_EXISTS, path="README.md")
    store.start("goal-1", reason="begin selected stage")
    session_cursor = SessionStore.for_repository(tmp_path).replay("chat-session").events[-1].event_id
    store.checkpoint(
        "goal-1",
        message_projection=[],
        recovery_reason="one controlled turn finished",
        session_event_cursor=session_cursor,
    )
    reopened = LongTaskStore.for_repository(tmp_path)
    projected = reopened.replay("goal-1")
    assert projected.selected_step_needs_evaluation
    assert projected.ready_step_ids == ()
    with pytest.raises(LongTaskStateError, match="needs evaluation"):
        reopened.select_goal_step("goal-1", "parallel")

    reopened.append(LongTaskEvent(task_id="goal-1", event_type="plan_revised", payload={"version": 9}))
    corrupt = reopened.replay("goal-1")
    assert corrupt.status is LongTaskStatus.RECOVERY_REQUIRED
    assert corrupt.recovery_blockers == ["goal_plan_event_invalid"]
    with pytest.raises(LongTaskStateError, match="cannot change"):
        reopened.record_goal_plan("goal-1", _document(_step("inspect"), reason="unsafe"))


class _Provider:
    model = "fake-goal-provider"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict] = []

    def chat(self, messages, tools=None, on_token=None):
        self.requests.append({"messages": messages, "tools": tools})
        response = next(self.responses)
        if on_token and response.content:
            on_token(response.content)
        return response


def _runtime(repository, provider, console, session_directory, *, resume_session_id=None):
    return RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=repository,
        event_sink=TerminalEventSink(console),
        session_directory=session_directory,
        resume_session_id=resume_session_id,
    ))


def test_chat_goal_plan_resume_shows_same_version_and_next_action(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    plan_file = repository / "goal-plan.json"
    plan_file.write_text(json.dumps(_document(_step("inspect"), _step("fix", ["inspect"]))), encoding="utf-8")
    session_directory = tmp_path / "sessions"
    first_output = StringIO()
    first_console = Console(file=first_output, force_terminal=False, color_system=None, width=180)
    first_provider = _Provider([LLMResponse(content="inspection recorded")])
    first = _runtime(repository, first_provider, first_console, session_directory)
    first_chat = ChatSession(first, console=first_console)
    first_chat._start_goal("Inspect and fix the fixture")
    task_id = first_chat._goal_workflow.task_id
    first_chat._goal("criterion file-exists README.md")
    first_chat._goal("plan load goal-plan.json")
    first_chat._goal("plan select inspect")
    first_chat._goal("start")
    first_chat._goal("pause")
    assert len(first_provider.requests) == 1

    other_output = StringIO()
    other_console = Console(file=other_output, force_terminal=False, color_system=None)
    other = _runtime(repository, _Provider([]), other_console, session_directory)
    ChatSession(other, console=other_console)._goal(f"resume {task_id}")
    assert "属于另一个 Session" in other_output.getvalue()

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=180)
    resumed_provider = _Provider([])
    resumed = _runtime(repository, resumed_provider, console, session_directory, resume_session_id=first.agent.session_id)
    chat = ChatSession(resumed, console=console)
    chat._goal(f"resume {task_id}")
    rendered = output.getvalue()
    assert "Stage plan" in rendered and "v1 (2 stages)" in rendered
    assert "inspect (needs_evaluation)" in rendered
    assert "fix ← inspect" in rendered
    assert "Review inspect evidence using Review inspect evidence" in rendered
    with pytest.raises(LongTaskStateError, match="needs evaluation"):
        chat._run_user_turn("continue the same stage")
    assert resumed_provider.requests == []

    old_output = StringIO()
    old_console = Console(file=old_output, force_terminal=False, color_system=None)
    old = _runtime(repository, _Provider([]), old_console, session_directory)
    old_chat = ChatSession(old, console=old_console)
    old_chat._start_goal("Legacy Goal without a plan")
    old_chat._show_goal_status()
    assert "unplanned" in old_output.getvalue()


def test_goal_plan_status_survives_a_real_process_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    (repository / "goal-plan.json").write_text(
        json.dumps(_document(_step("inspect"), _step("fix", ["inspect"]))), encoding="utf-8"
    )
    session_directory = tmp_path / "sessions"
    child_code = """
import json
import sys
from io import StringIO
from pathlib import Path
from rich.console import Console
from techpilot.chat.session import ChatSession, TerminalEventSink
from techpilot.engine.llm import LLMResponse
from techpilot.runtime import RuntimeBootstrap, RuntimeBootstrapInput

class Provider:
    model = 'fake-goal-provider'
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def chat(self, messages, tools=None, on_token=None):
        if on_token:
            on_token('inspection recorded')
        return LLMResponse(content='inspection recorded')

repository = Path(sys.argv[1])
console = Console(file=StringIO(), force_terminal=False, color_system=None)
runtime = RuntimeBootstrap(provider_factory=lambda _config: Provider()).build(RuntimeBootstrapInput(
    repository=repository,
    event_sink=TerminalEventSink(console),
    session_directory=Path(sys.argv[2]),
))
chat = ChatSession(runtime, console=console)
chat._start_goal('Inspect and fix the fixture')
chat._goal('criterion file-exists README.md')
chat._goal('plan load goal-plan.json')
chat._goal('plan select inspect')
chat._goal('start')
chat._goal('pause')
print(json.dumps({'session_id': runtime.agent.session_id, 'task_id': chat._goal_workflow.task_id}))
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(repository), str(session_directory)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    identities = json.loads(child.stdout)
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=180)
    resumed = _runtime(
        repository, _Provider([]), console, session_directory, resume_session_id=identities["session_id"]
    )
    ChatSession(resumed, console=console)._goal(f"resume {identities['task_id']}")
    rendered = output.getvalue()
    assert "v1 (2 stages)" in rendered
    assert "inspect (needs_evaluation)" in rendered
    assert "fix ← inspect" in rendered
