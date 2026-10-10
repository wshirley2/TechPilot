"""C2 RuntimeBootstrap and minimal event-driven CLI Chat tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.document import Document
from rich.console import Console

from techpilot.chat.executor import RepositoryToolExecutor, _normalized_command
from techpilot.chat.session import ChatSession, SlashCommandCompleter, TerminalEventSink
from techpilot.cli import _normalize_command
from techpilot.engine.events import RuntimeEvent, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.tool_execution import ToolEffect
from techpilot.engine.tools.glob_tool import GlobTool
from techpilot.engine.tools.grep import GrepTool
from techpilot.runtime import (
    GOAL_COMPLETION_TOOL_NAME,
    ActiveRole,
    ChatRuntime,
    CompletionCriterionKind,
    EffectJournalState,
    EffectJournalStore,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskStore,
    LongTaskWorkflow,
    RuntimeBootstrap,
    RuntimeBootstrapInput,
    TaskRuntime,
)
from techpilot.runtime.sessions import SessionEvent, SessionStore
from techpilot.safety.paths import ignored_child_names

BENCHMARK_ROOT = Path(__file__).parents[2] / "benchmarks" / "cli_data_tool"


def goal_setup_response(*, constraints=(), criteria=None, plan=None):
    return LLMResponse(tool_calls=[ToolCall("setup-proposal", "propose_goal_setup", {
        "summary": "检查目标产物及运行证据",
        "constraints": list(constraints),
        "criteria": criteria or [{"kind": "file_exists", "path": "src/cli_data_tool/cli.py"}],
        "plan": plan,
    })])


def copy_benchmark(destination: Path) -> Path:
    """Copy only source-controlled benchmark inputs, not local runtime artifacts."""

    return Path(shutil.copytree(
        BENCHMARK_ROOT,
        destination,
        ignore=lambda _directory, names: ignored_child_names(names),
    ))


def test_slash_command_completer_lists_and_filters_local_commands():
    completer = SlashCommandCompleter()

    all_commands = [item.text for item in completer.get_completions(Document("/"), None)]
    assert "/help" in all_commands
    assert "/status" in all_commands
    assert "/goal" in all_commands
    assert "/exit" in all_commands

    status_commands = [item.text for item in completer.get_completions(Document("/s"), None)]
    assert "/status" in status_commands
    assert "/sessions" in status_commands
    assert "/help" not in status_commands


def test_slash_command_completer_does_not_expose_plan_in_chat():
    assert "/plan" not in [item.text for item in SlashCommandCompleter().get_completions(Document("/"), None)]


def test_chat_goal_keeps_multiple_steps_in_one_durable_task_and_completes(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    provider = FakeProvider([
        goal_setup_response(),
        LLMResponse(content="验收约定已提出"),
        LLMResponse(tool_calls=[ToolCall("first-evidence", "glob", {"pattern": "**/*.py"})]),
        LLMResponse(content="first Goal step"),
        LLMResponse(tool_calls=[ToolCall("completion-evidence", "glob", {"pattern": "**/*.py"})]),
        LLMResponse(tool_calls=[ToolCall(
            "completion-proposal",
            GOAL_COMPLETION_TOOL_NAME,
            {"summary": "project sources inspected", "evidence_tool_call_ids": ["completion-evidence"]},
        )]),
        LLMResponse(content="second Goal step"),
    ])
    runtime = make_runtime(repository, provider, console, session_directory=tmp_path / "sessions")
    inputs = iter([
        "/goal inspect the project in two steps",
        "确认",
        "/goal",
        "/exit",
    ])

    assert ChatSession(runtime, console=console, input_fn=lambda _prompt: next(inputs)).run() == 0

    task_directories = list((repository / ".techpilot" / "tasks").glob("goal-*"))
    assert len(task_directories) == 1
    task = LongTaskStore.for_repository(repository, session_directory=tmp_path / "sessions").replay(task_directories[0].name)
    assert task.status is LongTaskStatus.SUCCEEDED
    assert len(task.checkpoints) == 2
    assert task.session_id == runtime.agent.session_id
    rendered = output.getvalue()
    assert "Goal setup created:" in rendered
    assert "Goal started:" in rendered
    assert "Goal completed:" in rendered
    assert "second Goal step" in rendered
    assert "Goal 验收约定（待确认）" in rendered
    assert "file_exists" in rendered
    assert {tool["function"]["name"] for tool in provider.requests[0]["tools"]}.isdisjoint({"bash", "write_file", "edit_file"})


def test_chat_goal_complete_requires_a_verified_completion_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = make_runtime(
        copy_benchmark(tmp_path / "repository"),
        FakeProvider([goal_setup_response(), LLMResponse(content="约定已提出"), LLMResponse(content="Goal step without evidence")]),
        console,
        session_directory=tmp_path / "sessions",
    )
    chat = ChatSession(runtime, console=console)
    chat._start_goal("inspect the project")
    chat._run_user_turn("确认")

    assert chat._goal_workflow is not None
    rendered = output.getvalue()
    assert "没有新的工具证据" in rendered
    assert chat._goal_workflow.store.replay(chat._goal_workflow.task_id).status is LongTaskStatus.RUNNING


def test_running_goal_change_requires_confirmation_and_survives_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    provider = FakeProvider([
        goal_setup_response(), LLMResponse(content="约定已提出"),
        LLMResponse(content="first Goal step without evidence"),
        LLMResponse(tool_calls=[ToolCall("change-1", "propose_goal_change", {
            "kind": "constraint_add", "summary": "只修改 calculator.py",
            "data": {"text": "Only modify calculator.py"},
        })]),
        LLMResponse(content="change proposed"),
    ])
    runtime = make_runtime(repository, provider, console, session_directory=tmp_path / "sessions")
    chat = ChatSession(runtime, console=console)
    chat._start_goal("Inspect project")
    chat._run_user_turn("确认")
    workflow = chat._goal_workflow
    assert workflow is not None

    response = chat._run_user_turn("以后只能改 calculator.py")
    assert "待确认 Goal 修订" in response
    pending = workflow.store.replay(workflow.task_id)
    assert pending.pending_goal_change is not None
    assert pending.active_constraints == ()
    assert {tool["function"]["name"] for tool in provider.requests[3]["tools"]}.isdisjoint({
        "bash", "write_file", "edit_file",
    })

    resumed_provider = FakeProvider([])
    resumed_runtime = make_runtime(
        repository, resumed_provider, console, session_directory=tmp_path / "sessions",
        resume_session_id=runtime.agent.session_id,
    )
    resumed_chat = ChatSession(resumed_runtime, console=console)
    resumed_chat._goal(f"resume {workflow.task_id}")
    assert resumed_provider.requests == []
    assert "不会自动续跑" in output.getvalue()
    assert "修订已生效" in resumed_chat._run_user_turn("确认")
    assert [item.text for item in workflow.store.replay(workflow.task_id).active_constraints] == [
        "Only modify calculator.py",
    ]
    assert resumed_provider.requests == []


def test_unclassified_goal_input_blocks_resume_until_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    console = Console(file=StringIO(), force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    runtime = make_runtime(
        repository,
        FakeProvider([goal_setup_response(), LLMResponse(content="约定已提出"), LLMResponse(content="first step")]),
        console,
        session_directory=tmp_path / "sessions",
    )
    chat = ChatSession(runtime, console=console)
    chat._start_goal("Inspect project")
    chat._run_user_turn("确认")
    workflow = chat._goal_workflow
    assert workflow is not None
    workflow.store.begin_goal_input(workflow.task_id, "Do not use network")

    resumed_provider = FakeProvider([])
    resumed_runtime = make_runtime(
        repository, resumed_provider, console, session_directory=tmp_path / "sessions",
        resume_session_id=runtime.agent.session_id,
    )
    resumed_chat = ChatSession(resumed_runtime, console=console)
    resumed_chat._goal(f"resume {workflow.task_id}")
    assert resumed_provider.requests == []
    assert "尚未完成只读分类" in resumed_chat._run_user_turn("继续工作")
    assert "已取消" in resumed_chat._run_user_turn("取消")
    assert workflow.store.replay(workflow.task_id).pending_goal_input is None


def test_running_goal_continuation_clears_read_only_preflight_before_work(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    console = Console(file=StringIO(), force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    provider = FakeProvider([
        goal_setup_response(), LLMResponse(content="setup proposed"),
        LLMResponse(content="first Goal step without evidence"),
        LLMResponse(tool_calls=[ToolCall("classify-continue", "propose_goal_change", {
            "kind": "continue", "summary": "Continue work", "data": {},
        })]),
        LLMResponse(content="classified"),
        LLMResponse(content="continued Goal work"),
    ])
    runtime = make_runtime(repository, provider, console, session_directory=tmp_path / "sessions")
    chat = ChatSession(runtime, console=console)
    chat._start_goal("Inspect project")
    chat._run_user_turn("确认")
    workflow = chat._goal_workflow
    assert workflow is not None

    assert chat._run_user_turn("继续检查项目") == "continued Goal work"
    task = workflow.store.replay(workflow.task_id)
    assert task.pending_goal_input is None
    assert task.pending_goal_change is None
    assert [event.event_type for event in task.events][-2:] == [
        "goal_input_resolved", "checkpoint_recorded",
    ]
    assert len(provider.requests) == 6


@pytest.mark.parametrize("continuation", ["继续", "continue"])
def test_exact_goal_continuation_uses_one_model_turn_without_preflight(tmp_path, monkeypatch, continuation):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    console = Console(file=StringIO(), force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    provider = FakeProvider([
        goal_setup_response(), LLMResponse(content="setup proposed"),
        LLMResponse(content="first Goal step without evidence"),
        LLMResponse(content="continued Goal work"),
    ])
    runtime = make_runtime(repository, provider, console, session_directory=tmp_path / "sessions")
    chat = ChatSession(runtime, console=console)
    chat._start_goal("Inspect project")
    chat._run_user_turn("确认")
    workflow = chat._goal_workflow
    assert workflow is not None

    assert chat._run_user_turn(continuation) == "continued Goal work"
    task = workflow.store.replay(workflow.task_id)
    assert task.pending_goal_input is None
    assert not any(event.event_type == "goal_input_started" for event in task.events)
    assert len(provider.requests) == 4


def test_goal_continuation_with_a_new_rule_still_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    console = Console(file=StringIO(), force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    provider = FakeProvider([
        goal_setup_response(), LLMResponse(content="setup proposed"),
        LLMResponse(content="first Goal step without evidence"),
        LLMResponse(tool_calls=[ToolCall("classify-rule", "propose_goal_change", {
            "kind": "constraint_add", "summary": "No network",
            "data": {"text": "Do not use network"},
        })]),
        LLMResponse(content="change proposed"),
    ])
    runtime = make_runtime(repository, provider, console, session_directory=tmp_path / "sessions")
    chat = ChatSession(runtime, console=console)
    chat._start_goal("Inspect project")
    chat._run_user_turn("确认")
    workflow = chat._goal_workflow
    assert workflow is not None

    assert "待确认 Goal 修订" in chat._run_user_turn("继续，但不能联网")
    task = workflow.store.replay(workflow.task_id)
    assert task.pending_goal_change is not None
    assert task.active_constraints == ()
    assert len(provider.requests) == 5


@pytest.mark.parametrize("crash_boundary", ["input_started", "change_proposed", "change_accepted"])
def test_goal_change_process_exit_recovers_without_reexecuting_prior_write(tmp_path, monkeypatch, crash_boundary):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    child_code = """
import json
import os
import sys
from pathlib import Path

from techpilot.engine.events import CallbackEventSink
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.tools import WriteFileTool
from techpilot.runtime import CompletionCriterionKind, LongTaskWorkflow, RuntimeBootstrap, RuntimeBootstrapInput

class Provider:
    model = 'fake-coder'
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self):
        self.responses = iter([
            LLMResponse(tool_calls=[ToolCall('initial-write', 'write_file', {
                'file_path': 'marker.txt', 'content': 'once\\n',
            })]),
            LLMResponse(content='marker written'),
        ])

    def chat(self, messages, tools=None, on_token=None):
        return next(self.responses)

class AllowPrompt:
    def decide(self, request):
        return PermissionDecision.allow('fault-injection fixture')

root = Path(sys.argv[1])
phase = sys.argv[2]
runtime = RuntimeBootstrap(provider_factory=lambda _config: Provider()).build(RuntimeBootstrapInput(
    repository=root,
    event_sink=CallbackEventSink(lambda _event: None),
    tools=[WriteFileTool()],
    model='fake-coder',
    permission_prompt=AllowPrompt(),
    session_directory=root / 'sessions',
))
workflow = LongTaskWorkflow(runtime, task_id='fault-goal', goal='Write a marker once')
workflow.start_or_resume()
workflow.store.record_completion_criterion(
    workflow.task_id, kind=CompletionCriterionKind.FILE_EXISTS, path='marker.txt',
)
workflow.run_step('Write marker.txt once')
workflow.store.begin_goal_input(workflow.task_id, 'Never edit marker.txt again')
if phase != 'input_started':
    workflow.store.propose_goal_change(
        workflow.task_id, kind='constraint_add', summary='Do not rewrite marker.txt',
        data={'text': 'Never edit marker.txt again'},
    )
if phase == 'change_accepted':
    workflow.store.resolve_goal_change(workflow.task_id, accept=True)
print(json.dumps({'session_id': runtime.agent.session_id}), flush=True)
os._exit(0)
"""
    child = subprocess.run(
        [sys.executable, "-B", "-c", child_code, str(tmp_path), crash_boundary],
        cwd=Path.cwd(), capture_output=True, text=True, timeout=30, check=False,
    )
    assert child.returncode == 0, child.stderr
    session_id = json.loads(child.stdout)["session_id"]
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "once\n"
    sessions = SessionStore.for_repository(tmp_path, tmp_path / "sessions")
    completed_before = [
        event for event in sessions.replay(session_id).events
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value
        and event.tool_call_id == "initial-write"
    ]
    assert len(completed_before) == 1

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    provider = FakeProvider([LLMResponse(content="continued without another write")])
    runtime = make_runtime(
        tmp_path, provider, console, session_directory=tmp_path / "sessions",
        resume_session_id=session_id,
    )
    chat = ChatSession(runtime, console=console)
    chat._goal("resume fault-goal")
    task = LongTaskStore.for_repository(tmp_path, session_directory=tmp_path / "sessions").replay("fault-goal")

    if crash_boundary == "input_started":
        assert task.pending_goal_input is not None
        assert task.pending_goal_change is None
        assert "重试" in output.getvalue()
        assert "确认" not in output.getvalue()
    elif crash_boundary == "change_proposed":
        assert task.pending_goal_input is None
        assert task.pending_goal_change is not None
        assert task.active_constraints == ()
        assert "确认" in output.getvalue()
    else:
        assert task.pending_goal_input is None
        assert task.pending_goal_change is None
        assert [item.text for item in task.active_constraints] == ["Never edit marker.txt again"]
        assert len(provider.requests) == 1
        assert "Never edit marker.txt again" in provider.requests[0]["messages"][0]["content"]

    if crash_boundary != "change_accepted":
        assert provider.requests == []
        assert chat._goal_workflow is not None
        assert "Goal" in chat._run_user_turn("继续")
        assert provider.requests == []
        with pytest.raises(LongTaskStateError, match="awaits classification or confirmation"):
            chat._goal_workflow.run_step("unsafe automatic retry")
    completed_after = [
        event for event in sessions.replay(session_id).events
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value
        and event.tool_call_id == "initial-write"
    ]
    assert len(completed_after) == 1
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "once\n"


def test_chat_goal_autoselects_and_assesses_confirmed_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    check = {"kind": "file_exists", "path": "src/cli_data_tool/cli.py"}
    def stage(step_id, dependencies):
        return {
            "id": step_id, "goal": f"Inspect {step_id}", "dependencies": dependencies,
            "expected_output": "source inspection", "work_scope": ["src/cli_data_tool/cli.py"],
            "evidence": "successful glob result", "check": check, "next_action": "Continue inspection",
        }
    provider = FakeProvider([
        goal_setup_response(plan={"reason": "Two source checks", "steps": [stage("first", []), stage("second", ["first"])]}),
        LLMResponse(content="约定已提出"),
        LLMResponse(tool_calls=[ToolCall("stage-one", "glob", {"pattern": "**/*.py"})]),
        LLMResponse(content="first stage done"),
        LLMResponse(tool_calls=[ToolCall("stage-two", "glob", {"pattern": "**/*.py"})]),
        LLMResponse(tool_calls=[ToolCall("stage-complete", GOAL_COMPLETION_TOOL_NAME, {
            "summary": "both stages checked", "evidence_tool_call_ids": ["stage-two"],
        })]),
        LLMResponse(content="second stage done"),
    ])
    runtime = make_runtime(repository, provider, console, session_directory=tmp_path / "sessions")
    chat = ChatSession(runtime, console=console)
    chat._start_goal("Inspect project in two stages")
    chat._run_user_turn("确认")

    task_id = next((repository / ".techpilot" / "tasks").glob("goal-*")).name
    task = LongTaskStore.for_repository(repository, session_directory=tmp_path / "sessions").replay(task_id)
    assert task.status is LongTaskStatus.SUCCEEDED
    assert task.verified_step_ids == ("first", "second")
    assert len(task.checkpoints) == 2
    assert "Goal stage assessed: second → verified" in output.getvalue()


def test_chat_goal_clear_cancels_and_legacy_subcommands_do_not_mutate(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    runtime = make_runtime(
        repository, FakeProvider([LLMResponse(content="请补充验收方式")]), console,
        session_directory=tmp_path / "sessions",
    )
    chat = ChatSession(runtime, console=console)
    chat._start_goal("Inspect project")
    workflow = chat._goal_workflow
    assert workflow is not None
    for command in ("start", "complete", "criterion file-exists fake.txt", "plan show", "evaluate", "cancel"):
        chat._goal(command)
    assert workflow.store.replay(workflow.task_id).status is LongTaskStatus.SETUP
    assert not workflow.store.replay(workflow.task_id).active_completion_criteria

    chat._goal("clear")
    assert chat._goal_workflow is None
    assert workflow.store.replay(workflow.task_id).status is LongTaskStatus.CANCELLED
    chat._goal(f"resume {workflow.task_id}")
    assert chat._goal_workflow is None
    assert "Goal 已结束，不能恢复" in output.getvalue()


def test_chat_goal_resume_finishes_a_checkpointed_claim_without_another_model_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = copy_benchmark(tmp_path / "repository")
    session_directory = tmp_path / "sessions"
    first_console = Console(file=StringIO(), force_terminal=False, color_system=None)
    first_provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall("evidence", "glob", {"pattern": "**/*.py"})]),
        LLMResponse(tool_calls=[ToolCall("claim", GOAL_COMPLETION_TOOL_NAME, {
            "summary": "source inspected", "evidence_tool_call_ids": ["evidence"],
        })]),
        LLMResponse(content="done"),
    ])
    first = make_runtime(repository, first_provider, first_console, session_directory=session_directory)
    workflow = LongTaskWorkflow(first, task_id="goal-crash-after-claim", goal="Inspect project")
    workflow.create_for_setup()
    workflow.record_completion_criterion(CompletionCriterionKind.FILE_EXISTS, path="src/cli_data_tool/cli.py")
    workflow.start()
    workflow.run_step("Inspect project")
    assert workflow.store.replay(workflow.task_id).active_completion_claim is not None

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    provider = FakeProvider([])
    resumed = make_runtime(
        repository, provider, console, session_directory=session_directory,
        resume_session_id=first.agent.session_id,
    )
    chat = ChatSession(resumed, console=console)
    chat._goal("resume")

    assert workflow.store.replay(workflow.task_id).status is LongTaskStatus.SUCCEEDED
    assert provider.requests == []
    assert "Goal completed:" in output.getvalue()


def test_chat_goal_constraint_is_durable_and_visible_to_later_goal_step(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    provider = FakeProvider([
        goal_setup_response(constraints=("Only modify formatter.py",)),
        LLMResponse(content="约定已提出"),
        LLMResponse(content="first Goal step"),
        LLMResponse(content="constrained step"),
    ])
    runtime = make_runtime(repository, provider, console, session_directory=tmp_path / "sessions")
    inputs = iter([
        "/goal update the formatter",
        "确认",
        "continue the Goal",
        "/exit",
    ])

    assert ChatSession(runtime, console=console, input_fn=lambda _prompt: next(inputs)).run() == 0

    task_directory = next((repository / ".techpilot" / "tasks").glob("goal-*"))
    task = LongTaskStore.for_repository(repository, session_directory=tmp_path / "sessions").replay(task_directory.name)
    assert [constraint.text for constraint in task.active_constraints] == ["Only modify formatter.py"]
    assert "Only modify formatter.py" in provider.requests[2]["messages"][0]["content"]
    rendered = output.getvalue()
    assert "硬约束" in rendered


def test_chat_goal_recover_cannot_write_an_operator_fact(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    runtime = make_runtime(
        repository,
        FakeProvider([goal_setup_response(), LLMResponse(content="约定已提出"), LLMResponse(content="first Goal step")]),
        console,
        session_directory=tmp_path / "sessions",
    )
    chat = ChatSession(runtime, console=console)
    chat._start_goal("inspect the project")
    chat._run_user_turn("确认")
    workflow = chat._goal_workflow
    assert workflow is not None
    journal = EffectJournalStore.for_session_directory(runtime.session_store.directory)
    journal.start(
        session_id=runtime.agent.session_id,
        tool_call_id="write-1",
        tool_name="write_file",
        arguments={"file_path": "notes.txt"},
        effect=ToolEffect.WRITE,
    )
    workflow.store.require_recovery(workflow.task_id, reason="Shared Effect Journal has an unfinished Tool effect")

    chat._goal("recover write-1 completed operator verified the target contents")

    effect = journal.replay(runtime.agent.session_id).entries["write-1"]
    assert effect.state is EffectJournalState.STARTED
    assert effect.reconciliation_reason is None
    assert workflow.store.replay(workflow.task_id).status is LongTaskStatus.RECOVERY_REQUIRED
    assert "Goal state cannot be resolved by a user command" in output.getvalue()
    chat._show_goal_status()
    assert "effect_outcome_unknown" in output.getvalue()
    assert "write-1 (write_file)" in output.getvalue()
    resumed_chat = ChatSession(runtime, console=console)
    resumed_chat._goal(f"resume {workflow.task_id}")
    assert workflow.store.replay(workflow.task_id).status is LongTaskStatus.RECOVERY_REQUIRED
    assert "Goal attached:" in output.getvalue()


def test_chat_goal_status_distinguishes_session_evidence_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=180)
    repository = copy_benchmark(tmp_path / "repository")
    runtime = make_runtime(
        repository,
        FakeProvider([]),
        console,
        session_directory=tmp_path / "sessions",
    )
    chat = ChatSession(runtime, console=console)
    chat._goal_workflow = LongTaskWorkflow(runtime, task_id="goal-status-test", goal="inspect the project")
    chat._goal_workflow.create_for_setup()
    workflow = chat._goal_workflow
    assert workflow is not None
    workflow.store.require_recovery(
        workflow.task_id,
        reason="Checkpoint Session cursor is missing",
        code="session_evidence_invalid",
    )

    chat._show_goal_status()

    rendered = output.getvalue()
    assert "session_evidence_invalid" in rendered
    assert "Checkpoint Session cursor is missing" in rendered
    assert "Restore trustworthy Session evidence before continuing" in rendered


def test_chat_exit_keeps_an_unstarted_goal_in_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    repository = copy_benchmark(tmp_path / "repository")
    runtime = make_runtime(
        repository,
        FakeProvider([LLMResponse(content="请补充验收方式")]),
        console,
        session_directory=tmp_path / "sessions",
    )
    inputs = iter(["/goal inspect the project", "/exit"])

    assert ChatSession(runtime, console=console, input_fn=lambda _prompt: next(inputs)).run() == 0

    task_directory = next((repository / ".techpilot" / "tasks").glob("goal-*"))
    task = LongTaskStore.for_repository(repository, session_directory=tmp_path / "sessions").replay(task_directory.name)
    assert task.status is LongTaskStatus.SETUP
    assert len(runtime.agent.llm.requests) == 1
    assert "Goal setup saved:" in output.getvalue()


def test_chat_resumes_a_paused_goal_only_after_resuming_its_session(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = copy_benchmark(tmp_path / "repository")
    session_directory = tmp_path / "sessions"
    first_console = Console(file=StringIO(), force_terminal=False, color_system=None)
    first = make_runtime(
        repository,
        FakeProvider([goal_setup_response(), LLMResponse(content="约定已提出"), LLMResponse(content="first Goal step")]),
        first_console,
        session_directory=session_directory,
    )
    first_inputs = iter([
        "/goal inspect the project",
        "确认",
        "/exit",
    ])
    assert ChatSession(first, console=first_console, input_fn=lambda _prompt: next(first_inputs)).run() == 0
    task_directory = next((repository / ".techpilot" / "tasks").glob("goal-*"))

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    resumed = make_runtime(
        repository,
        FakeProvider([
            LLMResponse(tool_calls=[ToolCall("recovery-evidence", "glob", {"pattern": "**/*.py"})]),
            LLMResponse(tool_calls=[ToolCall(
                "recovery-proposal",
                GOAL_COMPLETION_TOOL_NAME,
                {"summary": "recovered sources inspected", "evidence_tool_call_ids": ["recovery-evidence"]},
            )]),
            LLMResponse(content="recovered Goal step"),
        ]),
        console,
        session_directory=session_directory,
        resume_session_id=first.agent.session_id,
    )
    resumed_inputs = iter([
        f"/goal resume {task_directory.name}",
        "/exit",
    ])

    assert ChatSession(resumed, console=console, input_fn=lambda _prompt: next(resumed_inputs)).run() == 0

    task = LongTaskStore.for_repository(repository, session_directory=session_directory).replay(task_directory.name)
    assert task.status is LongTaskStatus.SUCCEEDED
    assert len(task.checkpoints) == 2
    assert "Goal resumed:" in output.getvalue()
    assert "recovered Goal step" in output.getvalue()


def test_limit_backfilled_read_file_is_rendered_as_not_executed():
    output = StringIO()
    sink = TerminalEventSink(Console(file=output, force_terminal=False, color_system=None))
    event_args = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "round_index": 2,
        "tool_call_id": "call-1",
    }

    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED,
        payload={"tool_name": "read_file", "arguments": {"file_path": "README.md"}},
        **event_args,
    ))
    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_COMPLETED,
        payload={"tool_name": "read_file", "result": "[limit reached]", "interrupted": False},
        **event_args,
    ))

    rendered = output.getvalue()
    assert "← read_file: not executed" in rendered
    assert "已读取 15 个字符" not in rendered
    assert "未执行：达到运行限制" in rendered


def test_block_reason_is_rendered_once_but_the_denied_detail_stays_available():
    output = StringIO()
    sink = TerminalEventSink(Console(file=output, force_terminal=False, color_system=None))
    event_args = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "round_index": 1,
        "tool_call_id": "blocked-shell",
    }
    result = "Policy denied bash: 该操作已被阻断，未执行。原因与证据：unsafe shell structure"

    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED,
        payload={"tool_name": "bash", "arguments": {"command": "dir /b 2>nul & dir /b"}},
        **event_args,
    ))
    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED,
        payload={
            "required_control": "block",
            "reasons": [{"message": "unsafe shell structure", "evidence": ["redirection"]}],
        },
        **event_args,
    ))
    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_COMPLETED,
        payload={"tool_name": "bash", "result": result, "interrupted": False},
        **event_args,
    ))

    rendered = output.getvalue()
    assert "操作已阻断" in rendered
    assert "unsafe shell structure：redirection" in rendered
    assert "← bash: denied" in rendered
    assert "details: /details blocked-shell" in rendered
    assert "Policy denied bash" not in rendered


def test_tool_result_summary_is_grouped_until_the_user_chooses_a_detail():
    output = StringIO()
    sink = TerminalEventSink(Console(file=output, force_terminal=False, color_system=None))
    event_args = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "round_index": 1,
        "tool_call_id": "shell-compact",
    }
    full_output = "private command output\n" * 400

    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED,
        payload={"tool_name": "bash", "arguments": {"command": "python -m pytest -q"}},
        **event_args,
    ))
    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_COMPLETED,
        payload={"tool_name": "bash", "result": full_output, "interrupted": False},
        **event_args,
    ))
    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TURN_COMPLETED,
        payload={},
        **event_args,
    ))

    rendered = output.getvalue()
    assert "工具调用已折叠：bash × 1（1 项完成" in rendered
    assert "/details 查看并选择详情" in rendered
    assert "bash(command='python -m pytest -q') · running" not in rendered
    assert "← bash: completed" not in rendered
    assert full_output not in rendered


def test_completed_tool_calls_share_one_collapsed_turn_summary():
    output = StringIO()
    sink = TerminalEventSink(Console(file=output, force_terminal=False, color_system=None))
    for call_id, tool_name, arguments, result in (
        ("shell-1", "bash", {"command": "dir /b"}, ".techpilot"),
        ("glob-1", "glob", {"pattern": "**/*.py"}, "a.py\nb.py"),
    ):
        event_args = {
            "session_id": "session-1",
            "turn_id": "turn-1",
            "round_index": 1,
            "tool_call_id": call_id,
        }
        sink.emit(RuntimeEvent(
            event_type=RuntimeEventType.TOOL_REQUESTED,
            payload={"tool_name": tool_name, "arguments": arguments},
            **event_args,
        ))
        sink.emit(RuntimeEvent(
            event_type=RuntimeEventType.TOOL_COMPLETED,
            payload={"tool_name": tool_name, "result": result, "interrupted": False},
            **event_args,
        ))
    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TURN_COMPLETED,
        payload={},
        session_id="session-1",
        turn_id="turn-1",
        round_index=1,
    ))

    rendered = output.getvalue()
    assert rendered.count("工具调用已折叠：") == 1
    assert "bash × 1、glob × 1（2 项完成" in rendered
    assert "shell-1" not in rendered
    assert "glob-1" not in rendered


def test_details_on_keeps_per_call_tool_output_compatibility():
    output = StringIO()
    sink = TerminalEventSink(Console(file=output, force_terminal=False, color_system=None))
    sink.show_full_results = True
    event_args = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "round_index": 1,
        "tool_call_id": "shell-full",
    }

    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED,
        payload={"tool_name": "bash", "arguments": {"command": "dir /b"}},
        **event_args,
    ))
    sink.emit(RuntimeEvent(
        event_type=RuntimeEventType.TOOL_COMPLETED,
        payload={"tool_name": "bash", "result": ".techpilot", "interrupted": False},
        **event_args,
    ))

    rendered = output.getvalue()
    assert "bash(command='dir /b') · running" in rendered
    assert "← bash: completed" in rendered
    assert ".techpilot" in rendered
    assert "工具调用已折叠" not in rendered


def test_details_reads_saved_shell_control_validation_and_file_diff_without_execution(tmp_path):
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    sink = TerminalEventSink(console)
    store = SessionStore(tmp_path / "sessions")
    session_id = "saved-details"
    store.create(session_id, repository_root=tmp_path, model="fake-model")
    store.append(SessionEvent(
        event_type="tool_requested",
        session_id=session_id,
        turn_id="turn-1",
        round_index=1,
        tool_call_id="shell-1",
        payload={"tool_name": "bash", "arguments": {"command": "python -m pytest -q"}},
    ))
    store.append(SessionEvent(
        event_type="execution_control_assessed",
        session_id=session_id,
        turn_id="turn-1",
        round_index=1,
        tool_call_id="shell-1",
        payload={
            "tool_name": "bash",
            "required_control": "confirm",
            "normalized_summary": {"command": "python -m pytest -q"},
            "reasons": [{"message": "General command", "evidence": ["command_kind=general"]}],
        },
    ))
    store.append(SessionEvent(
        event_type="tool_completed",
        session_id=session_id,
        turn_id="turn-1",
        round_index=1,
        tool_call_id="shell-1",
        payload={
            "tool_name": "bash",
            "result": "failed test\n[stderr]\ntraceback\n[exit code: 3]",
            "interrupted": False,
        },
    ))
    store.append(SessionEvent(
        event_type="tool_requested",
        session_id=session_id,
        turn_id="turn-2",
        round_index=1,
        tool_call_id="write-1",
        payload={"tool_name": "write_file", "arguments": {"file_path": "app.py", "content": "after\n"}},
    ))
    store.append(SessionEvent(
        event_type="tool_completed",
        session_id=session_id,
        turn_id="turn-2",
        round_index=1,
        tool_call_id="write-1",
        payload={
            "tool_name": "write_file",
            "result": "Wrote 1 lines to app.py\n--- a/app.py\n+++ b/app.py\n@@\n-before\n+after\n",
            "interrupted": False,
        },
    ))
    runtime = SimpleNamespace(
        agent=SimpleNamespace(event_sink=sink, session_id=session_id),
        session_store=store,
        repository=tmp_path,
        profile=SimpleNamespace(validation_commands=[["python", "-m", "pytest", "-q"]]),
    )
    session = ChatSession(runtime, console=console)

    session._details("shell-1")
    session._details("write-1")
    session._details("")

    rendered = output.getvalue()
    assert "Command: python -m pytest -q" in rendered
    assert "cwd:" in rendered
    assert "Exit code: 3" in rendered
    assert "stdout:" in rendered
    assert "failed test" in rendered
    assert "stderr:" in rendered
    assert "traceback" in rendered
    assert "Validation: failed" in rendered
    assert "Control: confirm" in rendered
    assert "General command：command_kind=general" in rendered
    assert "Path: app.py" in rendered
    assert "Change summary: Wrote 1 lines to app.py" in rendered
    assert "--- a/app.py" in rendered
    assert "Completed. The saved Tool Result below contains the Trusted Diff" in rendered
    assert "折叠的工具调用" in rendered
    assert "shell-1" in rendered
    assert "write-1" in rendered
    assert store.replay(session_id).events[-1].payload["result"].endswith("+after\n")


def test_details_pages_saved_large_results_and_keeps_detail_mode_compatibility(tmp_path):
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    sink = TerminalEventSink(console)
    store = SessionStore(tmp_path / "sessions")
    session_id = "paged-details"
    store.create(session_id, repository_root=tmp_path, model="fake-model")
    large_result = "x" * 16_100
    for event_type, payload in (
        ("tool_requested", {"tool_name": "read_file", "arguments": {"file_path": "large.txt"}}),
        ("tool_completed", {"tool_name": "read_file", "result": large_result, "interrupted": False}),
    ):
        store.append(SessionEvent(
            event_type=event_type,
            session_id=session_id,
            turn_id="turn-1",
            round_index=1,
            tool_call_id="read-large",
            payload=payload,
        ))
    runtime = SimpleNamespace(
        agent=SimpleNamespace(event_sink=sink, session_id=session_id),
        session_store=store,
        repository=tmp_path,
        profile=None,
    )
    session = ChatSession(runtime, console=console)

    session._details("on")
    session._details("read-large 2")

    rendered = output.getvalue()
    assert sink.show_full_results
    assert "Tool detail mode: on" in rendered
    assert "输出已分页：第 2/3 页" in rendered
    assert store.replay(session_id).events[-1].payload["result"] == large_result


class FakeProvider:
    model = "fake-coder"
    total_prompt_tokens = 12
    total_completion_tokens = 8
    estimated_cost = None

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, tools=None, on_token=None):
        self.requests.append({"messages": messages, "tools": tools})
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if on_token and response.content:
            on_token(response.content)
        return response


class AllowOncePrompt:
    def __init__(self):
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        return PermissionDecision.allow("test approval")


class DenyPrompt(AllowOncePrompt):
    def decide(self, request):
        self.requests.append(request)
        return PermissionDecision.deny("test rejection")


def make_runtime(
    repository: Path,
    provider: FakeProvider,
    console: Console,
    *,
    permission_prompt=None,
    session_directory: Path | None = None,
    resume_session_id: str | None = None,
):
    sink = TerminalEventSink(console)
    bootstrap = RuntimeBootstrap(provider_factory=lambda config: provider)
    return bootstrap.build(RuntimeBootstrapInput(
        repository=repository,
        event_sink=sink,
        permission_prompt=permission_prompt,
        session_directory=session_directory,
        resume_session_id=resume_session_id,
    ))


def test_runtime_bootstrap_builds_profile_context_and_repository_scoped_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    provider = FakeProvider([LLMResponse(content="ok")])
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    runtime = make_runtime(
        BENCHMARK_ROOT,
        provider,
        console,
        session_directory=tmp_path / "sessions",
    )

    assert runtime.repository == BENCHMARK_ROOT.resolve()
    assert isinstance(runtime, TaskRuntime)
    assert ChatRuntime is TaskRuntime
    assert runtime.identity.working_directory == BENCHMARK_ROOT.resolve()
    assert runtime.profile is not None
    assert runtime.profile.language == "python"
    assert "src/cli_data_tool/cli.py" in runtime.profile.entrypoints
    assert "Repository root:" in runtime.agent._system
    assert "This is a lightweight profile" in runtime.agent._system
    assert "Product: TechPilot" in runtime.agent._system
    assert f"Current model: {runtime.config.model}" in runtime.agent._system
    if sys.platform == "win32":
        assert "use cmd-compatible commands such as `dir`, not Unix commands such as `ls -la`" in runtime.agent._system
    assert {tool.name for tool in runtime.tools} == {
        "read_file",
        "glob",
        "grep",
        "edit_file",
        "write_file",
        "bash",
        "now",
    }


def test_role_switch_replaces_prior_role_tools_and_records_structured_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    runtime = make_runtime(
        BENCHMARK_ROOT,
        FakeProvider([]),
        Console(file=StringIO(), force_terminal=False, color_system=None),
        session_directory=tmp_path / "sessions",
    )

    runtime.activate_role("network-role", "network context", tool_names=("fetch_url",))
    runtime.activate_role("delegate-role", "delegate context", tool_names=("agent",))

    assert runtime.active_role == ActiveRole(
        role_id="delegate-role",
        context="delegate context",
        tool_names=("agent",),
    )
    assert "fetch_url" not in {tool.name for tool in runtime.tools}
    assert "agent" in {tool.name for tool in runtime.tools}
    assert "delegate context" in runtime.agent._system
    assert "network context" not in runtime.agent._system
    events = runtime.session_store.replay(runtime.agent.session_id).events
    assert [(event.event_type, event.payload) for event in events if event.event_type == "role_activated"] == [
        ("role_activated", {"role_id": "network-role", "tool_names": ["fetch_url"]}),
        ("role_activated", {"role_id": "delegate-role", "tool_names": ["agent"]}),
    ]


def test_failed_role_switch_restores_prior_runtime_and_agent_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    runtime = make_runtime(
        BENCHMARK_ROOT,
        FakeProvider([]),
        Console(file=StringIO(), force_terminal=False, color_system=None),
        session_directory=tmp_path / "sessions",
    )
    runtime.activate_role("network-role", "network context", tool_names=("fetch_url",))
    original_update = runtime.agent.update_system_context
    calls = 0

    def fail_once(context: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated context failure")
        original_update(context)

    monkeypatch.setattr(runtime.agent, "update_system_context", fail_once)

    with pytest.raises(RuntimeError, match="simulated context failure"):
        runtime.activate_role("delegate-role", "delegate context", tool_names=("agent",))

    assert runtime.active_role == ActiveRole(
        role_id="network-role",
        context="network context",
        tool_names=("fetch_url",),
    )
    assert "fetch_url" in {tool.name for tool in runtime.tools}
    assert "agent" not in {tool.name for tool in runtime.tools}
    assert "network context" in runtime.agent._system
    assert "delegate context" not in runtime.agent._system
    events = runtime.session_store.replay(runtime.agent.session_id).events
    assert [event.payload["role_id"] for event in events if event.event_type == "role_activated"] == ["network-role"]


def test_temporary_role_scope_clears_on_exception_and_restores_default_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    runtime = make_runtime(
        BENCHMARK_ROOT,
        FakeProvider([]),
        Console(file=StringIO(), force_terminal=False, color_system=None),
        session_directory=tmp_path / "sessions",
    )

    with pytest.raises(RuntimeError, match="cancelled temporary role"), runtime.role_scope(
        "temporary-role", "temporary context", tool_names=("fetch_url",)
    ):
        assert runtime.active_role is not None
        assert "fetch_url" in {tool.name for tool in runtime.tools}
        raise RuntimeError("cancelled temporary role")

    assert runtime.active_role is None
    assert runtime.role_context is None
    assert "fetch_url" not in {tool.name for tool in runtime.tools}
    assert "temporary context" not in runtime.agent._system
    events = runtime.session_store.replay(runtime.agent.session_id).events
    assert [(event.event_type, event.payload) for event in events if event.event_type.startswith("role_")] == [
        ("role_activated", {"role_id": "temporary-role", "tool_names": ["fetch_url"]}),
        ("role_cleared", {"role_id": "temporary-role"}),
    ]


def test_resuming_session_does_not_reactivate_historical_role(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    session_directory = tmp_path / "sessions"
    first = make_runtime(
        tmp_path,
        FakeProvider([]),
        Console(file=StringIO(), force_terminal=False, color_system=None),
        session_directory=session_directory,
    )
    first.activate_role("network-role", "network context", tool_names=("fetch_url",))

    resumed = RuntimeBootstrap(provider_factory=lambda _config: FakeProvider([])).build(RuntimeBootstrapInput(
        repository=tmp_path,
        event_sink=TerminalEventSink(Console(file=StringIO(), force_terminal=False, color_system=None)),
        session_directory=session_directory,
        resume_session_id=first.agent.session_id,
    ))

    assert resumed.active_role is None
    assert resumed.role_context is None
    assert "fetch_url" not in {tool.name for tool in resumed.tools}
    assert any("不会自动恢复 Role、工具或历史授权" in notice for notice in resumed.consume_recovery_notices())


def test_techpilot_model_setting_and_cli_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    monkeypatch.delenv("TECHPILOT_MAX_CONTEXT", raising=False)
    monkeypatch.delenv("TECHPILOT_MODEL", raising=False)
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    def build(model=None):
        return RuntimeBootstrap(provider_factory=lambda config: FakeProvider([])).build(
            RuntimeBootstrapInput(
                repository=BENCHMARK_ROOT,
                event_sink=TerminalEventSink(console),
                model=model,
                session_directory=tmp_path / "sessions",
            )
        )

    assert build().config.model == "gpt-5.5"

    monkeypatch.setenv("TECHPILOT_MODEL", "techpilot-model")
    assert build().config.model == "techpilot-model"
    assert build("command-line-model").config.model == "command-line-model"

    monkeypatch.setenv("TECHPILOT_MODEL", "deepseek-v4-flash")
    runtime = build()
    assert runtime.config.max_context_tokens == 1_000_000
    runtime.set_model("deepseek-v4-pro")
    assert runtime.config.max_context_tokens == 1_000_000


def test_repository_executor_denies_paths_outside_repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    executor = RepositoryToolExecutor(repository)

    class NeverRun:
        name = "read_file"

        def execute(self, **kwargs):
            raise AssertionError("outside request should not execute")

    result = executor.execute(NeverRun(), {"file_path": str(outside)})
    assert result.startswith("Policy denied read_file")

    NeverRun.name = "glob"
    result = executor.execute(NeverRun(), {"path": ".", "pattern": "../*.txt"})
    assert result.startswith("Policy denied glob")


def test_repository_executor_presents_search_hits_relative_to_the_repository(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "pkg" / "messages.py"
    source.parent.mkdir(parents=True)
    source.write_text("MESSAGE = 'Connection failed'\n", encoding="utf-8")
    executor = RepositoryToolExecutor(repository)

    glob_result = executor.execute(GlobTool(), {"pattern": "**/*.py"})
    grep_result = executor.execute(GrepTool(), {"pattern": "Connection failed", "include": "*.py"})

    assert "pkg/messages.py" in glob_result
    assert "pkg/messages.py:1: MESSAGE = 'Connection failed'" in grep_result
    assert str(repository) not in glob_result
    assert str(repository) not in grep_result


def test_repository_executor_allows_recovery_after_a_blocked_read_search(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "messages.py").write_text("MESSAGE = 'Connection failed'\n", encoding="utf-8")
    executor = RepositoryToolExecutor(repository)
    executor.begin_turn()

    denied = executor.execute(GlobTool(), {"pattern": "**/*", "path": ".."})
    recovered = executor.execute(GlobTool(), {"pattern": "**/*.py"})

    assert denied.startswith("Policy denied glob")
    assert executor.consume_turn_stop_message() is None
    assert recovered == "messages.py"


def test_default_cli_spelling_enters_chat_for_current_or_explicit_repository():
    assert _normalize_command([]) == ["chat", "."]
    assert _normalize_command(["."]) == ["chat", "."]
    assert _normalize_command([str(BENCHMARK_ROOT)]) == ["chat", str(BENCHMARK_ROOT)]
    assert _normalize_command(["profile", str(BENCHMARK_ROOT)]) == ["profile", str(BENCHMARK_ROOT)]


def test_chat_read_only_inspects_benchmark_without_modifying_files(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = copy_benchmark(tmp_path / "cli_data_tool")
    read_path = "README.md"
    original = (repository / read_path).read_bytes()
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall("read-1", "read_file", {"file_path": read_path})]),
        LLMResponse(content="仓库说明已读取，未修改文件。"),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    runtime = make_runtime(repository, provider, console)
    inputs = iter(["读取 README 并说明项目，不要修改文件", "/exit"])

    assert ChatSession(runtime, console=console, input_fn=lambda prompt: next(inputs)).run() == 0

    assert (repository / read_path).read_bytes() == original
    assert "工具调用已折叠：read_file × 1（1 项完成" in output.getvalue()
    assert len(provider.requests) == 2


def test_save_command_does_not_claim_success_when_session_save_fails(tmp_path):
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = SimpleNamespace(
        agent=SimpleNamespace(session_id="session-1"),
        session_store=SimpleNamespace(directory=tmp_path / "sessions"),
    )

    def fail_to_persist() -> None:
        raise OSError("simulated session write failure")

    runtime.ensure_persisted = fail_to_persist
    ChatSession(runtime, console=console)._show_save()

    rendered = output.getvalue()
    assert "Session autosave error: simulated session write failure" in rendered
    assert "Current Session could not be confirmed as saved." in rendered
    assert "自动保存已开启" not in rendered


def test_chat_reads_dependency_manifest_directly_without_prompting_or_isolating(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = copy_benchmark(tmp_path / "cli_data_tool")
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall("read-manifest", "read_file", {"file_path": "pyproject.toml"})]),
        LLMResponse(content="依赖配置已读取。"),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    prompt = DenyPrompt()
    runtime = make_runtime(repository, provider, console, permission_prompt=prompt)
    inputs = iter(["读取 pyproject.toml，不要修改文件", "/exit"])

    assert ChatSession(runtime, console=console, input_fn=lambda prompt: next(inputs)).run() == 0

    rendered = output.getvalue()
    assert "工具调用已折叠：read_file × 1（1 项完成" in rendered
    assert "需要隔离执行" not in rendered
    assert prompt.requests == []


def test_chat_directory_listing_command_executes_directly_without_prompting(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = copy_benchmark(tmp_path / "cli_data_tool")
    directory_command = "dir" if sys.platform == "win32" else "ls -la"
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall("list-files", "bash", {"command": directory_command})]),
        LLMResponse(content="目录已列出。"),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    prompt = DenyPrompt()
    runtime = make_runtime(repository, provider, console, permission_prompt=prompt)
    inputs = iter(["列出当前目录文件", "/exit"])

    assert ChatSession(runtime, console=console, input_fn=lambda prompt: next(inputs)).run() == 0

    rendered = output.getvalue()
    assert "工具调用已折叠：bash × 1（1 项完成" in rendered
    assert prompt.requests == []


def test_directory_listing_commands_are_normalized_as_read_only_shell_commands():
    assert _normalized_command("dir").kind.value == "read_only_shell"
    assert _normalized_command("ls -la").kind.value == "read_only_shell"


def test_chat_end_to_end_reads_edits_validates_and_continues_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = copy_benchmark(tmp_path / "cli_data_tool")
    read_path = "src/cli_data_tool/exporter.py"
    original = (repository / read_path).read_text(encoding="utf-8")
    old = 'return "\\n".join(items)'
    new = 'return "\\n".join(str(item) for item in items)'
    assert old in original
    test_command = subprocess.list2cmdline([
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ])

    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall("read-1", "read_file", {"file_path": read_path})]),
        LLMResponse(tool_calls=[ToolCall(
            "edit-1",
            "edit_file",
            {"file_path": read_path, "old_string": old, "new_string": new},
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "test-1",
            "bash",
            {"command": test_command},
        )]),
        LLMResponse(content="修改和测试已经完成。"),
        LLMResponse(content="我还记得刚才的修改。"),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    prompt = AllowOncePrompt()
    runtime = make_runtime(repository, provider, console, permission_prompt=prompt)
    inputs = iter(["完成一个小修改并运行测试", "总结刚才做了什么", "/exit"])

    assert ChatSession(runtime, console=console, input_fn=lambda prompt: next(inputs)).run() == 0

    assert new in (repository / read_path).read_text(encoding="utf-8")
    rendered = output.getvalue()
    assert "TechPilot Chat" in rendered
    assert "工具调用已折叠：read_file × 1、edit_file × 1、bash × 1（3 项完成" in rendered
    assert "→ read_file" not in rendered
    assert "← edit_file: completed" not in rendered
    assert "修改和测试已经完成。" in rendered
    assert "我还记得刚才的修改。" in rendered
    assert len(provider.requests) == 5
    assert [request.tool_name for request in prompt.requests] == ["edit_file"]


def test_rejected_write_stops_the_current_turn_without_retrying_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            "edit-denied",
            "edit_file",
            {"file_path": "notes.txt", "old_string": "original", "new_string": "changed"},
        )]),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    prompt = DenyPrompt()
    runtime = make_runtime(tmp_path, provider, console, permission_prompt=prompt)

    response = runtime.agent.chat("修改 notes.txt")

    assert response == "已按你的拒绝停止本轮后续操作；edit_file 没有执行。你可以继续说明下一步需求。"
    assert target.read_text(encoding="utf-8") == "original\n"
    assert len(provider.requests) == 1
    assert runtime.agent.messages[-2] == {
        "role": "tool",
        "tool_call_id": "edit-denied",
        "content": "Permission denied edit_file: test rejection",
    }
    assert runtime.agent.messages[-1]["content"] == response
    assert "edit_file: denied" in output.getvalue()


def test_blocked_command_stops_the_turn_without_a_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            "dangerous-command",
            "bash",
            {"command": "git reset --hard"},
        )]),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = make_runtime(tmp_path, provider, console)

    response = runtime.run_turn("执行 git reset --hard")

    assert "该操作已被阻断，未执行" in response
    assert len(provider.requests) == 1
    assert "bash: denied" in output.getvalue()
    assert "git reset --hard" in runtime.agent.messages[-2]["content"]
    saved = runtime.session_store.replay(runtime.agent.session_id)
    assessment = next(event for event in saved.events if event.event_type == "execution_control_assessed")
    assert assessment.payload["required_control"] == "block"
    assert assessment.tool_call_id == "dangerous-command"
    assert assessment.payload["reasons"][0]["evidence"]
    ChatSession(runtime, console=console)._session_command("show")
    assert "Tool Call blocked, not executed" in output.getvalue()


def test_chat_blocks_patch_application_without_a_source_promotion_capability(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            "apply-patch",
            "bash",
            {"command": "git apply changes.patch"},
        )]),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = make_runtime(tmp_path, provider, console)

    response = runtime.run_turn("把审查 Patch 应用到源仓库")

    assert "该操作已被阻断，未执行" in response
    assert "将 Patch 回写源仓库需要专用、可审查的应用能力" in response
    assert target.read_text(encoding="utf-8") == "original\n"
    assert len(provider.requests) == 1
    saved = runtime.session_store.replay(runtime.agent.session_id)
    assessment = next(event for event in saved.events if event.event_type == "execution_control_assessed")
    assert assessment.payload["required_control"] == "block"
    assert assessment.payload["reasons"][0]["code"] == "unsupported_patch_application"


def test_chat_blocks_patch_application_with_git_global_path_options(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            "apply-patch-in-worktree",
            "bash",
            {"command": "git -C . apply changes.patch"},
        )]),
    ])
    runtime = make_runtime(tmp_path, provider, Console(file=StringIO(), force_terminal=False, color_system=None))

    response = runtime.run_turn("在当前工作目录应用审查 Patch")

    assert "该操作已被阻断，未执行" in response
    saved = runtime.session_store.replay(runtime.agent.session_id)
    assessment = next(event for event in saved.events if event.event_type == "execution_control_assessed")
    assert assessment.payload["reasons"][0]["code"] == "unsupported_patch_application"


def test_chat_execution_control_assesses_direct_and_confirm_before_effects(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    target = tmp_path / "notes.py"
    target.write_text("before\n", encoding="utf-8")
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall("read-1", "read_file", {"file_path": "notes.py"})]),
        LLMResponse(tool_calls=[ToolCall("write-1", "write_file", {"file_path": "notes.py", "content": "after\n"})]),
        LLMResponse(content="完成。"),
    ])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = make_runtime(tmp_path, provider, console, permission_prompt=AllowOncePrompt())

    assert runtime.run_turn("先读取").startswith("完成")
    assert target.read_text(encoding="utf-8") == "after\n"
    events = runtime.session_store.replay(runtime.agent.session_id).events
    assessments = [event for event in events if event.event_type == "execution_control_assessed"]

    assert [event.payload["required_control"] for event in assessments] == ["direct", "confirm"]
    assert all(event.session_id == runtime.agent.session_id and event.turn_id for event in assessments)
    assert all(event.tool_call_id and event.payload["reasons"] for event in assessments)


def test_chat_confirms_high_impact_lock_write_in_source_with_reasons(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    lock_file = tmp_path / "poetry.lock"
    lock_file.write_text("original\n", encoding="utf-8")
    prompt = AllowOncePrompt()
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            "lock-write", "write_file", {"file_path": "poetry.lock", "content": "changed\n"},
        )]),
        LLMResponse(content="已完成。"),
    ])
    runtime = make_runtime(
        tmp_path,
        provider,
        Console(file=StringIO(), force_terminal=False, color_system=None),
        permission_prompt=prompt,
        session_directory=tmp_path / "sessions",
    )

    assert runtime.run_turn("更新锁文件") == "已完成。"
    assert lock_file.read_text(encoding="utf-8") == "changed\n"
    assert len(prompt.requests) == 1
    assert "Operation modifies a lock file" in prompt.requests[0].reason
    assert "file_category=lock_file" in prompt.requests[0].reason
    assert prompt.requests[0].trusted_preview
    assessment = next(
        event for event in runtime.session_store.replay(runtime.agent.session_id).events
        if event.event_type == "execution_control_assessed"
    )
    assert assessment.payload["required_control"] == "confirm"
    assert not list(tmp_path.glob("runs/*/workspace"))


def test_chat_rejection_keeps_high_impact_source_write_unexecuted(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text("[project]\nname = 'before'\n", encoding="utf-8")
    prompt = DenyPrompt()
    provider = FakeProvider([LLMResponse(tool_calls=[ToolCall(
        "manifest-write", "write_file", {"file_path": "pyproject.toml", "content": "[project]\nname = 'after'\n"},
    )])])
    runtime = make_runtime(
        tmp_path,
        provider,
        Console(file=StringIO(), force_terminal=False, color_system=None),
        permission_prompt=prompt,
        session_directory=tmp_path / "sessions",
    )

    response = runtime.run_turn("更新依赖配置")

    assert "没有执行" in response
    assert manifest.read_text(encoding="utf-8") == "[project]\nname = 'before'\n"
    assert len(prompt.requests) == 1
    assert "dependency manifest" in prompt.requests[0].reason
    assert not list(tmp_path.glob("runs/*/workspace"))


def test_chat_commands_and_eof_are_local_and_do_not_call_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    provider = FakeProvider([])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = RuntimeBootstrap(provider_factory=lambda config: provider).build(RuntimeBootstrapInput(
        repository=BENCHMARK_ROOT,
        event_sink=TerminalEventSink(console),
        session_directory=tmp_path / "sessions",
    ))
    commands = iter(["/help", "/status", "/tools", "/files", "/diff", "/tokens", "/compact", "/save", "/sessions", "/session show", "/model", "/clear"])

    def input_fn(prompt):
        try:
            return next(commands)
        except StopIteration as error:
            raise EOFError from error

    assert ChatSession(runtime, console=console, input_fn=input_fn).run() == 0
    rendered = output.getvalue()
    assert "TechPilot Commands" in rendered
    assert "操作保护：写入和命令会在执行前按实际影响进行确认或阻断。" in rendered
    assert "/plan" not in rendered
    assert "自动保存已开启" in rendered
    assert "TechPilot Sessions" in rendered
    assert "Session details" in rendered
    assert "Last result: -" in rendered
    assert "Current model:" in rendered
    assert "Bye!" in rendered
    assert provider.requests == []


def test_status_shows_live_session_and_context_summary_without_startup_panel(tmp_path):
    provider = FakeProvider([])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = RuntimeBootstrap(provider_factory=lambda config: provider).build(RuntimeBootstrapInput(
        repository=tmp_path,
        event_sink=TerminalEventSink(console),
        session_directory=tmp_path / "sessions",
    ))

    ChatSession(runtime, console=console)._show_status()

    rendered = output.getvalue()
    assert "Session ID" in rendered
    assert runtime.agent.session_id in rendered
    assert "Context" in rendered
    assert "Usage" in rendered
    assert "Estimated cost" in rendered
    assert "Repository" not in rendered
    assert "TechPilot Chat" not in rendered
    assert provider.requests == []


def test_diff_does_not_leak_a_parent_git_worktree(monkeypatch, tmp_path):
    parent = tmp_path / "parent-repository"
    repository = parent / "temporary-copy"
    repository.mkdir(parents=True)
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    session = ChatSession(SimpleNamespace(repository=repository), console=console)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, stdout=str(parent), stderr="")

    monkeypatch.setattr("techpilot.chat.session.subprocess.run", fake_run)

    session._show_diff()

    assert calls == [(["git", "rev-parse", "--show-toplevel"], repository)]
    assert "已避免展示父级仓库的变更" in output.getvalue()


def test_ctrl_c_cancels_only_current_turn_and_chat_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    provider = FakeProvider([KeyboardInterrupt(), LLMResponse(content="second turn works")])
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    runtime = make_runtime(
        BENCHMARK_ROOT,
        provider,
        console,
        session_directory=tmp_path / "sessions",
    )
    inputs = iter(["interrupt this turn", "continue", "/exit"])

    assert ChatSession(runtime, console=console, input_fn=lambda prompt: next(inputs)).run() == 0
    rendered = output.getvalue()
    assert "Turn cancelled. The chat session is still active." in rendered
    assert "second turn works" in rendered
    assert len(provider.requests) == 2


def test_profile_failure_warns_but_still_builds_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")

    class BrokenProfiler:
        def profile(self, repository):
            raise RuntimeError("broken parser")

    provider = FakeProvider([LLMResponse(content="fallback works")])
    sink = TerminalEventSink(Console(file=StringIO(), force_terminal=False, color_system=None))
    runtime = RuntimeBootstrap(
        provider_factory=lambda config: provider,
        profiler=BrokenProfiler(),
    ).build(RuntimeBootstrapInput(repository=tmp_path, event_sink=sink))

    assert runtime.profile is None
    assert runtime.profile_warning == "Repository profile unavailable: broken parser"
    assert runtime.agent.chat("hello") == "fallback works"
