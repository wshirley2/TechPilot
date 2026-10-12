"""G4 acceptance checks for explicitly bounded Goal progression."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from techpilot.chat.session import ChatSession
from techpilot.engine.events import CallbackEventSink, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.runtime_control import RuntimeLimits
from techpilot.engine.tool_execution import ToolEffect
from techpilot.engine.tools import ReadFileTool, WriteFileTool
from techpilot.runtime import (
    GOAL_COMPLETION_TOOL_NAME,
    CompletionCriterionKind,
    EffectJournalStore,
    LongTaskStatus,
    LongTaskWorkflow,
    RuntimeBootstrap,
    RuntimeBootstrapInput,
)
from techpilot.runtime.long_tasks import LongTaskEvent
from techpilot.runtime.sessions import SessionEvent


class FixedProvider:
    model = "fixed-g4-provider"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self, responses: list[LLMResponse | BaseException]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict] = []

    def chat(self, messages, tools=None, on_token=None):
        self.requests.append({"messages": messages, "tools": tools})
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if on_token and response.content:
            on_token(response.content)
        return response


def _step(step_id: str, path: str, dependencies: list[str] | None = None) -> dict:
    return {
        "id": step_id,
        "goal": f"Inspect {path}",
        "dependencies": dependencies or [],
        "expected_output": f"Evidence for {path}",
        "work_scope": [path],
        "evidence": "Recorded file read",
        "check": {"kind": "file_exists", "path": path},
        "next_action": f"Read {path}",
    }


def _chat(
    tmp_path, provider: FixedProvider, *, plan: bool = True,
    limits: RuntimeLimits | None = None, stage_count: int = 2,
    tools=None, permission_prompt=None,
):
    names = ("first", "second", "third", "fourth", "fifth")[:stage_count]
    for name in names:
        (tmp_path / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
    runtime = RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=tmp_path,
        event_sink=CallbackEventSink(lambda _event: None),
        tools=tools if tools is not None else [ReadFileTool()],
        model="fixed-g4-provider",
        session_directory=tmp_path / "sessions",
        limits=limits,
        permission_prompt=permission_prompt,
    ))
    workflow = LongTaskWorkflow(runtime, task_id="g4-continuous", goal="Inspect two files")
    workflow.create_for_setup()
    workflow.store.record_completion_criterion(
        workflow.task_id, kind=CompletionCriterionKind.FILE_EXISTS, path=f"{names[-1]}.txt",
    )
    if plan:
        workflow.store.record_goal_plan(workflow.task_id, {
            "reason": "Inspect all files", "steps": [
                _step(name, f"{name}.txt", [names[index - 1]] if index else [])
                for index, name in enumerate(names)
            ],
        })
        workflow.store.select_goal_step(workflow.task_id, "first")
    workflow.start()
    output = StringIO()
    chat = ChatSession(runtime, console=Console(file=output, force_terminal=False, color_system=None))
    chat._goal_workflow = workflow
    return chat, workflow, output


def test_explicit_run_finishes_two_verified_stages_and_records_stop(tmp_path):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="First file inspected"),
        LLMResponse(tool_calls=[ToolCall("keep-plan", "propose_goal_replan", {
            "action": "keep", "summary": "The second stage remains ready", "plan": None,
        })]),
        LLMResponse(content="Plan kept"),
        LLMResponse(tool_calls=[ToolCall("read-second", "read_file", {"file_path": str(tmp_path / "second.txt")})]),
        LLMResponse(tool_calls=[ToolCall("claim", GOAL_COMPLETION_TOOL_NAME, {
            "summary": "Both files inspected", "evidence_tool_call_ids": ["read-second"],
        })]),
        LLMResponse(content="Second file inspected"),
    ])
    chat, workflow, output = _chat(tmp_path, provider)

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.status is LongTaskStatus.SUCCEEDED
    assert task.verified_step_ids == ("first", "second")
    assert len(task.checkpoints) == 2
    assert task.last_continuous_stop == {
        "reason": "completed", "turns": 2, "tokens": 0, "cost_usd": None,
    }
    assert any(
        "Execute the next action of confirmed Goal stage first now" in message.get("content", "")
        and "Next action: Read first.txt" in message.get("content", "")
        for message in provider.requests[0]["messages"]
    )
    assert any(
        "Execute the next action of confirmed Goal stage second now" in message.get("content", "")
        for request in provider.requests for message in request["messages"]
    )
    assert "连续执行停止：completed" in output.getvalue()


def test_explicit_run_requires_a_plan_and_does_not_call_the_model(tmp_path):
    provider = FixedProvider([])
    chat, workflow, output = _chat(tmp_path, provider, plan=False)

    chat._goal("run")

    assert provider.requests == []
    assert workflow.store.replay(workflow.task_id).checkpoints == []
    assert workflow.store.replay(workflow.task_id).last_continuous_stop is None
    assert "需要已确认的结构化阶段计划" in output.getvalue()


def test_ordinary_goal_continuation_stops_after_one_verified_stage(tmp_path):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="First file inspected"),
        LLMResponse(tool_calls=[ToolCall("keep-plan", "propose_goal_replan", {
            "action": "keep", "summary": "The second stage remains ready", "plan": None,
        })]),
        LLMResponse(content="Plan kept"),
    ])
    chat, workflow, _output = _chat(tmp_path, provider)

    chat._advance_goal("Inspect the first file")

    task = workflow.store.replay(workflow.task_id)
    assert task.status is LongTaskStatus.RUNNING
    assert task.verified_step_ids == ("first",)
    assert task.ready_step_ids == ("second",)
    assert len(task.checkpoints) == 1
    assert len(provider.requests) == 4


def test_interrupted_run_records_a_stop_without_another_stage(tmp_path):
    provider = FixedProvider([KeyboardInterrupt()])
    chat, workflow, _output = _chat(tmp_path, provider)

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.last_continuous_stop is not None
    assert task.last_continuous_stop["reason"] == "interrupted"
    assert task.last_continuous_stop["turns"] == 0
    assert task.verified_step_ids == ()
    assert len(provider.requests) == 1


def test_interruption_after_verified_stage_does_not_start_next_side_effect(tmp_path):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="First file inspected"),
        LLMResponse(tool_calls=[ToolCall("keep-plan", "propose_goal_replan", {
            "action": "keep", "summary": "The second stage remains ready", "plan": None,
        })]),
        LLMResponse(content="Plan kept"),
        KeyboardInterrupt(),
    ])
    chat, workflow, _output = _chat(tmp_path, provider)

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.last_continuous_stop == {
        "reason": "interrupted", "turns": 1, "tokens": 0, "cost_usd": None,
    }
    assert task.verified_step_ids == ("first",)
    assert len(task.checkpoints) == 1
    assert len(provider.requests) == 5
    assert all(
        event.payload.get("arguments", {}).get("file_path") != str(tmp_path / "second.txt")
        for event in workflow.runtime.session_store.replay(workflow.runtime.agent.session_id).events
        if event.event_type == RuntimeEventType.TOOL_REQUESTED.value
    )


def test_cancellation_after_a_write_prevents_the_next_write(tmp_path):
    cancel_results = []

    class AllowPrompt:
        def decide(self, _request):
            return PermissionDecision.allow("G4 cancellation fixture")

    provider = FixedProvider([LLMResponse(tool_calls=[
        ToolCall("write-first", "write_file", {
            "file_path": str(tmp_path / "first.txt"), "content": "changed first\n",
        }),
        ToolCall("write-second", "write_file", {
            "file_path": str(tmp_path / "second.txt"), "content": "changed second\n",
        }),
    ])])
    chat, workflow, output = _chat(
        tmp_path, provider, tools=[WriteFileTool()], permission_prompt=AllowPrompt(),
    )
    delegate = workflow.runtime.agent.tool_executor.delegate
    original_write = delegate._execute_write

    def cancel_after_first_write(tool, normalized, tool_call_id, assessment):
        result = original_write(tool, normalized, tool_call_id, assessment)
        if tool_call_id == "write-first":
            cancel_results.append(workflow.runtime.agent.cancel_current_turn("cancel after first write"))
        return result

    delegate._execute_write = cancel_after_first_write

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert cancel_results == [True]
    assert task.last_continuous_stop is not None
    assert task.last_continuous_stop["reason"] in {"interrupted", "recovery_required"}, output.getvalue()
    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "changed first\n"
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "second\n"
    assert len(provider.requests) == 1


def test_invalid_stop_event_fails_closed_on_replay(tmp_path):
    chat, workflow, _output = _chat(tmp_path, FixedProvider([]))
    del chat
    workflow.store.append(LongTaskEvent(
        task_id=workflow.task_id,
        event_type="goal_run_stopped",
        payload={"reason": "completed", "turns": -1, "tokens": 0, "cost_usd": None},
    ))

    task = workflow.store.replay(workflow.task_id)
    assert task.status is LongTaskStatus.RECOVERY_REQUIRED
    assert task.recovery_blockers == ["goal_run_event_invalid"]


def test_unknown_effect_stops_before_any_continuous_model_turn(tmp_path):
    provider = FixedProvider([])
    chat, workflow, _output = _chat(tmp_path, provider)
    runtime = workflow.runtime
    arguments = {"file_path": "unknown.txt", "content": "unknown\n"}
    runtime.session_store.append(SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id=runtime.agent.session_id,
        turn_id="interrupted-turn",
        tool_call_id="unknown-write",
        payload={"tool_name": "write_file", "arguments": arguments},
    ))
    journal = EffectJournalStore.for_session_directory(runtime.session_store.directory)
    journal.start(
        session_id=runtime.agent.session_id,
        tool_call_id="unknown-write",
        tool_name="write_file",
        arguments=arguments,
        effect=ToolEffect.WRITE,
    )

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert provider.requests == []
    assert task.status is LongTaskStatus.RECOVERY_REQUIRED
    assert task.last_continuous_stop is not None
    assert task.last_continuous_stop["reason"] == "recovery_required"
    assert not (tmp_path / "unknown.txt").exists()


def test_runtime_provider_budget_stops_continuous_run(tmp_path):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
    ])
    chat, workflow, output = _chat(tmp_path, provider, limits=RuntimeLimits(max_provider_calls=1))

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.last_continuous_stop is not None
    assert task.last_continuous_stop["reason"] == "budget_exhausted"
    assert len(provider.requests) == 1
    assert task.verified_step_ids == ("first",)
    assert task.ready_step_ids == ("second",)
    assert len(task.checkpoints) == 1
    assert "Runtime limit and saved" in output.getvalue()
    assert "adjust the limit before /goal run" in output.getvalue()


def test_planning_turn_budget_stops_before_next_stage(tmp_path):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="First file inspected"),
        LLMResponse(tool_calls=[ToolCall("inspect-plan", "read_file", {"file_path": str(tmp_path / "second.txt")})]),
        LLMResponse(tool_calls=[ToolCall("keep-plan", "propose_goal_replan", {
            "action": "keep", "summary": "The second stage remains ready", "plan": None,
        })]),
    ])
    chat, workflow, _output = _chat(tmp_path, provider, limits=RuntimeLimits(max_provider_calls=2))

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.last_continuous_stop is not None
    assert task.last_continuous_stop["reason"] == "budget_exhausted"
    assert task.verified_step_ids == ("first",)
    assert len(task.checkpoints) == 1
    assert len(provider.requests) == 4


def test_explicit_run_stops_at_four_stage_turns_with_next_stage_ready(tmp_path):
    responses = []
    for name in ("first", "second", "third", "fourth"):
        responses.extend([
            LLMResponse(tool_calls=[ToolCall(f"read-{name}", "read_file", {
                "file_path": str(tmp_path / f"{name}.txt"),
            })]),
            LLMResponse(content=f"{name} inspected"),
            LLMResponse(tool_calls=[ToolCall(f"keep-{name}", "propose_goal_replan", {
                "action": "keep", "summary": "Next stage remains ready", "plan": None,
            })]),
            LLMResponse(content="Plan kept"),
        ])
    provider = FixedProvider(responses)
    chat, workflow, _output = _chat(tmp_path, provider, stage_count=5)

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.last_continuous_stop == {
        "reason": "round_limit", "turns": 4, "tokens": 0, "cost_usd": None,
    }
    assert task.verified_step_ids == ("first", "second", "third", "fourth")
    assert task.ready_step_ids == ("fifth",)
    assert len(task.checkpoints) == 4
    assert len(provider.requests) == 16


def test_explicit_run_stops_when_final_claim_has_no_new_tool_evidence(tmp_path):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="First file inspected"),
        LLMResponse(tool_calls=[ToolCall("keep-plan", "propose_goal_replan", {
            "action": "keep", "summary": "The second stage remains ready", "plan": None,
        })]),
        LLMResponse(content="Plan kept"),
        LLMResponse(tool_calls=[ToolCall("read-second", "read_file", {"file_path": str(tmp_path / "second.txt")})]),
        LLMResponse(content="Second file inspected, but no completion claim"),
        LLMResponse(content="No further evidence"),
    ])
    chat, workflow, _output = _chat(tmp_path, provider)

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.status is LongTaskStatus.RUNNING
    assert task.last_continuous_stop is not None
    assert task.last_continuous_stop["reason"] == "no_progress"
    assert task.verified_step_ids == ("first", "second")
    assert task.active_completion_claim is None
    assert len(provider.requests) == 7


def test_revised_plan_stops_before_the_next_stage(tmp_path):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="First file inspected"),
        LLMResponse(tool_calls=[ToolCall("replace-plan", "propose_goal_replan", {
            "action": "replace", "summary": "Use a different second-stage check",
            "plan": {"reason": "New evidence", "steps": [
                _step("first", "first.txt"),
                _step("second", "third.txt", ["first"]),
            ]},
        })]),
        LLMResponse(content="Plan revised"),
    ])
    chat, workflow, _output = _chat(tmp_path, provider, stage_count=3)

    chat._goal("run")

    task = workflow.store.replay(workflow.task_id)
    assert task.last_continuous_stop is not None
    assert task.last_continuous_stop["reason"] == "planning_boundary"
    assert task.verified_step_ids == ("first",)
    assert task.plan is not None and task.plan.version == 2
    assert len(task.checkpoints) == 1
    assert len(provider.requests) == 4
