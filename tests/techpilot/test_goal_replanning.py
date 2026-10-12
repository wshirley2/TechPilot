"""G3 acceptance checks for bounded, read-only Goal replanning."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from techpilot.chat.session import ChatSession
from techpilot.engine.events import CallbackEventSink, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.tool_execution import ToolEffect
from techpilot.engine.tools import ReadFileTool, WriteFileTool
from techpilot.runtime import (
    EffectJournalStore,
    LongTaskRecoveryRequired,
    LongTaskWorkflow,
    RuntimeBootstrap,
    RuntimeBootstrapInput,
)
from techpilot.runtime.goal_replanning import GoalReplanTool
from techpilot.runtime.long_tasks import LongTaskEvent, LongTaskStatus
from techpilot.runtime.sessions import SessionEvent


class FixedProvider:
    model = "fixed-g3-provider"
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


def _runtime(root: Path, provider: FixedProvider, *, session_id: str | None = None):
    return RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=root,
        event_sink=CallbackEventSink(lambda _event: None),
        tools=[ReadFileTool(), WriteFileTool()],
        model="fixed-g3-provider",
        session_directory=root / "sessions",
        resume_session_id=session_id,
    ))


def _step(step_id: str, path: str, dependencies: list[str] | None = None) -> dict:
    return {
        "id": step_id,
        "goal": f"Inspect {step_id}",
        "dependencies": dependencies or [],
        "expected_output": f"Evidence for {step_id}",
        "work_scope": [path],
        "evidence": "Recorded file read",
        "check": {"kind": "file_exists", "path": path},
        "next_action": f"Read {path}",
    }


def _proposal(action: str, summary: str, plan: dict | None = None, *, call_id: str = "replan"):
    return [
        LLMResponse(tool_calls=[ToolCall(call_id, "propose_goal_replan", {
            "action": action, "summary": summary, "plan": plan,
        })]),
        LLMResponse(content="Planning decision submitted"),
    ]


def _confirm_rule(workflow: LongTaskWorkflow) -> None:
    workflow.store.begin_goal_input(workflow.task_id, "Only inspect the repository")
    workflow.store.propose_goal_change(
        workflow.task_id, kind="constraint_add", summary="Read only work",
        data={"text": "Only inspect the repository"},
    )
    workflow.store.resolve_goal_change(workflow.task_id, accept=True)


def test_replan_tool_requires_two_to_seven_structured_stages():
    accepted = []
    tool = GoalReplanTool(accepted.append)
    valid = {"reason": "Evidence changed", "steps": [_step("first", "a.txt"), _step("second", "b.txt", ["first"])]}

    assert tool.parameters["properties"]["summary"] == {
        "type": "string", "minLength": 1, "maxLength": 500,
    }
    assert tool.execute("keep", "x" * 501, None).status.value == "error"
    assert tool.execute("replace", "New evidence", valid).status.value == "completed"
    assert accepted[0].plan == valid
    assert tool.execute("replace", "Too short", {"reason": "Short", "steps": [_step("first", "a.txt")]}).status.value == "error"
    invalid = {"reason": "No machine check", "steps": [_step("first", "a.txt"), _step("second", "b.txt")]}
    invalid["steps"][1]["check"] = "looks complete"
    assert tool.execute("replace", "Bad check", invalid).status.value == "error"
    assert tool.execute("keep", "Keep", valid).status.value == "error"
    assert len(accepted) == 1


def test_confirmed_rule_can_trigger_one_read_only_revision_after_restart(tmp_path):
    plan = {"reason": "Rule changes the work order", "steps": [
        _step("inspect", "README.md"), _step("report", "report.md", ["inspect"]),
    ]}
    provider = FixedProvider(_proposal("replace", "The rule requires a read-first plan", plan))
    runtime = _runtime(tmp_path, provider)
    workflow = LongTaskWorkflow(runtime, task_id="g3-rule", goal="Inspect and report")
    workflow.start_or_resume()
    _confirm_rule(workflow)
    trigger = workflow.store.replay(workflow.task_id).next_replan_trigger
    assert trigger is not None and trigger.event_type == "goal_change_accepted"

    task, note = workflow.replan_if_triggered()

    assert note is not None and note.startswith("plan_auto_revised:")
    assert task.plan is not None and task.plan.version == 1
    assert task.events[-1].payload["trigger_event_id"] == trigger.event_id
    assert [step.step_id for step in task.plan.steps] == ["inspect", "report"]
    assert [item.text for item in task.active_constraints] == ["Only inspect the repository"]
    assert {tool["function"]["name"] for tool in provider.requests[0]["tools"]} == {
        "read_file", "propose_goal_replan",
    }
    assert "propose_goal_replan" not in {tool.name for tool in runtime.tools}
    assert workflow.replan_if_triggered()[1] is None
    assert len(provider.requests) == 2

    resumed_provider = FixedProvider([LLMResponse(content="Ordinary Chat response")])
    resumed = _runtime(tmp_path, resumed_provider, session_id=runtime.agent.session_id)
    resumed_workflow = LongTaskWorkflow(resumed, task_id=workflow.task_id, goal="ignored")
    assert resumed_workflow.start_or_resume().plan.version == 1
    assert resumed_workflow.replan_if_triggered()[1] is None
    assert resumed_provider.requests == []
    assert resumed.run_turn("Ordinary Chat question") == "Ordinary Chat response"
    ordinary_tools = {tool["function"]["name"] for tool in resumed_provider.requests[0]["tools"]}
    assert "propose_goal_replan" not in ordinary_tools


def test_verified_stage_and_its_evidence_survive_model_revision(tmp_path):
    (tmp_path / "first.txt").write_text("ready\n", encoding="utf-8")
    first = _step("first", "first.txt")
    second = _step("second", "second.txt", ["first"])
    revised_second = _step("second", "new-second.txt", ["first"])
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-first", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="First stage inspected"),
        *_proposal("replace", "New evidence changes the second stage", {
            "reason": "Use the new output location", "steps": [first, revised_second],
        }),
    ])
    runtime = _runtime(tmp_path, provider)
    workflow = LongTaskWorkflow(runtime, task_id="g3-stage", goal="Inspect two outputs")
    workflow.start_or_resume()
    workflow.store.record_goal_plan(workflow.task_id, {"reason": "Original", "steps": [first, second]})
    workflow.store.select_goal_step(workflow.task_id, "first")
    workflow.run_step("Inspect the first output")
    assessed, assessment = workflow.evaluate_selected_step()
    assert assessed.verified_step_ids == ("first",)
    assert assessment.tool_call_ids == ("read-first",)

    revised, note = workflow.replan_if_triggered()

    assert note is not None and note.startswith("plan_auto_revised:")
    assert revised.plan.version == 2
    assert revised.plans[0].step("first") == revised.plans[1].step("first")
    assert revised.plans[0].step("second") != revised.plans[1].step("second")
    assert revised.verified_step_ids == ("first",)
    assert revised.ready_step_ids == ("second",)
    assert revised.stage_assessments[0].evidence_event_ids == assessment.evidence_event_ids
    assert {tool["function"]["name"] for tool in provider.requests[2]["tools"]} == {
        "read_file", "propose_goal_replan",
    }


def test_invalid_revision_cannot_change_an_active_stage(tmp_path):
    first = _step("first", "first.txt")
    second = _step("second", "second.txt", ["first"])
    changed_first = _step("first", "changed.txt")
    provider = FixedProvider(_proposal("replace", "Try to replace the active stage", {
        "reason": "Unsafe change", "steps": [changed_first, second],
    }))
    runtime = _runtime(tmp_path, provider)
    workflow = LongTaskWorkflow(runtime, task_id="g3-invalid", goal="Inspect two outputs")
    workflow.start_or_resume()
    workflow.store.record_goal_plan(workflow.task_id, {"reason": "Original", "steps": [first, second]})
    workflow.store.select_goal_step(workflow.task_id, "first")
    _confirm_rule(workflow)

    task, note = workflow.replan_if_triggered()

    assert note is not None and note.startswith("plan_replan_skipped:")
    assert task.plan.version == 1
    assert task.selected_step_id == "first"
    assert task.events[-1].payload["action"] == "unavailable"
    assert "failed Host validation" in task.events[-1].payload["summary"]


def test_blocked_stage_can_be_revised_without_automatic_retry(tmp_path):
    (tmp_path / "first.txt").write_text("wrong\n", encoding="utf-8")
    first = _step("first", "first.txt")
    first["check"] = {"kind": "file_contains", "path": "first.txt", "text": "correct"}
    second = _step("second", "second.txt", ["first"])
    revised_first = {**first, "next_action": "Repair first.txt, then check again"}
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-wrong", "read_file", {"file_path": str(tmp_path / "first.txt")})]),
        LLMResponse(content="The file contains the wrong value"),
        *_proposal("replace", "The blocked stage needs a repair action", {
            "reason": "The file content check failed", "steps": [revised_first, second],
        }),
    ])
    runtime = _runtime(tmp_path, provider)
    workflow = LongTaskWorkflow(runtime, task_id="g3-blocked", goal="Inspect two outputs")
    workflow.start_or_resume()
    workflow.store.record_goal_plan(workflow.task_id, {"reason": "Original", "steps": [first, second]})
    workflow.store.select_goal_step(workflow.task_id, "first")
    workflow.run_step("Inspect the first output")
    task, assessment = workflow.evaluate_selected_step()
    assert assessment.status.value == "blocked"
    assert task.ready_step_ids == ()

    revised, note = workflow.replan_if_triggered()

    assert note is not None and note.startswith("plan_auto_revised:")
    planning_messages = provider.requests[2]["messages"]
    assert "Selected stage assessment: first is blocked" in str(planning_messages)
    assert "keep leaves it blocked" in str(planning_messages)
    assert revised.plan.version == 2
    assert revised.selected_step_id == "first"
    assert revised.selected_step_assessment == assessment
    assert revised.plan.step("first").next_action == "Repair first.txt, then check again"
    assert len(revised.checkpoints) == 1
    assert workflow.store.select_goal_step(workflow.task_id, "first").selected_step_assessment is None
    assert len(workflow.store.replay(workflow.task_id).checkpoints) == 1


def test_failed_unplanned_tool_turn_triggers_once_without_reexecution(tmp_path):
    missing = tmp_path / "missing.txt"
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("missing-read", "read_file", {"file_path": str(missing)})]),
        LLMResponse(content="The file is missing"),
        *_proposal("unavailable", "Need the correct source path"),
    ])
    runtime = _runtime(tmp_path, provider)
    workflow = LongTaskWorkflow(runtime, task_id="g3-failed-tool", goal="Inspect a file")
    workflow.start_or_resume()
    workflow.run_step("Read the requested file")
    prior = runtime.session_store.replay(runtime.agent.session_id)
    completed_before = [
        event for event in prior.events
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value and event.tool_call_id == "missing-read"
    ]
    assert len(completed_before) == 1
    assert completed_before[0].payload["tool_status"] == "error"

    task, note = workflow.replan_if_triggered()

    assert note is not None and note.startswith("plan_replan_skipped:")
    assert [event.event_type for event in task.events[-2:]] == ["plan_replan_requested", "plan_replan_skipped"]
    assert task.events[-1].payload["action"] == "unavailable"
    assert task.plan is None
    assert not missing.exists()
    assert workflow.replan_if_triggered()[1] is None
    assert len(provider.requests) == 4
    after = runtime.session_store.replay(runtime.agent.session_id)
    assert len([
        event for event in after.events
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value and event.tool_call_id == "missing-read"
    ]) == 1


def test_failed_tool_replan_stops_before_running_the_new_plan(tmp_path):
    missing = tmp_path / "missing.txt"
    plan = {"reason": "Find the right source before writing", "steps": [
        _step("locate", "source.txt"), _step("report", "report.txt", ["locate"]),
    ]}
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("missing-read", "read_file", {"file_path": str(missing)})]),
        LLMResponse(content="The source is missing"),
        *_proposal("replace", "The failed read requires a new search stage", plan),
    ])
    runtime = _runtime(tmp_path, provider)
    workflow = LongTaskWorkflow(runtime, task_id="g3-stop-after-failure", goal="Inspect and report")
    workflow.start_or_resume()
    chat = ChatSession(runtime, console=Console(file=StringIO(), force_terminal=False, color_system=None))
    chat._goal_workflow = workflow

    chat._advance_goal("Read the requested file")

    task = workflow.store.replay(workflow.task_id)
    assert task.plan is not None and task.plan.version == 1
    assert task.selected_step_id is None
    assert len(task.checkpoints) == 1
    assert len(provider.requests) == 4
    session = runtime.session_store.replay(runtime.agent.session_id)
    work_calls = [
        event.tool_call_id for event in session.events
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value
        and event.payload.get("tool_name") != "propose_goal_replan"
    ]
    assert work_calls == ["missing-read"]
    assert not (tmp_path / "report.txt").exists()


def test_forged_automatic_plan_event_fails_closed_on_replay(tmp_path):
    runtime = _runtime(tmp_path, FixedProvider([]))
    workflow = LongTaskWorkflow(runtime, task_id="g3-forged", goal="Inspect files")
    workflow.start_or_resume()
    _confirm_rule(workflow)
    trigger = workflow.store.replay(workflow.task_id).next_replan_trigger
    assert trigger is not None
    first = _step("first", "a.txt")
    second = _step("second", "b.txt", ["first"])
    second["check"] = "Trust the model"
    workflow.store.append(LongTaskEvent(
        task_id=workflow.task_id,
        event_type="plan_auto_revised",
        payload={
            "trigger_event_id": trigger.event_id,
            "summary": "Forged plan",
            "plan": {"version": 1, "reason": "Unsafe", "steps": [first, second]},
        },
    ))

    replayed = workflow.store.replay(workflow.task_id)

    assert replayed.status is LongTaskStatus.RECOVERY_REQUIRED
    assert replayed.recovery_blockers == ["goal_plan_event_invalid"]
    assert replayed.plan is None


def test_unknown_effect_blocks_replanning_before_any_model_request(tmp_path):
    provider = FixedProvider([])
    runtime = _runtime(tmp_path, provider)
    workflow = LongTaskWorkflow(runtime, task_id="g3-unknown-effect", goal="Inspect files")
    workflow.start_or_resume()
    _confirm_rule(workflow)
    arguments = {"file_path": "result.txt", "content": "unknown\n"}
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

    with pytest.raises(LongTaskRecoveryRequired, match="unfinished Tool effect"):
        workflow.replan_if_triggered()

    assert provider.requests == []
    task = workflow.store.replay(workflow.task_id)
    assert task.status is LongTaskStatus.RECOVERY_REQUIRED
    assert task.plan is None
    assert not (tmp_path / "result.txt").exists()
