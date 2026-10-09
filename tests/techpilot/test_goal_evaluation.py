"""G2 acceptance checks for independent, durable Goal stage assessment."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from techpilot.engine.events import CallbackEventSink
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.tools import BashTool, WriteFileTool
from techpilot.runtime import (
    GoalStageAssessmentStatus,
    LongTaskRecoveryRequired,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskWorkflow,
    RuntimeBootstrap,
    RuntimeBootstrapInput,
)


class FixedProvider:
    model = "fixed-goal-evaluator-provider"
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


class PermissionPrompt:
    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow

    def decide(self, request):
        del request
        return PermissionDecision.allow("fixture approval") if self.allow else PermissionDecision.deny("fixture rejection")


@dataclass
class CountingExecutor:
    delegate: object
    calls: int = 0

    def begin_turn(self):
        return self.delegate.begin_turn()

    def consume_turn_stop_message(self):
        return self.delegate.consume_turn_stop_message()

    def describe_call(self, tool, arguments):
        return self.delegate.describe_call(tool, arguments)

    def execute_call(self, tool, arguments, *, tool_call_id, execution_context=None):
        self.calls += 1
        return self.delegate.execute_call(
            tool, arguments, tool_call_id=tool_call_id, execution_context=execution_context,
        )


def _runtime(root: Path, provider: FixedProvider, *, session_id: str | None = None, tools=None, allow=True):
    runtime = RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=root,
        event_sink=CallbackEventSink(lambda _event: None),
        tools=tools or [WriteFileTool()],
        model="fixed-goal-evaluator-provider",
        permission_prompt=PermissionPrompt(allow=allow),
        session_directory=root / "sessions",
        resume_session_id=session_id,
    ))
    runtime.agent.tool_executor = CountingExecutor(runtime.agent.tool_executor)
    return runtime


def _step(step_id: str, check: str | dict, dependencies: list[str] | None = None) -> dict:
    return {
        "id": step_id,
        "goal": f"Complete {step_id}",
        "dependencies": dependencies or [],
        "expected_output": f"{step_id} result",
        "work_scope": ["result.txt"],
        "evidence": "Durable Session Tool events",
        "check": check,
        "next_action": f"Work on {step_id}",
    }


def _planned(runtime, *steps: dict) -> LongTaskWorkflow:
    workflow = LongTaskWorkflow(runtime, task_id="goal-evaluation", goal="Complete the fixture")
    workflow.start_or_resume()
    workflow.store.record_goal_plan(workflow.task_id, {"reason": "G2 fixture plan", "steps": list(steps)})
    workflow.store.select_goal_step(workflow.task_id, steps[0]["id"])
    return workflow


def _write_call(call_id: str, content: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(
        id=call_id,
        name="write_file",
        arguments={"file_path": "result.txt", "content": content},
    )])


def test_file_check_verifies_durable_evidence_and_resume_does_not_rerun_tool(tmp_path):
    provider = FixedProvider([_write_call("write-good", "verified result\n"), LLMResponse(content="done")])
    runtime = _runtime(tmp_path, provider)
    workflow = _planned(
        runtime,
        _step("write", {"kind": "file_contains", "path": "result.txt", "text": "verified result"}),
        _step("next", "Review next stage", ["write"]),
    )
    workflow.run_step("write the file")
    requests_before = len(provider.requests)
    tool_calls_before = runtime.agent.tool_executor.calls

    task, assessment = workflow.evaluate_selected_step()

    assert assessment.status is GoalStageAssessmentStatus.VERIFIED
    assert assessment.tool_call_ids == ("write-good",)
    assert assessment.session_event_cursor in assessment.evidence_event_ids
    assert task.verified_step_ids == ("write",)
    assert task.ready_step_ids == ("next",)
    assert len(provider.requests) == requests_before
    assert runtime.agent.tool_executor.calls == tool_calls_before == 1
    with pytest.raises(LongTaskStateError, match="unverified stages: next"):
        workflow.finish(require_completion_claim=False)

    resumed_provider = FixedProvider([])
    resumed = _runtime(tmp_path, resumed_provider, session_id=runtime.agent.session_id)
    resumed_workflow = LongTaskWorkflow(resumed, task_id=workflow.task_id, goal="ignored constructor text")
    replayed = resumed_workflow.start_or_resume()
    assert replayed.verified_step_ids == ("write",)
    assert replayed.ready_step_ids == ("next",)
    assert resumed_provider.requests == []
    assert resumed.agent.tool_executor.calls == 0
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "verified result\n"


def test_failed_file_check_stays_blocked_until_explicit_same_stage_retry(tmp_path):
    provider = FixedProvider([
        _write_call("write-wrong", "wrong\n"), LLMResponse(content="I finished"),
        _write_call("write-correct", "expected\n"), LLMResponse(content="corrected"),
    ])
    runtime = _runtime(tmp_path, provider)
    workflow = _planned(
        runtime,
        _step("write", {"kind": "file_contains", "path": "result.txt", "text": "expected"}),
        _step("next", "Review next", ["write"]),
    )
    workflow.run_step("first attempt")
    task, failed = workflow.evaluate_selected_step()
    assert failed.status is GoalStageAssessmentStatus.BLOCKED
    assert failed.tool_call_ids == ("write-wrong",)
    assert failed.session_event_cursor in failed.evidence_event_ids
    assert task.ready_step_ids == ()
    assert task.verified_step_ids == ()
    with pytest.raises(LongTaskStateError, match="requires review or replanning"):
        workflow.run_step("unsafe automatic retry")
    with pytest.raises(LongTaskStateError, match="unevaluated checkpoint"):
        workflow.evaluate_selected_step()

    workflow.store.select_goal_step(workflow.task_id, "write")
    assert workflow.store.replay(workflow.task_id).selected_step_assessment is None
    workflow.run_step("correct the file")
    task, passed = workflow.evaluate_selected_step()
    assert passed.status is GoalStageAssessmentStatus.VERIFIED
    assert task.ready_step_ids == ("next",)
    assert [item.status for item in task.stage_assessments] == [
        GoalStageAssessmentStatus.BLOCKED, GoalStageAssessmentStatus.VERIFIED,
    ]
    assert runtime.agent.tool_executor.calls == 2


@pytest.mark.parametrize(
    ("requested", "expected", "accepted"),
    [
        ("echo verified", "echo verified", True),
        ("exit 7", "exit 7", False),
        ("echo other", "echo verified", False),
    ],
)
def test_exact_test_command_requires_matching_exit_zero_result(tmp_path, requested, expected, accepted):
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("bash-check", "bash", {"command": requested})]),
        LLMResponse(content="command finished"),
    ])
    runtime = _runtime(tmp_path, provider, tools=[BashTool()])
    workflow = _planned(runtime, _step("check", {"kind": "test_command", "command": expected}))
    workflow.run_step("run the check")

    task, assessment = workflow.evaluate_selected_step()

    assert assessment.status is (GoalStageAssessmentStatus.VERIFIED if accepted else GoalStageAssessmentStatus.BLOCKED)
    assert assessment.session_event_cursor in assessment.evidence_event_ids
    assert "bash-check" in assessment.tool_call_ids
    assert task.verified_step_ids == (("check",) if accepted else ())


def test_permission_rejection_and_model_claim_cannot_verify_file_stage(tmp_path):
    (tmp_path / "result.txt").write_text("preexisting\n", encoding="utf-8")
    provider = FixedProvider([_write_call("denied-write", "new\n"), LLMResponse(content="I completed it")])
    runtime = _runtime(tmp_path, provider, allow=False)
    workflow = _planned(runtime, _step("write", {"kind": "file_exists", "path": "result.txt"}))
    workflow.run_step("write the file")

    task, assessment = workflow.evaluate_selected_step()

    assert assessment.status is GoalStageAssessmentStatus.BLOCKED
    assert task.verified_step_ids == ()
    assert task.selected_step_id == "write"
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "preexisting\n"


def test_natural_language_check_requires_explicit_user_review(tmp_path):
    runtime = _runtime(tmp_path, FixedProvider([LLMResponse(content="I completed it")]))
    workflow = _planned(
        runtime,
        _step("inspect", "Review the investigation"),
        _step("next", "Review next", ["inspect"]),
    )
    workflow.run_step("inspect")
    task, assessment = workflow.evaluate_selected_step()
    assert assessment.status is GoalStageAssessmentStatus.NEEDS_REVIEW
    assert assessment.tool_call_ids == ()
    assert task.ready_step_ids == ()
    with pytest.raises(LongTaskStateError, match="requires review or replanning"):
        workflow.run_step("continue without review")

    reviewed = workflow.review_selected_step(GoalStageAssessmentStatus.VERIFIED, reason="I checked the investigation")
    assert reviewed.verified_step_ids == ("inspect",)
    assert reviewed.ready_step_ids == ("next",)
    assert reviewed.stage_assessments[-1].source == "user_review"
    with pytest.raises(LongTaskStateError, match="Only a needs_review"):
        workflow.review_selected_step(GoalStageAssessmentStatus.BLOCKED, reason="change mind")


def test_missing_session_cursor_requires_recovery_without_rerunning_tool(tmp_path):
    provider = FixedProvider([_write_call("write-once", "durable\n"), LLMResponse(content="done")])
    runtime = _runtime(tmp_path, provider)
    workflow = _planned(runtime, _step("write", {"kind": "file_exists", "path": "result.txt"}))
    workflow.run_step("write once")
    checkpoint_cursor = workflow.store.replay(workflow.task_id).checkpoints[-1].session_event_cursor
    session_path = runtime.session_store.directory / f"{runtime.agent.session_id}.jsonl"
    records = [json.loads(line) for line in session_path.read_text(encoding="utf-8").splitlines()]
    session_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records if record["event_id"] != checkpoint_cursor),
        encoding="utf-8",
    )
    tool_calls_before = runtime.agent.tool_executor.calls
    requests_before = len(provider.requests)

    with pytest.raises(LongTaskRecoveryRequired, match="cursor"):
        workflow.evaluate_selected_step()

    assert workflow.store.replay(workflow.task_id).status is LongTaskStatus.RECOVERY_REQUIRED
    assert runtime.agent.tool_executor.calls == tool_calls_before == 1
    assert len(provider.requests) == requests_before


def test_process_exit_leaves_checkpoint_for_read_only_assessment(tmp_path):
    child_code = """
import json
import sys
from pathlib import Path
from techpilot.engine.events import CallbackEventSink
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.tools import WriteFileTool
from techpilot.runtime import LongTaskWorkflow, RuntimeBootstrap, RuntimeBootstrapInput

class Provider:
    model = 'fixed-goal-evaluator-provider'
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None
    def __init__(self):
        self.responses = iter([
            LLMResponse(tool_calls=[ToolCall('write-child', 'write_file', {
                'file_path': 'result.txt', 'content': 'from child process\\n',
            })]),
            LLMResponse(content='written'),
        ])
    def chat(self, messages, tools=None, on_token=None):
        return next(self.responses)

class AllowPrompt:
    def decide(self, request):
        return PermissionDecision.allow('fixture approval')

root = Path(sys.argv[1])
runtime = RuntimeBootstrap(provider_factory=lambda _config: Provider()).build(RuntimeBootstrapInput(
    repository=root,
    event_sink=CallbackEventSink(lambda _event: None),
    tools=[WriteFileTool()],
    model='fixed-goal-evaluator-provider',
    permission_prompt=AllowPrompt(),
    session_directory=root / 'sessions',
))
workflow = LongTaskWorkflow(runtime, task_id='child-goal', goal='write the result')
workflow.start_or_resume()
workflow.store.record_goal_plan('child-goal', {'reason': 'process recovery fixture', 'steps': [{
    'id': 'write', 'goal': 'Write result', 'dependencies': [],
    'expected_output': 'result.txt', 'work_scope': ['result.txt'],
    'evidence': 'Session Tool event',
    'check': {'kind': 'file_contains', 'path': 'result.txt', 'text': 'from child process'},
    'next_action': 'Assess the checkpoint',
}]})
workflow.store.select_goal_step('child-goal', 'write')
workflow.run_step('write the result')
print(json.dumps({'session_id': runtime.agent.session_id}))
"""
    child = subprocess.run(
        [sys.executable, "-B", "-c", child_code, str(tmp_path)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    session_id = json.loads(child.stdout)["session_id"]
    provider = FixedProvider([])
    resumed = _runtime(tmp_path, provider, session_id=session_id)
    workflow = LongTaskWorkflow(resumed, task_id="child-goal", goal="ignored constructor text")
    assert workflow.store.replay(workflow.task_id).selected_step_needs_evaluation

    task, assessment = workflow.evaluate_selected_step()

    assert assessment.status is GoalStageAssessmentStatus.VERIFIED
    assert assessment.tool_call_ids == ("write-child",)
    assert task.verified_step_ids == ("write",)
    assert provider.requests == []
    assert resumed.agent.tool_executor.calls == 0
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "from child process\n"
