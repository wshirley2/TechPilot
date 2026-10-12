"""Read-only assessment of one checkpointed Goal stage.

This module reads durable Runtime facts and current files. It never executes a
Tool Call, repeats a command, or treats model prose as verification.
"""

from __future__ import annotations

from pathlib import Path

from techpilot.engine.events import RuntimeEventType
from techpilot.engine.tool_results import ToolStatus, display_tool_status

from .effect_journal import EffectJournalStore
from .goal_plans import GoalStageAssessment, GoalStageAssessmentStatus, GoalStageCheck, GoalStageCheckKind
from .long_tasks import LongTaskProjection, LongTaskStateError
from .sessions import SessionEvent, SessionStore


class GoalStageEvidenceError(LongTaskStateError):
    """A checkpoint or referenced Session/Effect fact cannot be trusted."""


def evaluate_goal_stage(
    task: LongTaskProjection,
    *,
    session_store: SessionStore,
    effect_store: EffectJournalStore,
) -> GoalStageAssessment:
    """Assess one selected stage without starting another Runtime turn."""

    plan = task.plan
    step = plan.step(task.selected_step_id) if plan and task.selected_step_id else None
    if step is None or not task.selected_step_needs_evaluation or not task.checkpoints:
        raise LongTaskStateError("Goal stage needs a selected, unevaluated checkpoint")
    if task.session_id is None or task.repository_root is None:
        raise GoalStageEvidenceError("Goal is missing its Session or repository identity")
    try:
        session = session_store.replay(task.session_id)
        journal = effect_store.replay(task.session_id)
    except OSError as error:
        raise GoalStageEvidenceError(f"Could not read durable stage evidence: {error}") from error
    if session.warnings or session.repository_root != task.repository_root:
        raise GoalStageEvidenceError("Session has recovery warnings or belongs to a different repository")
    if journal.warnings or journal.ambiguous_entries:
        raise GoalStageEvidenceError("Effect Journal has unresolved or invalid side-effect facts")
    if len({event.event_id for event in session.events}) != len(session.events):
        raise GoalStageEvidenceError("Session has duplicate event IDs")

    checkpoint = task.checkpoints[-1]
    indices = [index for index, event in enumerate(session.events) if event.event_id == checkpoint.session_event_cursor]
    if len(indices) != 1:
        raise GoalStageEvidenceError("Checkpoint Session cursor is missing or ambiguous")
    cursor_index = indices[0]
    terminal = session.events[cursor_index]
    if terminal.event_type not in {
        RuntimeEventType.TURN_COMPLETED.value, RuntimeEventType.TURN_LIMIT_REACHED.value,
    } or not terminal.turn_id:
        raise GoalStageEvidenceError("Checkpoint does not reference a completed Runtime turn")
    if terminal.event_type == RuntimeEventType.TURN_LIMIT_REACHED.value and terminal.payload.get("pending_tool_call_ids") != []:
        raise GoalStageEvidenceError("Limited Runtime turn has unfinished Tool Calls")
    previous_count = task.selected_checkpoint_count or 0
    if previous_count:
        prior_cursor = task.checkpoints[previous_count - 1].session_event_cursor
        prior_indices = [index for index, event in enumerate(session.events) if event.event_id == prior_cursor]
        if len(prior_indices) != 1 or prior_indices[0] >= cursor_index:
            raise GoalStageEvidenceError("Previous checkpoint Session cursor is missing or out of order")
        start_index = prior_indices[0] + 1
    else:
        start_index = 0
    turn_events = [
        event for event in session.events[start_index:cursor_index + 1]
        if event.turn_id == terminal.turn_id
    ]
    if not any(event.event_type == RuntimeEventType.TURN_STARTED.value for event in turn_events):
        raise GoalStageEvidenceError("Checkpoint turn has no durable start event")
    request_positions = {
        event.tool_call_id: index for index, event in enumerate(turn_events)
        if event.event_type == RuntimeEventType.TOOL_REQUESTED.value and event.tool_call_id
    }
    requests = {
        event.tool_call_id: event for event in turn_events
        if event.event_type == RuntimeEventType.TOOL_REQUESTED.value and event.tool_call_id
    }
    completed = [
        event for event in turn_events
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value and event.tool_call_id
    ]
    request_ids = [
        event.tool_call_id for event in turn_events
        if event.event_type == RuntimeEventType.TOOL_REQUESTED.value and event.tool_call_id
    ]
    completion_ids = [event.tool_call_id for event in completed]
    if len(requests) != len(request_ids) or len(set(completion_ids)) != len(completion_ids):
        raise GoalStageEvidenceError("Runtime turn has duplicate Tool Call IDs")
    if set(request_ids) != set(completion_ids):
        raise GoalStageEvidenceError("Runtime turn has unfinished Tool Calls")
    if any(
        event.tool_call_id not in requests or request_positions[event.tool_call_id] >= index
        for index, event in enumerate(turn_events)
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value and event.tool_call_id
    ):
        raise GoalStageEvidenceError("A Tool completion has no earlier matching request in this turn")

    successful = [
        event for event in completed
        if _successful_tool(event) and requests[event.tool_call_id].payload.get("tool_name") == event.payload.get("tool_name")
        and event.payload.get("tool_name") != "submit_goal_completion"
    ]
    check = step.check
    if isinstance(check, str):
        status = GoalStageAssessmentStatus.BLOCKED
        detail = f"Stage check has no structured Runtime verifier: {check}"
        supporting = (successful or completed)[-31:]
    elif check.kind is GoalStageCheckKind.TEST_COMMAND:
        matching = [
            event for event in completed
            if requests[event.tool_call_id].payload.get("tool_name") == "bash"
            and event.payload.get("tool_name") == "bash"
            and _requested_command(requests[event.tool_call_id]) == check.command
        ]
        supporting = (matching or completed)[-31:]
        latest = matching[-1] if matching else None
        passed = latest is not None and _successful_tool(latest) and _successful_test_result(latest)
        status = GoalStageAssessmentStatus.VERIFIED if passed else GoalStageAssessmentStatus.BLOCKED
        command_label = check.command if len(check.command or "") <= 1_000 else f"{check.command[:1_000]}…"
        detail = (
            f"Exact test command completed with exit code 0: {command_label}"
            if passed else f"Latest exact test command has no completed, non-truncated exit-0 result: {command_label}"
        )
    else:
        supporting = successful[-31:] if successful else completed[-31:]
        passed, detail = _check_file(task.repository_root, check)
        if not successful:
            passed = False
            detail += (
                "; provider output token budget ended before any successful Tool evidence"
                if terminal.event_type == RuntimeEventType.TURN_LIMIT_REACHED.value
                and terminal.payload.get("limit") == "output_tokens"
                else "; this turn has no successful Tool evidence"
            )
        failed_tools = [
            event for event in completed
            if not _successful_tool(event)
            and requests[event.tool_call_id].payload.get("tool_name") != "submit_goal_completion"
        ]
        if passed and failed_tools:
            supporting = completed[-31:]
            status = GoalStageAssessmentStatus.BLOCKED
            detail += "; a Tool did not complete successfully, so the file outcome cannot prove this stage"
        else:
            status = GoalStageAssessmentStatus.VERIFIED if passed else GoalStageAssessmentStatus.BLOCKED

    cited = tuple(dict.fromkeys([*(event.event_id for event in supporting), terminal.event_id]))
    tool_calls = tuple(dict.fromkeys(event.tool_call_id for event in supporting if event.tool_call_id))
    if status is GoalStageAssessmentStatus.VERIFIED:
        verified = set(task.verified_step_ids) | {step.step_id}
        next_ready = next(
            (item.step_id for item in plan.steps if item.step_id not in verified and set(item.dependencies).issubset(verified)),
            None,
        )
        next_action = f"Runtime may select next verified-ready stage: {next_ready}" if next_ready else "Check final Goal criteria and completion claim"
    else:
        next_action = (
            "Resolve the cited Runtime blocker or revise this stage's structured verifier; "
            "do not rerun an unknown effect"
        )
    return GoalStageAssessment(
        plan_version=plan.version,
        step_id=step.step_id,
        checkpoint_id=checkpoint.checkpoint_id,
        session_event_cursor=terminal.event_id,
        status=status,
        evidence_event_ids=cited,
        tool_call_ids=tool_calls,
        detail=detail,
        next_action=next_action,
    )


def _successful_tool(event: SessionEvent) -> bool:
    payload = event.payload
    return (
        payload.get("interrupted") is not True
        and payload.get("tool_status") == ToolStatus.COMPLETED.value
        and display_tool_status(str(payload.get("result", "")), payload.get("tool_status")) == ToolStatus.COMPLETED.value
    )


def _requested_command(event: SessionEvent) -> str | None:
    arguments = event.payload.get("arguments")
    command = arguments.get("command") if isinstance(arguments, dict) else None
    return command.strip() if isinstance(command, str) else None


def _successful_test_result(event: SessionEvent) -> bool:
    facts = event.payload.get("result_facts")
    return (
        isinstance(facts, dict)
        and type(facts.get("exit_code")) is int
        and facts.get("exit_code") == 0
        and facts.get("timed_out") is False
        and facts.get("truncated") is False
        and event.payload.get("interrupted") is not True
    )


def _check_file(repository: Path, check: GoalStageCheck) -> tuple[bool, str]:
    try:
        candidate = (repository / (check.path or "")).resolve()
        if not candidate.is_relative_to(repository.resolve()):
            return False, f"Check path escapes the repository: {check.path}"
        if not candidate.is_file():
            return False, f"Expected file is missing: {check.path}"
        if check.kind is GoalStageCheckKind.FILE_EXISTS:
            return True, f"File exists: {check.path}"
        if candidate.stat().st_size > 2_000_000:
            return False, f"File is too large to check safely: {check.path}"
        content = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return False, f"Could not read {check.path}: {error}"
    if check.text is None:
        return False, f"File content check has no expected text: {check.path}"
    return (
        (True, f"Confirmed text found in {check.path}")
        if check.text in content else (False, f"Confirmed text is missing from {check.path}")
    )
