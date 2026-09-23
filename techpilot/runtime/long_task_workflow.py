"""Goal lifecycle over the existing Chat Runtime.

Goals add durable lifecycle, outcome, constraints, and checkpoints. Tool
execution and effect recovery stay in the shared Runtime Effect Journal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from techpilot.engine.agent import ToolExecutionContext
from techpilot.engine.events import RuntimeEventType
from techpilot.engine.tool_execution import ToolEffect, ToolExecutionDescription, default_tool_description
from techpilot.engine.tool_results import ToolStatus, display_tool_status, tool_result
from techpilot.engine.tools.base import Tool

from .bootstrap import TaskRuntime
from .effect_journal import EffectJournalStore
from .goal_completion import GOAL_COMPLETION_TOOL_NAME, GoalCompletionProposal, GoalCompletionTool
from .long_tasks import (
    CompletionCriterionKind,
    EffectDisposition,
    LongTaskBudget,
    LongTaskLease,
    LongTaskProjection,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskStore,
    TaskCheckpoint,
    TaskCompletionClaim,
    TaskCompletionCriterion,
    TaskCompletionEvidence,
)


class LongTaskRecoveryRequired(LongTaskStateError):
    """A Goal cannot advance until shared Runtime recovery is reconciled."""


FaultHook = Callable[[str, str], None]


@dataclass(frozen=True, slots=True)
class LongTaskWorkflowResult:
    """One completed controlled turn and its durable long-task checkpoint."""

    response: str
    task: LongTaskProjection
    checkpoint: TaskCheckpoint


class LongTaskWorkflow:
    """Run an explicit durable Goal through a normal ``TaskRuntime``."""

    def __init__(
        self,
        runtime: TaskRuntime,
        *,
        task_id: str,
        goal: str,
        budget: LongTaskBudget | None = None,
    ) -> None:
        if runtime.session_store is None:
            raise ValueError("Long Task workflow requires Runtime Session persistence")
        self.runtime = runtime
        self.task_id = task_id
        self.goal = goal
        # Kept only to replay historical Task records. Goal no longer applies
        # its own per-effect budget; Runtime owns Tool execution control.
        self.budget = budget or LongTaskBudget()
        self.store = LongTaskStore.for_repository(
            runtime.repository,
            session_directory=runtime.session_store.directory,
        )
        self._completion_proposal: GoalCompletionProposal | None = None

    def start_or_resume(self, *, reason: str = "continue explicit long Task") -> LongTaskProjection:
        """Create once, or re-evaluate a prior Task before another controlled turn."""

        try:
            task = self.store.replay(self.task_id)
        except FileNotFoundError:
            return self.store.create(
                self.task_id,
                goal=self.goal,
                repository_root=self.runtime.repository,
                session_id=self.runtime.agent.session_id,
                budget=self.budget,
            )
        if task.repository_root != self.runtime.repository or task.session_id != self.runtime.agent.session_id:
            raise LongTaskStateError("Long Task does not belong to this Runtime Session")
        if task.status in {LongTaskStatus.PAUSED, LongTaskStatus.RECOVERY_REQUIRED} or task.ambiguous_effect_ids:
            task = self.store.resume(self.task_id, reason=reason)
        return task

    def run_turn(self, user_input: str) -> LongTaskWorkflowResult:
        """Compatibility shortcut: run one step and explicitly finish the Task."""

        step = self.run_step(user_input)
        # This helper preserves the frozen, pre-Goal-Chat evaluator contract.
        # The product-facing Chat path uses ``finish`` with its default claim
        # requirement instead.
        task = self.finish(result=step.response, require_completion_claim=False)
        return LongTaskWorkflowResult(response=step.response, task=task, checkpoint=step.checkpoint)

    def run_step(self, user_input: str) -> LongTaskWorkflowResult:
        """Execute one Runtime turn and checkpoint it while keeping the Task running."""

        task = self.start_or_resume()
        self._require_effect_recovery_clear(task)
        if task.status is LongTaskStatus.RECOVERY_REQUIRED:
            raise LongTaskRecoveryRequired(
                "Long Task has an interrupted effect; inspect and reconcile it before continuing"
            )
        if task.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Long Task cannot run: {task.status.value}")
        prior_session_events = len(self.runtime.session_store.replay(self.runtime.agent.session_id).events)
        self._completion_proposal = None
        with self.store.acquire_lease(self.task_id):
            with self.runtime.goal_scope(
                self._goal_context(task),
                tools=(GoalCompletionTool(self._submit_completion_proposal),),
            ):
                response = self.runtime.run_turn(user_input)
            self._require_effect_recovery_clear(task)
            task = self.store.replay(self.task_id)
            if task.status is not LongTaskStatus.RUNNING:
                raise LongTaskStateError(f"Long Task stopped during turn: {task.status.value}")
            session = self.runtime.session_store.replay(self.runtime.agent.session_id)
            if not session.events:
                raise LongTaskStateError("Runtime Session did not record a turn cursor")
            checkpoint = self.store.checkpoint(
                self.task_id,
                message_projection=list(self.runtime.agent.messages),
                recovery_reason="completed controlled Runtime turn",
                session_event_cursor=session.events[-1].event_id,
            )
            task = self.store.replay(self.task_id)
            self._record_completion_claim(self._completion_proposal, session.events[prior_session_events:], checkpoint)
            task = self.store.replay(self.task_id)
        return LongTaskWorkflowResult(response=response, task=task, checkpoint=checkpoint)

    @staticmethod
    def _goal_context(task: LongTaskProjection) -> str:
        """Render durable task facts as non-compressible per-step context."""

        lines = [
            "## Current durable Goal",
            f"- Task ID: {task.task_id}",
            f"- Required outcome: {task.goal}",
            f"- Lifecycle state: {task.status.value}",
            "- Treat the required outcome as active runtime context. Do not assume a conversation summary preserves it.",
        ]
        if task.active_constraints:
            lines.extend(("", "## Confirmed hard constraints"))
            lines.extend(f"- {constraint.text}" for constraint in task.active_constraints)
        if task.active_completion_criteria:
            lines.extend(("", "## Runtime completion criteria"))
            lines.extend(
                f"- {LongTaskWorkflow._criterion_description(criterion)}"
                for criterion in task.active_completion_criteria
            )
        lines.extend((
            "",
            "## Completion proposal protocol",
            "- When the required outcome is ready, call submit_goal_completion exactly once after this Goal step has real completed Tool evidence.",
            "- Supply a concise summary and the completed Tool Call IDs that prove it. Do not emit any special completion markup in your user-facing reply.",
            "- This is only a proposal. It does not complete the Goal; Runtime verifies the referenced raw Session events and the user must still run /goal complete.",
        ))
        return "\n".join(lines)

    def finish(self, *, result: str = "", require_completion_claim: bool = True) -> LongTaskProjection:
        """Explicitly mark a fully checkpointed Task complete after its final step."""

        task = self.store.replay(self.task_id)
        if task.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Long Task cannot finish: {task.status.value}")
        if require_completion_claim:
            claim = task.active_completion_claim
            if claim is None:
                raise LongTaskStateError(
                    "Long Task requires a fresh, Runtime-verified completion claim before finishing"
                )
            if not task.active_completion_criteria:
                raise LongTaskStateError("Long Task requires at least one Runtime completion criterion before finishing")
            verification_results = self._verify_completion_criteria(task, claim)
            verified = self.store.record_completion_verification(
                self.task_id,
                claim_id=claim.claim_id,
                checkpoint_id=claim.checkpoint_id,
                results=verification_results,
            )
            failed = [detail for _criterion_id, passed, detail in verification_results if not passed]
            if failed:
                raise LongTaskStateError(f"Goal completion criteria failed: {'; '.join(failed)}")
            task = verified
            result = result or claim.summary
        return self.store.succeed(self.task_id, result=result)

    def record_constraint(self, text: str) -> LongTaskProjection:
        """Persist a user-confirmed hard rule for all later Goal steps."""

        return self.store.record_constraint(self.task_id, text)

    def record_completion_criterion(
        self,
        kind: CompletionCriterionKind,
        *,
        path: str | None = None,
        expected_text: str | None = None,
        command: str | None = None,
    ) -> LongTaskProjection:
        return self.store.record_completion_criterion(
            self.task_id,
            kind=kind,
            path=path,
            expected_text=expected_text,
            command=command,
        )

    def _record_completion_claim(
        self,
        proposal: GoalCompletionProposal | None,
        session_events: list[object],
        checkpoint: TaskCheckpoint,
    ) -> None:
        """Accept only a strict model proposal backed by this step's raw events."""

        if proposal is None:
            return
        completed_events = [
            event for event in session_events
            if getattr(event, "event_type", None) == RuntimeEventType.TOOL_COMPLETED.value
        ]
        proposal_events = [
            event for event in completed_events
            if isinstance(getattr(event, "payload", None), dict)
            and event.payload.get("tool_name") == GOAL_COMPLETION_TOOL_NAME
        ]
        if len(proposal_events) != 1 or not completed_events or completed_events[-1] is not proposal_events[0]:
            return
        evidence_by_call: dict[str, TaskCompletionEvidence] = {}
        for event in completed_events:
            tool_call_id = getattr(event, "tool_call_id", None)
            event_id = getattr(event, "event_id", None)
            payload = getattr(event, "payload", None)
            if not isinstance(tool_call_id, str) or not isinstance(event_id, str) or not isinstance(payload, dict):
                continue
            status = display_tool_status(str(payload.get("result", "")), payload.get("tool_status"))
            if status != ToolStatus.COMPLETED.value or payload.get("interrupted") is True:
                continue
            tool_name = payload.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name or tool_name == GOAL_COMPLETION_TOOL_NAME:
                continue
            evidence_by_call[tool_call_id] = TaskCompletionEvidence(
                tool_call_id=tool_call_id,
                session_event_id=event_id,
                tool_name=tool_name,
                tool_status=status,
            )
        try:
            evidence = tuple(evidence_by_call[call_id] for call_id in proposal.evidence_tool_call_ids)
        except KeyError:
            return
        self.store.record_completion_claim(
            self.task_id,
            summary=proposal.summary,
            checkpoint_id=checkpoint.checkpoint_id,
            session_event_cursor=checkpoint.session_event_cursor or "",
            evidence=evidence,
        )

    def _submit_completion_proposal(self, proposal: GoalCompletionProposal) -> None:
        if self._completion_proposal is not None:
            raise ValueError("only one completion proposal is allowed per Goal step")
        self._completion_proposal = proposal

    def _verify_completion_criteria(
        self,
        task: LongTaskProjection,
        claim: TaskCompletionClaim,
    ) -> tuple[tuple[str, bool, str], ...]:
        events = {
            event.event_id: event
            for event in self.runtime.session_store.replay(self.runtime.agent.session_id).events
        }
        results: list[tuple[str, bool, str]] = []
        for criterion in task.active_completion_criteria:
            if criterion.kind is CompletionCriterionKind.FILE_EXISTS:
                passed, detail = self._verify_file_exists(criterion)
            elif criterion.kind is CompletionCriterionKind.FILE_CONTAINS:
                passed, detail = self._verify_file_contains(criterion)
            else:
                passed, detail = self._verify_test_command(criterion, claim, events)
            results.append((criterion.criterion_id, passed, detail))
        return tuple(results)

    def _verify_file_exists(self, criterion: TaskCompletionCriterion) -> tuple[bool, str]:
        path = self._criterion_path(criterion)
        if path is None:
            return False, f"{criterion.criterion_id}: invalid repository path"
        return (
            (True, f"{criterion.criterion_id}: file exists: {criterion.path}")
            if path.is_file()
            else (False, f"{criterion.criterion_id}: missing file: {criterion.path}")
        )

    def _verify_file_contains(self, criterion: TaskCompletionCriterion) -> tuple[bool, str]:
        path = self._criterion_path(criterion)
        if path is None or criterion.expected_text is None:
            return False, f"{criterion.criterion_id}: invalid file content criterion"
        try:
            if not path.is_file():
                return False, f"{criterion.criterion_id}: missing file: {criterion.path}"
            if path.stat().st_size > 2_000_000:
                return False, f"{criterion.criterion_id}: file is too large to verify safely: {criterion.path}"
            contents = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            return False, f"{criterion.criterion_id}: could not read {criterion.path}: {error}"
        return (
            (True, f"{criterion.criterion_id}: expected text found in {criterion.path}")
            if criterion.expected_text in contents
            else (False, f"{criterion.criterion_id}: expected text not found in {criterion.path}")
        )

    def _verify_test_command(
        self,
        criterion: TaskCompletionCriterion,
        claim: TaskCompletionClaim,
        events: dict[str, object],
    ) -> tuple[bool, str]:
        if not criterion.command:
            return False, f"{criterion.criterion_id}: missing expected test command"
        for evidence in claim.evidence:
            event = events.get(evidence.session_event_id)
            payload = getattr(event, "payload", None)
            if not isinstance(payload, dict) or payload.get("tool_name") != "bash":
                continue
            if display_tool_status(str(payload.get("result", "")), payload.get("tool_status")) != ToolStatus.COMPLETED.value:
                continue
            facts = payload.get("result_facts")
            if not isinstance(facts, dict) or facts.get("exit_code") != 0:
                continue
            if facts.get("timed_out") is not False or facts.get("truncated") is not False:
                continue
            request = next(
                (
                    candidate for candidate in events.values()
                    if getattr(candidate, "tool_call_id", None) == evidence.tool_call_id
                    and getattr(candidate, "event_type", None) == RuntimeEventType.TOOL_REQUESTED.value
                ),
                None,
            )
            request_payload = getattr(request, "payload", None)
            arguments = request_payload.get("arguments") if isinstance(request_payload, dict) else None
            command = arguments.get("command") if isinstance(arguments, dict) else None
            if isinstance(command, str) and " ".join(command.split()) == criterion.command:
                return True, f"{criterion.criterion_id}: cited test command exited 0"
        return False, f"{criterion.criterion_id}: no cited successful execution of {criterion.command!r}"

    def _criterion_path(self, criterion: TaskCompletionCriterion) -> Path | None:
        if not criterion.path:
            return None
        repository = self.runtime.repository.resolve()
        candidate = (repository / criterion.path).resolve()
        return candidate if candidate == repository or repository in candidate.parents else None

    @staticmethod
    def _criterion_description(criterion: TaskCompletionCriterion) -> str:
        if criterion.kind is CompletionCriterionKind.FILE_EXISTS:
            return f"{criterion.criterion_id}: file exists at {criterion.path}"
        if criterion.kind is CompletionCriterionKind.FILE_CONTAINS:
            return f"{criterion.criterion_id}: file {criterion.path} contains the confirmed text"
        return f"{criterion.criterion_id}: exact test command exits 0: {criterion.command}"

    def _require_effect_recovery_clear(self, task: LongTaskProjection) -> None:
        """Project shared recovery facts into the Goal lifecycle, without copying them."""

        effects = EffectJournalStore.for_session_directory(self.runtime.session_store.directory)
        ambiguous = effects.replay(self.runtime.agent.session_id).ambiguous_entries
        if not ambiguous:
            return
        self.store.require_recovery(
            task.task_id,
            reason="Shared Effect Journal has an unfinished Tool effect",
        )
        raise LongTaskRecoveryRequired(
            "Shared Effect Journal has an unfinished Tool effect; inspect and reconcile it before continuing"
        )


class LegacyLongTaskWorkflow(LongTaskWorkflow):
    """Compatibility adapter for frozen RS-6 effect-ledger replay cases.

    It is intentionally not exported as the Chat/Goal workflow. New product
    execution uses :class:`LongTaskWorkflow` and the shared Effect Journal.
    """

    def __init__(self, *args: object, fault_hook: FaultHook | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.fault_hook = fault_hook

    def run_step(self, user_input: str) -> LongTaskWorkflowResult:
        task = self.start_or_resume()
        if task.status is LongTaskStatus.RECOVERY_REQUIRED:
            raise LongTaskRecoveryRequired(
                "Long Task has an interrupted legacy effect; inspect and reconcile it before continuing"
            )
        if task.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Long Task cannot run: {task.status.value}")
        original_executor = self.runtime.agent.tool_executor
        with self.store.acquire_lease(self.task_id) as lease:
            with self.runtime.goal_scope(self._goal_context(task)):
                self.runtime.agent.tool_executor = _LegacyLongTaskToolExecutor(
                    delegate=original_executor,
                    store=self.store,
                    task_id=self.task_id,
                    lease=lease,
                    fault_hook=self.fault_hook,
                )
                try:
                    response = self.runtime.run_turn(user_input)
                finally:
                    self.runtime.agent.tool_executor = original_executor
            task = self.store.replay(self.task_id)
            if task.status is not LongTaskStatus.RUNNING:
                raise LongTaskStateError(f"Long Task stopped during turn: {task.status.value}")
            session = self.runtime.session_store.replay(self.runtime.agent.session_id)
            if not session.events:
                raise LongTaskStateError("Runtime Session did not record a turn cursor")
            checkpoint = self.store.checkpoint(
                self.task_id,
                message_projection=list(self.runtime.agent.messages),
                recovery_reason="completed legacy controlled Runtime turn",
                session_event_cursor=session.events[-1].event_id,
            )
            task = self.store.replay(self.task_id)
        return LongTaskWorkflowResult(response=response, task=task, checkpoint=checkpoint)


class _LegacyLongTaskToolExecutor:
    """Frozen evaluator-only copy of the retired Task effect ledger boundary."""

    def __init__(
        self,
        *,
        delegate: object | None,
        store: LongTaskStore,
        task_id: str,
        lease: LongTaskLease,
        fault_hook: FaultHook | None,
    ) -> None:
        self.delegate = delegate
        self.store = store
        self.task_id = task_id
        self.lease = lease
        self.fault_hook = fault_hook
        self._turn_stop_message: str | None = None

    def begin_turn(self) -> None:
        self._turn_stop_message = None
        begin = getattr(self.delegate, "begin_turn", None)
        if callable(begin):
            begin()

    def consume_turn_stop_message(self) -> str | None:
        consume = getattr(self.delegate, "consume_turn_stop_message", None)
        message = consume() if callable(consume) else None
        return message or self._turn_stop_message

    def describe_call(self, tool: Tool, arguments: dict[str, object]) -> ToolExecutionDescription | None:
        describe = getattr(self.delegate, "describe_call", None)
        if callable(describe):
            return describe(tool, dict(arguments))
        return None

    def execute(self, tool: Tool, arguments: dict[str, object]) -> str:
        return self.execute_call(tool, arguments, tool_call_id="uncorrelated-tool-call")

    def execute_call(
        self,
        tool: Tool,
        arguments: dict[str, object],
        *,
        tool_call_id: str,
        execution_context: ToolExecutionContext | None = None,
    ) -> str:
        action_id = f"tool-{tool_call_id}"
        task = self.store.replay(self.task_id)
        action = task.actions.get(action_id)
        description = self._describe(tool, arguments)
        is_effect = description.effect is not ToolEffect.READ
        if action is None:
            self.store.plan_action(
                self.task_id,
                action_id=action_id,
                kind=f"tool:{tool.name}",
                effect_id=f"effect-{tool_call_id}" if is_effect else None,
            )
        if is_effect:
            disposition = self.store.effect_disposition(self.task_id, action_id)
            if disposition is EffectDisposition.SKIP_SUCCEEDED:
                return "Long Task recovery: this completed tool effect was skipped."
            if disposition is EffectDisposition.RECONCILE_REQUIRED:
                self._turn_stop_message = (
                    "Long Task recovery required: a previous tool effect may have run; "
                    "no further tools were executed."
                )
                return self._turn_stop_message
            if disposition is not EffectDisposition.EXECUTE:
                self._turn_stop_message = f"Long Task cannot execute this tool: {disposition.value}."
                return self._turn_stop_message
            self._fault("before_effect", action_id)
            self.store.start_effect(self.task_id, action_id, lease=self.lease)
            self._fault("after_effect_started", action_id)
            result = self._delegate_execute(tool, arguments, tool_call_id, execution_context)
            if _known_not_executed(result):
                self.store.fail_effect(self.task_id, action_id, reason=result)
                self._turn_stop_message = "Long Task tool effect was not approved or was blocked; turn stopped."
                return result
            if tool_result(result).status is ToolStatus.EFFECT_UNKNOWN:
                self._turn_stop_message = (
                    "Long Task recovery required: the tool result leaves the effect outcome unknown; "
                    "no further tools were executed."
                )
                return result
            self.store.complete_effect(self.task_id, action_id, result=result)
            self._fault("after_effect_completed", action_id)
            return result
        result = self._delegate_execute(tool, arguments, tool_call_id, execution_context)
        self.store.complete_action(self.task_id, action_id, result=result)
        return result

    def _delegate_execute(
        self,
        tool: Tool,
        arguments: dict[str, object],
        tool_call_id: str,
        execution_context: ToolExecutionContext | None,
    ) -> str:
        execute_call = getattr(self.delegate, "execute_call", None)
        if callable(execute_call):
            return execute_call(tool, dict(arguments), tool_call_id=tool_call_id, execution_context=execution_context)
        execute = getattr(self.delegate, "execute", None)
        if callable(execute):
            return execute(tool, dict(arguments))
        return tool.execute(**arguments)

    def _describe(self, tool: Tool, arguments: dict[str, object]) -> ToolExecutionDescription:
        described = self.describe_call(tool, arguments)
        if isinstance(described, ToolExecutionDescription):
            return described
        return default_tool_description(tool.name, dict(arguments))

    def _fault(self, phase: str, action_id: str) -> None:
        if self.fault_hook is not None:
            self.fault_hook(phase, action_id)


def _known_not_executed(result: str) -> bool:
    return result.startswith(("Denied ", "Permission denied ", "Policy denied "))
