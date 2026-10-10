"""Goal lifecycle over the existing Chat Runtime.

Goals add durable lifecycle, outcome, constraints, and checkpoints. Tool
execution and effect recovery stay in the shared Runtime Effect Journal.
"""

from __future__ import annotations

import hashlib
import json
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
from .goal_change import GOAL_CHANGE_TOOL_NAME, GoalChangeDecision, GoalChangeTool
from .goal_completion import GOAL_COMPLETION_TOOL_NAME, GoalCompletionProposal, GoalCompletionTool
from .goal_evaluation import GoalStageEvidenceError, evaluate_goal_stage
from .goal_plans import GoalStageAssessment
from .goal_replanning import GoalReplanDecision, GoalReplanTool
from .goal_setup import GoalSetupProposal, GoalSetupTool
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
    """A Goal cannot advance until shared Runtime recovery has a verifiable outcome."""


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
        self._setup_proposal: GoalSetupProposal | None = None
        self._change_decision: GoalChangeDecision | None = None
        self._replan_decision: GoalReplanDecision | None = None
        self._replan_decision_invalid = False
        self._change_decision_invalid = False

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
        if task.status in {
            LongTaskStatus.SUCCEEDED,
            LongTaskStatus.FAILED,
            LongTaskStatus.CANCELLED,
            LongTaskStatus.LIMIT_REACHED,
        }:
            return task
        try:
            self._require_effect_recovery_clear(task)
        except LongTaskRecoveryRequired:
            return self.store.replay(self.task_id)
        task = self.store.replay(self.task_id)
        if task.status in {LongTaskStatus.PAUSED, LongTaskStatus.RECOVERY_REQUIRED} or task.ambiguous_effect_ids:
            task = self.store.resume(self.task_id, reason=reason)
        return task

    def create_for_setup(self) -> LongTaskProjection:
        """Create a Chat Goal whose acceptance contract is set before execution."""

        try:
            task = self.store.replay(self.task_id)
        except FileNotFoundError:
            return self.store.create(
                self.task_id,
                goal=self.goal,
                repository_root=self.runtime.repository,
                session_id=self.runtime.agent.session_id,
                budget=self.budget,
                initial_status=LongTaskStatus.SETUP,
            )
        if task.repository_root != self.runtime.repository or task.session_id != self.runtime.agent.session_id:
            raise LongTaskStateError("Long Task does not belong to this Runtime Session")
        return task

    def start(self, *, reason: str = "started from Chat Goal command") -> LongTaskProjection:
        return self.store.start(self.task_id, reason=reason)

    @property
    def setup_proposal(self) -> GoalSetupProposal | None:
        return self._setup_proposal

    def prepare_setup(self, user_input: str) -> str:
        """Investigate an acceptance contract without exposing write Tools."""

        task = self.create_for_setup()
        if task.status is not LongTaskStatus.SETUP:
            raise LongTaskStateError(f"Goal setup cannot continue: {task.status.value}")
        self._setup_proposal = None
        context = "\n".join((
            "## Read-only Goal setup",
            f"Required outcome: {task.goal}",
            "Investigate only with the available read Tools. No writes or commands are available.",
            "Propose only checks that actually verify the requested outcome. Do not invent hard constraints.",
            "If evidence is insufficient, ask a specific clarification question instead of calling propose_goal_setup.",
            "When the contract is clear, call propose_goal_setup exactly once with 1–12 structured completion checks.",
            "A short Goal can use plan=null. A multi-stage Goal may include a bounded structured plan.",
            "The proposal must be confirmed by the user before any Goal execution starts.",
        ))
        read_tools = frozenset({"read_file", "glob", "grep", "now", GoalSetupTool.name})
        with self.runtime.goal_scope(
            context,
            tools=(GoalSetupTool(self._submit_setup_proposal),),
            allowed_tool_names=read_tools,
        ):
            return self.runtime.run_turn(user_input)

    def _submit_setup_proposal(self, proposal: GoalSetupProposal) -> None:
        if self._setup_proposal is not None:
            raise ValueError("only one Goal setup proposal is allowed per turn")
        self._setup_proposal = proposal

    def classify_goal_input(self, user_input: str, *, retry: bool = False) -> tuple[str, GoalChangeDecision | None]:
        """Read-only preflight for a human turn before exposing effectful Tools."""

        task = self.store.replay(self.task_id)
        if task.status is not LongTaskStatus.RUNNING or task.pending_goal_change is not None:
            raise LongTaskStateError("Goal input cannot be classified at this boundary")
        if retry:
            if task.pending_goal_input is None or task.pending_goal_input["text"] != user_input:
                raise LongTaskStateError("No matching Goal input is available for retry")
        else:
            task = self.store.begin_goal_input(self.task_id, user_input)
        self._change_decision = None
        self._change_decision_invalid = False
        context = "\n".join((
            self._goal_context(task),
            "## Human-turn preflight (read-only)",
            "No work or write tools are available. Classify this exact user message before Goal execution.",
            "Call propose_goal_change exactly once: continue for ordinary work; clarify if ambiguous;",
            "otherwise propose exactly one change to a hard constraint, completion criterion, or stage plan.",
            "Never infer a change merely from old conversation. Never treat a proposal as confirmed.",
            "For removals use an existing ID from the Goal facts. If several changes are requested, clarify which to apply first.",
            "For a blocked stage, a plan replacement must keep verified stages unchanged and keep the blocked stage ID.",
            "Active constraint IDs: " + ", ".join(item.constraint_id for item in task.active_constraints),
            "Active criterion IDs: " + ", ".join(item.criterion_id for item in task.active_completion_criteria),
        ))
        read_tools = frozenset({"read_file", "glob", "grep", "now", GOAL_CHANGE_TOOL_NAME})
        with self.runtime.goal_scope(
            context,
            tools=(GoalChangeTool(self._submit_change_decision),),
            allowed_tool_names=read_tools,
        ):
            response = self.runtime.run_turn(user_input)
        return response, None if self._change_decision_invalid else self._change_decision

    def _submit_change_decision(self, decision: GoalChangeDecision) -> None:
        if self._change_decision is not None:
            self._change_decision_invalid = True
            raise ValueError("only one Goal input classification is allowed per turn")
        self._change_decision = decision

    def accept_setup(self, proposal: GoalSetupProposal) -> LongTaskProjection:
        """Persist a user-confirmed contract, then leave SETUP."""

        task = self.store.replay(self.task_id)
        if task.status is not LongTaskStatus.SETUP:
            raise LongTaskStateError(f"Goal setup cannot be accepted: {task.status.value}")
        # Revalidate an ephemeral proposal before the first durable write.
        proposal = GoalSetupTool._validate(
            proposal.summary, list(proposal.constraints), list(proposal.criteria), proposal.plan,
        )
        for constraint in proposal.constraints:
            task = self.store.record_constraint(
                self.task_id, constraint, source="user_confirmed_goal_setup",
            )
        for criterion in proposal.criteria:
            criterion_id = "setup-" + hashlib.sha256(
                json.dumps(criterion, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
            if criterion_id in task.completion_criteria:
                continue
            task = self.store.record_completion_criterion(
                self.task_id,
                kind=CompletionCriterionKind(criterion["kind"]),
                path=criterion.get("path"),
                expected_text=criterion.get("expected_text"),
                command=criterion.get("command"),
                criterion_id=criterion_id,
            )
        if proposal.plan is not None and task.plan is None:
            task = self.store.record_goal_plan(self.task_id, proposal.plan)
        if task.plan is not None and task.selected_step_id is None:
            ready = task.ready_step_ids
            if not ready:
                raise LongTaskStateError("Goal plan has no ready first stage")
            task = self.store.select_goal_step(self.task_id, ready[0])
        return self.start(reason="User confirmed Goal setup in Chat")

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
        if task.pending_goal_change is not None or task.pending_goal_input is not None:
            raise LongTaskStateError("Goal input awaits classification or confirmation; do not run work Tools")
        self._require_effect_recovery_clear(task)
        if task.status is LongTaskStatus.RECOVERY_REQUIRED:
            raise LongTaskRecoveryRequired(
                "Long Task has an interrupted effect with no verifiable outcome; do not rerun it before continuing"
            )
        if task.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Long Task cannot run: {task.status.value}")
        self._completion_proposal = None
        with self.store.acquire_lease(self.task_id):
            task = self.store.replay(self.task_id)
            if task.pending_goal_change is not None or task.pending_goal_input is not None:
                raise LongTaskStateError("Goal input awaits classification or confirmation; do not run work Tools")
            self._require_effect_recovery_clear(task)
            task = self.store.replay(self.task_id)
            if task.status is not LongTaskStatus.RUNNING:
                raise LongTaskStateError(f"Long Task cannot run: {task.status.value}")
            self._require_selected_step_ready(task)
            prior_session_events = len(self.runtime.session_store.replay(self.runtime.agent.session_id).events)
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
        if task.plan is not None:
            lines.extend(("", f"## Confirmed stage plan v{task.plan.version}"))
            all_verified = len(task.verified_step_ids) == len(task.plan.steps)
            step = task.plan.step(task.selected_step_id) if task.selected_step_id and not all_verified else None
            if all_verified:
                lines.append("- All stages are verified; gather only evidence needed for final Goal completion.")
            if step is not None:
                lines.extend((
                    f"- Current stage: {step.step_id} — {step.goal}",
                    f"- Expected output: {step.expected_output}",
                    f"- Work scope (descriptive, not permission): {', '.join(step.work_scope)}",
                    f"- Expected evidence: {step.evidence}",
                    f"- Acceptance check: {step.check}",
                    f"- Next action: {step.next_action}",
                    "- Work only on this stage. A checkpoint or your own completion claim does not verify the stage.",
                ))
        lines.extend((
            "",
            "## Completion proposal protocol",
            "- When the required outcome is ready, call submit_goal_completion exactly once after this Goal step has real completed Tool evidence.",
            "- Supply a concise summary and the completed Tool Call IDs that prove it. Do not emit any special completion markup in your user-facing reply.",
            "- This is only a proposal. Runtime verifies the referenced raw Session events and every acceptance check before completing the Goal.",
        ))
        return "\n".join(lines)

    @staticmethod
    def _require_selected_step_ready(task: LongTaskProjection) -> None:
        if task.plan is None:
            return
        if len(task.verified_step_ids) == len(task.plan.steps):
            # The final completion proposal may need one more evidence turn
            # after all stages have been assessed. This is not a stage retry.
            return
        if task.selected_step_id is None:
            raise LongTaskStateError("Select a ready Goal plan step before running another turn")
        step = task.plan.step(task.selected_step_id)
        if step is None or not set(step.dependencies).issubset(set(task.verified_step_ids)):
            raise LongTaskStateError("The selected Goal plan step is missing or blocked by dependencies")
        if task.selected_step_needs_evaluation:
            raise LongTaskStateError("The selected Goal plan step has a checkpoint and needs evaluation before another turn")
        if task.selected_step_assessment is not None:
            raise LongTaskStateError("The selected Goal plan step is assessed and requires replanning before another turn")

    def evaluate_selected_step(self) -> tuple[LongTaskProjection, GoalStageAssessment]:
        """Read Session/effect facts and files, then append one stage judgment."""

        with self.store.acquire_lease(self.task_id) as lease:
            task = self.store.replay(self.task_id)
            self._require_effect_recovery_clear(task)
            if task.status not in {LongTaskStatus.RUNNING, LongTaskStatus.PAUSED}:
                raise LongTaskStateError(f"Goal stage cannot be evaluated while Task is {task.status.value}")
            try:
                assessment = evaluate_goal_stage(
                    task,
                    session_store=self.runtime.session_store,
                    effect_store=EffectJournalStore.for_session_directory(self.runtime.session_store.directory),
                )
            except GoalStageEvidenceError as error:
                self.store.require_recovery(
                    self.task_id,
                    reason=f"Goal stage evidence needs recovery: {error}",
                    code=(
                        "shared_effect_journal"
                        if str(error).startswith("Effect Journal")
                        else "session_evidence_invalid"
                    ),
                )
                raise LongTaskRecoveryRequired(str(error)) from error
            updated = self.store.record_stage_assessment(self.task_id, assessment, lease=lease)
            if updated.recovery_blockers:
                raise LongTaskRecoveryRequired("Stage assessment evidence could not be replayed safely")
            return updated, assessment

    def replan_if_triggered(self) -> tuple[LongTaskProjection, str | None]:
        """Ask for one read-only plan decision at a durable planning boundary."""

        with self.store.acquire_lease(self.task_id) as lease:
            task = self.store.replay(self.task_id)
            if task.status is not LongTaskStatus.RUNNING or task.pending_goal_input or task.pending_goal_change:
                return task, None
            self._require_effect_recovery_clear(task)
            task = self.store.replay(self.task_id)
            if task.status is not LongTaskStatus.RUNNING or task.recovery_blockers or task.selected_step_needs_evaluation:
                return task, None
            if task.plan is not None and len(task.verified_step_ids) == len(task.plan.steps):
                return task, None
            if task.plan is None and task.checkpoints:
                task = self.store.request_replan_for_failed_turn(
                    self.task_id, checkpoint_id=task.checkpoints[-1].checkpoint_id, lease=lease,
                )
            trigger = task.next_replan_trigger
            if trigger is None:
                return task, None
            session = self.runtime.session_store.replay(self.runtime.agent.session_id)
            if session.warnings or session.repository_root != task.repository_root:
                raise LongTaskRecoveryRequired("Goal replanning requires trustworthy Session evidence")
            self._replan_decision = None
            self._replan_decision_invalid = False
            verified = ", ".join(task.verified_step_ids) or "none"
            prior = "\n".join(
                f"- {step.step_id}: {step.goal}; depends on {', '.join(step.dependencies) or 'none'}; "
                f"check {step.check}"
                for step in task.plan.steps
            ) if task.plan is not None else "none"
            context = "\n".join((
                self._goal_context(task),
                "## Read-only Goal replanning",
                f"Trigger: {trigger.event_type} ({trigger.event_id})",
                f"Verified stages (must remain unchanged): {verified}",
                f"Current plan: {prior}",
                "Inspect repository facts if needed. Do not execute commands or change files.",
                "Call propose_goal_replan exactly once. Use replace only when new evidence justifies a complete 2–7 stage plan.",
                "Keep verified stages unchanged. Keep the active stage unchanged unless its check is blocked;",
                "a blocked stage may be revised but must retain its ID and verified dependencies.",
                "Use keep if the current plan still fits; use unavailable with the specific missing fact if no sound plan can be formed.",
                "A proposal cannot change final acceptance criteria, hard constraints, or Tool permissions.",
            ))
            read_tools = frozenset({"read_file", "glob", "grep", "now", GoalReplanTool.name})
            with self.runtime.goal_scope(
                context,
                tools=(GoalReplanTool(self._submit_replan_decision),),
                allowed_tool_names=read_tools,
            ):
                response = self.runtime.run_turn(
                    "Review the latest durable Goal planning boundary and propose one planning outcome."
                )
            decision = None if self._replan_decision_invalid else self._replan_decision
            if decision is None:
                decision = GoalReplanDecision(
                    "unavailable", "No valid planning proposal was returned; "
                    + (response.strip()[:350] or "the model did not identify a concrete next plan"), None,
                )
            try:
                updated = self.store.record_model_replan(
                    self.task_id, trigger_event_id=trigger.event_id,
                    action=decision.action, summary=decision.summary,
                    plan=decision.plan, lease=lease,
                )
            except (TypeError, ValueError) as error:
                updated = self.store.record_model_replan(
                    self.task_id, trigger_event_id=trigger.event_id,
                    action="unavailable", summary=f"Proposed revision failed Host validation: {error}"[:500],
                    plan=None, lease=lease,
                )
            outcome = updated.events[-1]
            return updated, f"{outcome.event_type}: {outcome.payload['summary']}"

    def _submit_replan_decision(self, decision: GoalReplanDecision) -> None:
        if self._replan_decision is not None:
            self._replan_decision_invalid = True
            raise ValueError("only one Goal planning outcome is allowed per turn")
        self._replan_decision = decision

    def finish(self, *, result: str = "", require_completion_claim: bool = True) -> LongTaskProjection:
        """Explicitly mark a fully checkpointed Task complete after its final step."""

        task = self.store.replay(self.task_id)
        if task.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Long Task cannot finish: {task.status.value}")
        if task.pending_goal_change is not None or task.pending_goal_input is not None:
            raise LongTaskStateError("Long Task cannot finish while a Goal input or change is pending")
        if task.plan is not None:
            unverified = [step.step_id for step in task.plan.steps if step.step_id not in task.verified_step_ids]
            if unverified:
                raise LongTaskStateError(f"Goal cannot finish with unverified stages: {', '.join(unverified)}")
        if require_completion_claim:
            claim = task.active_completion_claim
            if claim is None:
                raise LongTaskStateError(
                    "Long Task requires a fresh, Runtime-verified completion claim before finishing"
                )
            if not task.active_completion_criteria:
                raise LongTaskStateError("Long Task requires at least one Runtime completion criterion before finishing")
            prior = task.last_completion_verification
            verification_results = self._verify_completion_criteria(task, claim)
            if (
                prior is not None and prior.claim_id == claim.claim_id and prior.passed
                and all(passed for _criterion_id, passed, _detail in verification_results)
            ):
                # A crash may occur after durable verification but before
                # task_succeeded. Re-check current facts but do not invent a
                # second verification event when they still pass.
                pass
            else:
                task = self.store.record_completion_verification(
                    self.task_id,
                    claim_id=claim.claim_id,
                    checkpoint_id=claim.checkpoint_id,
                    results=verification_results,
                )
            failed = [detail for _criterion_id, passed, detail in verification_results if not passed]
            if failed:
                raise LongTaskStateError(f"Goal completion criteria failed: {'; '.join(failed)}")
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
        session_id = self.runtime.agent.session_id
        try:
            for entry in effects.replay(session_id).ambiguous_entries:
                effects.verify_non_execution(session_id, entry.tool_call_id)
            journal = effects.replay(session_id)
        except OSError as error:
            reason = f"Shared Effect Journal could not be verified: {error}"
            self.store.require_recovery(task.task_id, reason=reason)
            raise LongTaskRecoveryRequired(reason) from error
        ambiguous = journal.ambiguous_entries
        if ambiguous:
            self.store.require_recovery(
                task.task_id,
                reason="Shared Effect Journal has an unfinished Tool effect",
            )
            raise LongTaskRecoveryRequired(
                "Shared Effect Journal has an unfinished Tool effect with no verifiable outcome; do not rerun it"
            )
        repaired_session = False
        try:
            for entry in journal.entries.values():
                if entry.verification is not None:
                    repaired_session = (
                        effects.complete_verified_session_tool_result(session_id, entry.tool_call_id) is not None
                        or repaired_session
                    )
        except OSError as error:
            reason = f"Runtime Session verified Tool result could not be saved: {error}"
            self.store.require_recovery(task.task_id, reason=reason, code="session_evidence_invalid")
            raise LongTaskRecoveryRequired(reason) from error
        try:
            session = self.runtime.session_store.replay(session_id)
        except (OSError, ValueError) as error:
            reason = f"Runtime Session evidence could not be read: {error}"
            self.store.require_recovery(task.task_id, reason=reason, code="session_evidence_invalid")
            raise LongTaskRecoveryRequired(reason) from error
        if session.warnings:
            reason = f"Runtime Session evidence has recovery warnings: {session.warnings[0]}"
            self.store.require_recovery(task.task_id, reason=reason, code="session_evidence_invalid")
            raise LongTaskRecoveryRequired(reason)
        if repaired_session or (
            task.status is LongTaskStatus.RECOVERY_REQUIRED
            and any(entry.verification is not None for entry in journal.entries.values())
        ):
            try:
                self.runtime.resume_session(session_id)
            except (OSError, ValueError) as error:
                reason = f"Runtime Session model history could not be refreshed: {error}"
                self.store.require_recovery(task.task_id, reason=reason, code="session_evidence_invalid")
                raise LongTaskRecoveryRequired(reason) from error


class LegacyLongTaskWorkflow(LongTaskWorkflow):
    """Compatibility adapter for frozen RS-6 effect-ledger replay cases.

    It is intentionally not exported as the Chat/Goal workflow. New product
    execution uses :class:`LongTaskWorkflow` and the shared Effect Journal.
    """

    def __init__(self, *args: object, fault_hook: FaultHook | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.fault_hook = fault_hook

    def finish(self, *, result: str = "", require_completion_claim: bool = False) -> LongTaskProjection:
        """Preserve the frozen replay contract without Goal completion claims.

        The evaluator-only workflow represents the retired Task-local effect
        ledger. Its historical cards complete after their final checkpoint;
        they must not inherit the product Goal's newer user-acceptance gate.
        """

        return super().finish(result=result, require_completion_claim=require_completion_claim)

    def run_step(self, user_input: str) -> LongTaskWorkflowResult:
        task = self.start_or_resume()
        if task.status is LongTaskStatus.RECOVERY_REQUIRED:
            raise LongTaskRecoveryRequired(
                "Long Task has an interrupted legacy effect with no verifiable outcome; do not rerun it"
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
