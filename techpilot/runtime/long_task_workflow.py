"""Explicit long-task execution over the existing Chat Runtime.

This opt-in adapter does not alter normal Chat, permission decisions, Tool
effects, C5 scheduling, or Session facts.  It only places durable Task facts
around a Tool effect so recovery can execute, skip, or request reconciliation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from techpilot.engine.agent import ToolExecutionContext
from techpilot.engine.tool_execution import ToolEffect, ToolExecutionDescription, default_tool_description
from techpilot.engine.tools.base import Tool

from .bootstrap import TaskRuntime
from .long_tasks import (
    EffectDisposition,
    LongTaskBudget,
    LongTaskLease,
    LongTaskProjection,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskStore,
    TaskCheckpoint,
)

FaultHook = Callable[[str, str], None]


class LongTaskRecoveryRequired(LongTaskStateError):
    """A resumed Task has an effect whose outcome cannot be inferred safely."""


@dataclass(frozen=True, slots=True)
class LongTaskWorkflowResult:
    """One completed controlled turn and its durable long-task checkpoint."""

    response: str
    task: LongTaskProjection
    checkpoint: TaskCheckpoint


class LongTaskWorkflow:
    """Run explicitly managed work through a normal ``TaskRuntime``.

    ``fault_hook`` exists only for deterministic interruption tests.  It may
    raise ``KeyboardInterrupt`` at a named effect boundary to model process
    loss; production callers leave it unset.
    """

    def __init__(
        self,
        runtime: TaskRuntime,
        *,
        task_id: str,
        goal: str,
        budget: LongTaskBudget | None = None,
        fault_hook: FaultHook | None = None,
    ) -> None:
        if runtime.session_store is None:
            raise ValueError("Long Task workflow requires Runtime Session persistence")
        self.runtime = runtime
        self.task_id = task_id
        self.goal = goal
        self.budget = budget or LongTaskBudget()
        self.fault_hook = fault_hook
        self.store = LongTaskStore.for_repository(
            runtime.repository,
            session_directory=runtime.session_store.directory,
        )

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
        task = self.finish(result=step.response)
        return LongTaskWorkflowResult(response=step.response, task=task, checkpoint=step.checkpoint)

    def run_step(self, user_input: str) -> LongTaskWorkflowResult:
        """Execute one Runtime turn and checkpoint it while keeping the Task running."""

        task = self.start_or_resume()
        if task.status is LongTaskStatus.RECOVERY_REQUIRED:
            raise LongTaskRecoveryRequired(
                "Long Task has an interrupted effect; inspect and reconcile it before continuing"
            )
        if task.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Long Task cannot run: {task.status.value}")
        original_executor = self.runtime.agent.tool_executor
        with self.store.acquire_lease(self.task_id) as lease:
            self.runtime.agent.tool_executor = _LongTaskToolExecutor(
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
                recovery_reason="completed controlled Runtime turn",
                session_event_cursor=session.events[-1].event_id,
            )
            task = self.store.replay(self.task_id)
        return LongTaskWorkflowResult(response=response, task=task, checkpoint=checkpoint)

    def finish(self, *, result: str = "") -> LongTaskProjection:
        """Explicitly mark a fully checkpointed Task complete after its final step."""

        task = self.store.replay(self.task_id)
        if task.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Long Task cannot finish: {task.status.value}")
        return self.store.succeed(self.task_id, result=result)


class _LongTaskToolExecutor:
    """Decorate the current executor; never replace its permissions or policy."""

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
            task = self.store.plan_action(
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
            return execute_call(
                tool,
                dict(arguments),
                tool_call_id=tool_call_id,
                execution_context=execution_context,
            )
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
    """Recognize only results that prove C3 or policy blocked the call."""

    return result.startswith(("Denied ", "Permission denied ", "Policy denied "))
