from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from techpilot.engine.events import CallbackEventSink
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.tools import WriteFileTool
from techpilot.runtime import (
    LongTaskRecoveryRequired,
    LongTaskStatus,
    LongTaskWorkflow,
    RuntimeBootstrap,
    RuntimeBootstrapInput,
)


class FixedProvider:
    """A deterministic stand-in for one fixed provider/model workflow."""

    model = "fixed-long-task-provider"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict] = []

    def chat(self, messages, tools=None, on_token=None):
        self.requests.append({"messages": messages, "tools": tools})
        response = next(self._responses)
        if on_token and response.content:
            on_token(response.content)
        return response


class AllowPrompt:
    def decide(self, request):
        del request
        return PermissionDecision.allow("test approval")


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
            tool,
            arguments,
            tool_call_id=tool_call_id,
            execution_context=execution_context,
        )


@dataclass
class InterruptAfterExecuteExecutor:
    """Test-only process-loss injector for the ordinary Session control path."""

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
        self.delegate.execute_call(
            tool,
            arguments,
            tool_call_id=tool_call_id,
            execution_context=execution_context,
        )
        raise KeyboardInterrupt("test interruption after ordinary tool execution")


def _runtime(
    root: Path,
    provider: FixedProvider,
    *,
    session_id: str | None = None,
):
    runtime = RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=root,
        event_sink=CallbackEventSink(lambda _event: None),
        tools=[WriteFileTool()],
        model="fixed-long-task-provider",
        permission_prompt=AllowPrompt(),
        session_directory=root / "sessions",
        resume_session_id=session_id,
    ))
    runtime.agent.tool_executor = CountingExecutor(runtime.agent.tool_executor)
    return runtime


def _write_call(call_id: str = "write-1") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(
        id=call_id,
        name="write_file",
        arguments={"file_path": "result.txt", "content": "durable result\n"},
    )])


def test_long_task_workflow_runs_real_runtime_and_binds_checkpoint_to_session(tmp_path):
    runtime = _runtime(tmp_path, FixedProvider([_write_call(), LLMResponse(content="done")]))

    result = LongTaskWorkflow(runtime, task_id="task-1", goal="write result").run_turn("write the result")

    assert result.response == "done"
    assert result.task.status is LongTaskStatus.SUCCEEDED
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "durable result\n"
    assert result.checkpoint.session_event_cursor
    assert runtime.session_store.replay(runtime.agent.session_id).events[-1].event_id == result.checkpoint.session_event_cursor
    assert runtime.agent.tool_executor.calls == 1


def test_recovery_skips_a_completed_effect_after_interruption(tmp_path):
    first = _runtime(tmp_path, FixedProvider([_write_call()]))
    first_counter = first.agent.tool_executor

    def interrupt_after_completion(phase: str, action_id: str) -> None:
        assert action_id == "tool-write-1"
        if phase == "after_effect_completed":
            raise KeyboardInterrupt("test interruption after durable completion")

    with pytest.raises(KeyboardInterrupt):
        LongTaskWorkflow(
            first,
            task_id="task-2",
            goal="write result",
            fault_hook=interrupt_after_completion,
        ).run_turn("write the result")

    resumed = _runtime(
        tmp_path,
        FixedProvider([_write_call(), LLMResponse(content="recovered")]),
        session_id=first.agent.session_id,
    )
    resumed_counter = resumed.agent.tool_executor
    result = LongTaskWorkflow(resumed, task_id="task-2", goal="write result").run_turn("continue")

    assert result.response == "recovered"
    assert result.task.status is LongTaskStatus.SUCCEEDED
    assert first_counter.calls == 1
    assert resumed_counter.calls == 0
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "durable result\n"


def test_recovery_stops_when_interruption_precedes_known_tool_outcome(tmp_path):
    runtime = _runtime(tmp_path, FixedProvider([_write_call()]))
    counter = runtime.agent.tool_executor

    def interrupt_after_started(phase: str, action_id: str) -> None:
        assert action_id == "tool-write-1"
        if phase == "after_effect_started":
            raise KeyboardInterrupt("test interruption before executor")

    with pytest.raises(KeyboardInterrupt):
        LongTaskWorkflow(
            runtime,
            task_id="task-3",
            goal="write result",
            fault_hook=interrupt_after_started,
        ).run_turn("write the result")

    resumed = _runtime(tmp_path, FixedProvider([LLMResponse(content="must not run")]), session_id=runtime.agent.session_id)
    workflow = LongTaskWorkflow(resumed, task_id="task-3", goal="write result")

    with pytest.raises(LongTaskRecoveryRequired):
        workflow.run_turn("continue")
    assert workflow.store.replay("task-3").status is LongTaskStatus.RECOVERY_REQUIRED
    assert counter.calls == 0
    assert resumed.agent.tool_executor.calls == 0
    assert not (tmp_path / "result.txt").exists()


def test_plain_session_reissues_the_requested_write_after_interruption(tmp_path):
    """Control comparison: Session recovery alone has no completed-effect ledger."""

    first = _runtime(tmp_path, FixedProvider([_write_call()]))
    first.agent.tool_executor = InterruptAfterExecuteExecutor(first.agent.tool_executor)

    with pytest.raises(KeyboardInterrupt):
        first.run_turn("write the result")

    resumed = _runtime(
        tmp_path,
        FixedProvider([_write_call(), LLMResponse(content="ordinary recovery")]),
        session_id=first.agent.session_id,
    )
    assert resumed.run_turn("continue") == "ordinary recovery"
    assert first.agent.tool_executor.calls == 1
    assert resumed.agent.tool_executor.calls == 1
