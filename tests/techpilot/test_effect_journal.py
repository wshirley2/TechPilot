"""Durable execution-boundary tests shared by Chat and Goal runtimes."""

from __future__ import annotations

from dataclasses import dataclass

from techpilot.engine.agent import ToolExecutionContext
from techpilot.engine.events import CallbackEventSink
from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect, ToolExecutionDescription
from techpilot.engine.tool_results import ToolResult, ToolStatus, tool_result
from techpilot.engine.tools.base import Tool
from techpilot.runtime import EffectJournalExecutor, EffectJournalState, EffectJournalStore


class MarkerTool(Tool):
    name = "marker_write"
    description = "Write a test marker."
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

    def execute(self, value: str) -> str:
        return f"tool executed: {value}"


@dataclass
class RecordingExecutor:
    result: str = "marker written"
    calls: int = 0

    def describe_call(self, tool: Tool, arguments: dict[str, object]) -> ToolExecutionDescription:
        del tool, arguments
        return ToolExecutionDescription(ToolEffect.WRITE, ToolConcurrency.EXCLUSIVE, ("marker",))

    # Deliberately omits ``prepared_call``: public wrappers must remain
    # compatible with executors written before PreparedToolCall existed.
    def execute_call(self, tool: Tool, arguments: dict[str, object], *, tool_call_id: str, execution_context=None) -> str:
        del tool, arguments, tool_call_id, execution_context
        self.calls += 1
        return self.result


def _context(session_id: str = "session-1") -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id=session_id,
        turn_id="turn-1",
        round_index=0,
        event_sink=CallbackEventSink(lambda _event: None),
    )


def test_completed_effect_is_returned_from_journal_without_reexecuting_delegate(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    delegate = RecordingExecutor()
    first = EffectJournalExecutor(delegate, store)

    assert first.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context()) == "marker written"
    assert delegate.calls == 1

    resumed = EffectJournalExecutor(delegate, store)
    assert resumed.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context()) == "marker written"
    assert delegate.calls == 1
    entry = store.replay("session-1").entries["write-1"]
    assert entry.state is EffectJournalState.FINISHED
    assert entry.tool_status is ToolStatus.COMPLETED


def test_unknown_effect_blocks_later_side_effects_until_reconciled(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    delegate = RecordingExecutor(ToolResult("[effect unknown] partial write", ToolStatus.EFFECT_UNKNOWN))
    executor = EffectJournalExecutor(delegate, store)

    first = executor.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context())
    second = executor.execute_call(MarkerTool(), {"value": "two"}, tool_call_id="write-2", execution_context=_context())

    assert tool_result(first).status is ToolStatus.EFFECT_UNKNOWN
    assert tool_result(second).status is ToolStatus.EFFECT_UNKNOWN
    assert delegate.calls == 1
    assert store.replay("session-1").entries["write-1"].state is EffectJournalState.STARTED


def test_operator_reconciliation_terminates_unknown_effect_and_allows_later_effect(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )

    entry = store.reconcile(
        "session-1",
        "write-1",
        state=EffectJournalState.NOT_EXECUTED,
        result=ToolResult("[reconciled not executed] target did not change", ToolStatus.NOT_EXECUTED),
        reason="Operator inspected the target and found no change.",
    )

    assert entry.state is EffectJournalState.NOT_EXECUTED
    assert entry.reconciliation_reason == "Operator inspected the target and found no change."
    replayed = store.replay("session-1").entries["write-1"]
    assert replayed.state is EffectJournalState.NOT_EXECUTED
    assert replayed.reconciliation_reason == entry.reconciliation_reason

    delegate = RecordingExecutor()
    result = EffectJournalExecutor(delegate, store).execute_call(
        MarkerTool(), {"value": "two"}, tool_call_id="write-2", execution_context=_context(),
    )
    assert result == "marker written"
    assert delegate.calls == 1
