"""Durable execution-boundary tests shared by Chat and Goal runtimes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from techpilot.chat.executor import RepositoryToolExecutor
from techpilot.engine.agent import ToolExecutionContext
from techpilot.engine.events import CallbackEventSink, RuntimeEventType
from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect, ToolExecutionDescription
from techpilot.engine.tool_results import ToolResult, ToolStatus, tool_result
from techpilot.engine.tools.base import Tool
from techpilot.engine.tools.write import WriteFileTool
from techpilot.runtime import EffectJournalExecutor, EffectJournalState, EffectJournalStore
from techpilot.runtime.sessions import SessionEvent, SessionStore


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


def test_reused_tool_call_id_with_different_arguments_cannot_reuse_old_result(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    delegate = RecordingExecutor()
    executor = EffectJournalExecutor(delegate, store)
    executor.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context())

    reused = executor.execute_call(
        MarkerTool(), {"value": "different"}, tool_call_id="write-1", execution_context=_context(),
    )

    assert tool_result(reused).status is ToolStatus.EFFECT_UNKNOWN
    assert delegate.calls == 1
    with pytest.raises(ValueError, match="different effect"):
        store.start(
            session_id="session-1",
            tool_call_id="write-1",
            tool_name="marker_write",
            arguments={"value": "different"},
            effect=ToolEffect.WRITE,
        )


def test_corrupt_journal_does_not_return_a_previously_completed_result(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    delegate = RecordingExecutor()
    executor = EffectJournalExecutor(delegate, store)
    executor.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context())
    with store.path_for("session-1").open("a", encoding="utf-8") as stream:
        stream.write("{damaged event}\n")

    result = executor.execute_call(
        MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context(),
    )

    assert tool_result(result).status is ToolStatus.EFFECT_UNKNOWN
    assert delegate.calls == 1


def test_error_effect_is_durable_and_is_not_reexecuted_after_restart(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    error = ToolResult("Error: command exited with code 1", ToolStatus.ERROR)
    delegate = RecordingExecutor(error)

    first = EffectJournalExecutor(delegate, store)
    result = first.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context())

    assert tool_result(result).status is ToolStatus.ERROR
    assert delegate.calls == 1
    entry = store.replay("session-1").entries["write-1"]
    assert entry.state is EffectJournalState.FINISHED
    assert entry.tool_status is ToolStatus.ERROR

    resumed = EffectJournalExecutor(delegate, store)
    assert tool_result(
        resumed.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context())
    ).status is ToolStatus.ERROR
    assert delegate.calls == 1


def test_unknown_effect_blocks_later_side_effects_without_durable_proof(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    delegate = RecordingExecutor(ToolResult("[effect unknown] partial write", ToolStatus.EFFECT_UNKNOWN))
    executor = EffectJournalExecutor(delegate, store)

    first = executor.execute_call(MarkerTool(), {"value": "one"}, tool_call_id="write-1", execution_context=_context())
    second = executor.execute_call(MarkerTool(), {"value": "two"}, tool_call_id="write-2", execution_context=_context())

    assert tool_result(first).status is ToolStatus.EFFECT_UNKNOWN
    assert tool_result(second).status is ToolStatus.EFFECT_UNKNOWN
    assert delegate.calls == 1
    assert store.replay("session-1").entries["write-1"].state is EffectJournalState.STARTED


def test_durable_control_block_proves_non_execution_and_allows_later_effect(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=tmp_path, model="fake")
    request = SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "arguments": {"value": "one"}},
    )
    sessions.append(request)
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    control = SessionEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "required_control": "block"},
    )
    sessions.append(control)

    entry = store.verify_non_execution("session-1", "write-1")

    assert entry is not None
    assert entry.state is EffectJournalState.NOT_EXECUTED
    assert entry.verification is not None
    assert entry.verification.verifier == "runtime_execution_control_block_v1"
    assert entry.verification.evidence_refs == (request.event_id, control.event_id)
    replayed = store.replay("session-1").entries["write-1"]
    assert replayed.state is EffectJournalState.NOT_EXECUTED
    assert replayed.verification == entry.verification

    delegate = RecordingExecutor()
    result = EffectJournalExecutor(delegate, store).execute_call(
        MarkerTool(), {"value": "two"}, tool_call_id="write-2", execution_context=_context(),
    )
    assert result == "marker written"
    assert delegate.calls == 1


def test_verified_block_completes_session_once_with_provenance(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=tmp_path, model="fake")
    request = SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "arguments": {"value": "one"}},
    )
    sessions.append(request)
    store = EffectJournalStore.for_session_directory(sessions.directory)
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    control = SessionEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "required_control": "block"},
    )
    sessions.append(control)
    assert store.verify_non_execution("session-1", "write-1") is not None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(
            lambda _: EffectJournalStore.for_session_directory(sessions.directory).complete_verified_session_tool_result(
                "session-1", "write-1",
            ),
            range(2),
        ))
    assert sum(event is not None for event in outcomes) == 1
    completed = next(event for event in outcomes if event is not None)

    assert completed is not None
    assert completed.payload["recovery"] == {
        "source": "effect_journal_verified",
        "verifier": "runtime_execution_control_block_v1",
        "evidence_event_ids": [request.event_id, control.event_id],
    }
    assert store.complete_verified_session_tool_result("session-1", "write-1") is None
    projection = sessions.replay("session-1")
    assert projection.warnings == []
    assert projection.model_messages[-1]["tool_call_id"] == "write-1"
    assert len([event for event in projection.events if event.event_type == RuntimeEventType.TOOL_COMPLETED.value]) == 1


@pytest.mark.parametrize("durable_before_failure", [False, True])
def test_completion_write_failure_is_idempotent(tmp_path, monkeypatch, durable_before_failure):
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=tmp_path, model="fake")
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "arguments": {"value": "one"}},
    ))
    store = EffectJournalStore.for_session_directory(sessions.directory)
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "required_control": "block"},
    ))
    store.verify_non_execution("session-1", "write-1")
    append = store.session_store.append

    def append_then_fail(event):
        if durable_before_failure:
            append(event)
        raise OSError("simulated Session append failure")

    monkeypatch.setattr(store.session_store, "append", append_then_fail)
    with pytest.raises(OSError, match="simulated Session append failure"):
        store.complete_verified_session_tool_result("session-1", "write-1")

    restarted = EffectJournalStore.for_session_directory(sessions.directory)
    retry = restarted.complete_verified_session_tool_result("session-1", "write-1")
    assert (retry is None) is durable_before_failure
    assert restarted.replay("session-1").warnings == []
    assert len([
        event for event in sessions.replay("session-1").events
        if event.event_type == RuntimeEventType.TOOL_COMPLETED.value
    ]) == 1


def test_verified_block_cannot_insert_reply_after_later_turn(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=tmp_path, model="fake")
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "arguments": {"value": "one"}},
    ))
    store = EffectJournalStore.for_session_directory(sessions.directory)
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "required_control": "block"},
    ))
    store.verify_non_execution("session-1", "write-1")
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.TURN_STARTED.value,
        session_id="session-1",
        turn_id="turn-2",
        payload={"user_input": "later message"},
    ))

    assert store.complete_verified_session_tool_result("session-1", "write-1") is None
    projection = sessions.replay("session-1")
    assert any("incomplete tool round" in warning for warning in projection.warnings)
    assert not any(event.event_type == RuntimeEventType.TOOL_COMPLETED.value for event in projection.events)


@pytest.mark.parametrize(
    "case", ["missing", "not_blocked", "wrong_arguments", "before_start", "request_after_start", "duplicate_request"]
)
def test_unproven_control_outcome_keeps_effect_started(tmp_path, case):
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=tmp_path, model="fake")
    request = SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={
            "tool_name": "marker_write",
            "arguments": {"value": "different" if case == "wrong_arguments" else "one"},
        },
    )
    if case != "request_after_start":
        sessions.append(request)
    control = SessionEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "required_control": "direct" if case == "not_blocked" else "block"},
    )
    if case == "before_start":
        sessions.append(control)
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    if case == "request_after_start":
        assert store.replay("session-1").entries["write-1"].request_event_id is None
        sessions.append(request)
    if case not in {"missing", "before_start"}:
        sessions.append(control)
    if case == "duplicate_request":
        sessions.append(request)

    assert store.verify_non_execution("session-1", "write-1") is None
    assert store.replay("session-1").entries["write-1"].state is EffectJournalState.STARTED


def test_legacy_started_effect_without_request_marker_stays_unknown(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=tmp_path, model="fake")
    request = SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "arguments": {"value": "one"}},
    )
    sessions.append(request)
    store = EffectJournalStore.for_session_directory(sessions.directory)
    store._append({
        "schema_version": 2,
        "event_type": "effect_started",
        "session_id": "session-1",
        "tool_call_id": "write-1",
        "tool_name": "marker_write",
        "arguments": {"value": "one"},
        "effect": "write",
        "created_at": request.created_at,
    })
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "required_control": "block"},
    ))

    assert store.verify_non_execution("session-1", "write-1") is None
    entry = store.replay("session-1").entries["write-1"]
    assert entry.request_event_id is None
    assert entry.state is EffectJournalState.STARTED


def test_forged_verification_event_cannot_unlock_an_effect(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    store._append({
        "schema_version": 2,
        "event_type": "effect_verified",
        "session_id": "session-1",
        "tool_call_id": "write-1",
        "state": "not_executed",
        "result": "[not executed] Runtime execution control blocked marker_write before delegation.",
        "tool_status": "not executed",
        "verification": {
            "verifier": "runtime_execution_control_block_v1",
            "evidence_refs": ["invented-request", "invented-control"],
            "detail": "Durable Runtime execution control blocked this Tool Call before delegation.",
        },
        "created_at": "2026-10-09T00:01:00+00:00",
    })

    projection = store.replay("session-1")
    assert projection.warnings
    assert projection.entries["write-1"].state is EffectJournalState.STARTED
    assert projection.ambiguous_entries


def test_damaged_session_evidence_cannot_prove_non_execution(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=tmp_path, model="fake")
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "arguments": {"value": "one"}},
    ))
    store = EffectJournalStore.for_session_directory(sessions.directory)
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    with sessions.path_for("session-1").open("a", encoding="utf-8") as stream:
        stream.write("{damaged middle event}\n")
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "marker_write", "required_control": "block"},
    ))

    assert sessions.replay("session-1").warnings
    assert store.verify_non_execution("session-1", "write-1") is None
    assert store.replay("session-1").entries["write-1"].state is EffectJournalState.STARTED


def test_new_operator_reconciliation_event_cannot_unlock_an_effect(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    store.start(
        session_id="session-1",
        tool_call_id="write-1",
        tool_name="marker_write",
        arguments={"value": "one"},
        effect=ToolEffect.WRITE,
    )
    store._append({
        "schema_version": 2,
        "event_type": "effect_reconciled",
        "session_id": "session-1",
        "tool_call_id": "write-1",
        "state": "not_executed",
        "result": "[reconciled not executed] user said so",
        "tool_status": "not executed",
        "reconciliation_reason": "User said so",
        "created_at": "2026-10-09T00:01:00+00:00",
    })

    projection = store.replay("session-1")
    assert projection.warnings
    assert projection.entries["write-1"].state is EffectJournalState.STARTED


def test_real_control_block_survives_lost_journal_terminal_without_running_tool(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("original\n", encoding="utf-8")
    sessions = SessionStore(tmp_path / "sessions")
    sessions.create("session-1", repository_root=repository, model="fake")
    arguments = {"file_path": str(outside), "content": "changed\n"}
    sessions.append(SessionEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED.value,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="write-1",
        payload={"tool_name": "write_file", "arguments": arguments},
    ))
    context = ToolExecutionContext(
        session_id="session-1",
        turn_id="turn-1",
        round_index=0,
        event_sink=CallbackEventSink(sessions.append_runtime),
    )
    store = EffectJournalStore.for_session_directory(sessions.directory)
    executor = EffectJournalExecutor(RepositoryToolExecutor(repository), store)

    def lose_terminal(*_args):
        raise OSError("simulated crash before Journal terminal")

    monkeypatch.setattr(store, "finish", lose_terminal)
    with pytest.raises(OSError, match="simulated crash"):
        executor.execute_call(WriteFileTool(), arguments, tool_call_id="write-1", execution_context=context)

    assert outside.read_text(encoding="utf-8") == "original\n"
    restarted = EffectJournalStore.for_session_directory(sessions.directory)
    assert restarted.replay("session-1").entries["write-1"].state is EffectJournalState.STARTED
    result = EffectJournalExecutor(RepositoryToolExecutor(repository), restarted).execute_call(
        WriteFileTool(), arguments, tool_call_id="write-1", execution_context=context,
    )
    assert tool_result(result).status is ToolStatus.NOT_EXECUTED
    assert restarted.replay("session-1").entries["write-1"].verification is not None
    assert outside.read_text(encoding="utf-8") == "original\n"


def test_legacy_operator_reconciliation_events_remain_readable(tmp_path):
    store = EffectJournalStore.for_session_directory(tmp_path / "sessions")
    path = store.path_for("session-1")
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join([
            json.dumps({
                "schema_version": 1,
                "event_type": "effect_started",
                "session_id": "session-1",
                "tool_call_id": "write-1",
                "tool_name": "marker_write",
                "arguments": {"value": "one"},
                "effect": "write",
                "created_at": "2026-10-09T00:00:00+00:00",
            }),
            json.dumps({
                "schema_version": 1,
                "event_type": "effect_reconciled",
                "session_id": "session-1",
                "tool_call_id": "write-1",
                "state": "not_executed",
                "result": "[reconciled not executed] legacy operator record",
                "tool_status": "not executed",
                "reconciliation_reason": "Legacy operator record.",
                "created_at": "2026-10-09T00:01:00+00:00",
            }),
            "",
        ]),
        encoding="utf-8",
    )

    entry = store.replay("session-1").entries["write-1"]

    assert entry.state is EffectJournalState.NOT_EXECUTED
    assert entry.reconciliation_reason == "Legacy operator record."
    assert entry.verification is None
