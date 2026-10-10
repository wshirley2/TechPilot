"""Durable side-effect facts shared by ordinary Chat and long-running Goals.

Session events describe the conversation.  This journal instead records the
execution boundary of a non-read Tool Call before the delegate can change the
outside world.  A missing terminal record is deliberately treated as unknown.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO

from techpilot.engine.agent import ToolExecutionContext
from techpilot.engine.events import RuntimeEventType
from techpilot.engine.tool_execution import (
    PreparedToolCall,
    ToolEffect,
    ToolExecutionDescription,
    default_tool_description,
)
from techpilot.engine.tool_results import ToolResult, ToolStatus, tool_result
from techpilot.engine.tools.base import Tool

from .sessions import SessionEvent, SessionStore

_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, _SCHEMA_VERSION})
_SAFE_SESSION_ID = re.compile(r"[^A-Za-z0-9._-]+")
_CONTROL_BLOCK_VERIFIER = "runtime_execution_control_block_v1"


class EffectJournalError(OSError):
    """A required durable side-effect record could not be written or replayed."""


class EffectJournalState(str, Enum):
    STARTED = "started"
    FINISHED = "finished"
    NOT_EXECUTED = "not_executed"


@dataclass(frozen=True, slots=True)
class EffectVerification:
    """Host-produced evidence required to settle an interrupted effect."""

    verifier: str
    evidence_refs: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.verifier, str) or not self.verifier.strip() or len(self.verifier.strip()) > 100:
            raise ValueError("Effect verifier must be a non-empty string up to 100 characters")
        if not 1 <= len(self.evidence_refs) <= 32:
            raise ValueError("Effect verification requires 1–32 evidence references")
        if any(not isinstance(ref, str) or not ref.strip() or len(ref.strip()) > 200 for ref in self.evidence_refs):
            raise ValueError("Effect verification evidence reference is invalid")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("Effect verification evidence references must be unique")
        if not isinstance(self.detail, str) or not self.detail.strip() or len(self.detail.strip()) > 2_000:
            raise ValueError("Effect verification detail must be a non-empty string up to 2000 characters")

    def to_payload(self) -> dict[str, object]:
        return {
            "verifier": self.verifier.strip(),
            "evidence_refs": [ref.strip() for ref in self.evidence_refs],
            "detail": self.detail.strip(),
        }

    @classmethod
    def from_payload(cls, value: object) -> EffectVerification:
        if not isinstance(value, dict) or set(value) != {"verifier", "evidence_refs", "detail"}:
            raise ValueError("Effect verification has invalid fields")
        evidence = value["evidence_refs"]
        if not isinstance(evidence, list):
            raise TypeError("Effect verification evidence references must be a list")
        return cls(
            verifier=value["verifier"],
            evidence_refs=tuple(evidence),
            detail=value["detail"],
        )


@dataclass(frozen=True, slots=True)
class EffectJournalEntry:
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    effect: ToolEffect
    state: EffectJournalState = EffectJournalState.STARTED
    result: str | None = None
    tool_status: ToolStatus | None = None
    started_at: str = ""
    completed_at: str | None = None
    reconciliation_reason: str | None = None
    verification: EffectVerification | None = None

    @property
    def is_ambiguous(self) -> bool:
        return self.state is EffectJournalState.STARTED


@dataclass
class EffectJournalProjection:
    session_id: str
    entries: dict[str, EffectJournalEntry] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ambiguous_entries(self) -> tuple[EffectJournalEntry, ...]:
        return tuple(entry for entry in self.entries.values() if entry.is_ambiguous)


class EffectJournalStore:
    """Append-only execution facts under one Runtime Session directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.session_store = SessionStore(self.directory.parent)
        self._lock = threading.RLock()

    @classmethod
    def for_session_directory(cls, session_directory: Path) -> EffectJournalStore:
        return cls(session_directory / "effects")

    def replay(self, session_id: str) -> EffectJournalProjection:
        safe_id = self._normalize_session_id(session_id)
        path = self.path_for(safe_id)
        projection = EffectJournalProjection(session_id=safe_id)
        if not path.exists():
            return projection
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise EffectJournalError(f"Could not read Effect Journal {path}") from error
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines, start=1):
            if index == len(lines) and not line.endswith((b"\n", b"\r")):
                projection.warnings.append("Ignored incomplete trailing Effect Journal event")
                break
            try:
                value = json.loads(line.decode("utf-8"))
                self._apply(projection, value)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                projection.warnings.append(f"Ignored invalid Effect Journal event on line {index}: {error}")
        if projection.warnings:
            # A malformed journal must never permit another effect in this Session.
            projection.entries["__journal_invalid__"] = EffectJournalEntry(
                session_id=safe_id,
                tool_call_id="__journal_invalid__",
                tool_name="journal",
                arguments={},
                effect=ToolEffect.UNKNOWN,
            )
        return projection

    def start(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        effect: ToolEffect,
    ) -> EffectJournalEntry:
        safe_session = self._normalize_session_id(session_id)
        safe_call = self._normalize_tool_call_id(tool_call_id)
        existing = self.replay(safe_session).entries.get(safe_call)
        if existing is not None:
            if existing.tool_name != tool_name or existing.arguments != dict(arguments) or existing.effect is not ToolEffect(effect):
                raise ValueError("Tool Call ID is already bound to a different effect")
            return existing
        created_at = datetime.now(timezone.utc).isoformat()
        entry = EffectJournalEntry(
            session_id=safe_session,
            tool_call_id=safe_call,
            tool_name=tool_name,
            arguments=dict(arguments),
            effect=ToolEffect(effect),
            started_at=created_at,
        )
        self._append({
            "schema_version": _SCHEMA_VERSION,
            "event_type": "effect_started",
            "session_id": safe_session,
            "tool_call_id": safe_call,
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "effect": entry.effect.value,
            "created_at": created_at,
        })
        return entry

    def finish(self, session_id: str, tool_call_id: str, result: str | ToolResult) -> EffectJournalEntry:
        entry = self._required_started(session_id, tool_call_id)
        outcome = tool_result(result)
        if outcome.status is ToolStatus.EFFECT_UNKNOWN:
            raise ValueError("Unknown effect results must remain started for reconciliation")
        state = EffectJournalState.NOT_EXECUTED if outcome.status in {
            ToolStatus.DENIED,
            ToolStatus.NOT_EXECUTED,
        } else EffectJournalState.FINISHED
        completed_at = datetime.now(timezone.utc).isoformat()
        self._append({
            "schema_version": _SCHEMA_VERSION,
            "event_type": "effect_finished",
            "session_id": entry.session_id,
            "tool_call_id": entry.tool_call_id,
            "state": state.value,
            "result": str(outcome),
            "tool_status": outcome.status.value,
            "created_at": completed_at,
        })
        return EffectJournalEntry(
            session_id=entry.session_id,
            tool_call_id=entry.tool_call_id,
            tool_name=entry.tool_name,
            arguments=entry.arguments,
            effect=entry.effect,
            state=state,
            result=str(outcome),
            tool_status=outcome.status,
            started_at=entry.started_at,
            completed_at=completed_at,
        )

    def verify_non_execution(self, session_id: str, tool_call_id: str) -> EffectJournalEntry | None:
        """Settle only a durable Runtime control block that preceded delegation."""

        with self._lock, self._recovery_lease(session_id):
            projection = self.replay(session_id)
            if projection.warnings:
                return None
            entry = projection.entries.get(self._normalize_tool_call_id(tool_call_id))
            if entry is None or entry.state is not EffectJournalState.STARTED:
                return None
            verification = self._control_block_verification(entry)
            if verification is None:
                return None
            result = self._blocked_result(entry)
            completed_at = datetime.now(timezone.utc).isoformat()
            self._append({
                "schema_version": _SCHEMA_VERSION,
                "event_type": "effect_verified",
                "session_id": entry.session_id,
                "tool_call_id": entry.tool_call_id,
                "state": EffectJournalState.NOT_EXECUTED.value,
                "result": str(result),
                "tool_status": result.status.value,
                "verification": verification.to_payload(),
                "created_at": completed_at,
            })
            return EffectJournalEntry(
                session_id=entry.session_id,
                tool_call_id=entry.tool_call_id,
                tool_name=entry.tool_name,
                arguments=entry.arguments,
                effect=entry.effect,
                state=EffectJournalState.NOT_EXECUTED,
                result=str(result),
                tool_status=result.status,
                started_at=entry.started_at,
                completed_at=completed_at,
                verification=verification,
            )

    def complete_verified_session_tool_result(self, session_id: str, tool_call_id: str) -> SessionEvent | None:
        """Append a missing Tool reply only for one isolated, proven blocked call."""

        with self._lock, self._recovery_lease(session_id):
            journal = self.replay(session_id)
            if journal.warnings:
                return None
            entry = journal.entries.get(self._normalize_tool_call_id(tool_call_id))
            if (
                entry is None
                or entry.state is not EffectJournalState.NOT_EXECUTED
                or entry.verification is None
                or entry.verification != self._control_block_verification(entry)
            ):
                return None
            try:
                session = self.session_store.replay(entry.session_id)
            except (OSError, TypeError, ValueError):
                return None
            if any(
                not warning.startswith("Excluded incomplete tool round from model history:")
                for warning in session.warnings
            ):
                return None
            request = next(
                (event for event in session.events if event.event_id == entry.verification.evidence_refs[0]),
                None,
            )
            if request is None:
                return None
            request_index = session.events.index(request)
            round_requests = [
                event for event in session.events
                if event.event_type == RuntimeEventType.TOOL_REQUESTED.value
                and (event.turn_id, event.round_index) == (request.turn_id, request.round_index)
            ]
            if len(round_requests) != 1:
                return None
            # A late reply cannot be inserted after a later model-visible turn.
            forbidden_after_request = {
                RuntimeEventType.TURN_STARTED.value,
                RuntimeEventType.PROVIDER_STARTED.value,
                RuntimeEventType.TOOL_REQUESTED.value,
                RuntimeEventType.TOOL_COMPLETED.value,
                RuntimeEventType.CONTEXT_COMPRESSED.value,
                RuntimeEventType.TURN_COMPLETED.value,
            }
            if any(
                event.event_type in forbidden_after_request
                for event in session.events[request_index + 1:]
            ):
                return None
            completions = [
                event for event in session.events
                if event.event_type == RuntimeEventType.TOOL_COMPLETED.value
                and event.tool_call_id == entry.tool_call_id
            ]
            if completions:
                return None
            recovered = SessionEvent(
                event_type=RuntimeEventType.TOOL_COMPLETED.value,
                session_id=entry.session_id,
                turn_id=request.turn_id,
                round_index=request.round_index,
                tool_call_id=entry.tool_call_id,
                payload=self._recovered_session_result(entry, entry.verification),
            )
            self.session_store.append(recovered)
            return recovered

    @staticmethod
    def _blocked_result(entry: EffectJournalEntry) -> ToolResult:
        return ToolResult(
            f"[not executed] Runtime execution control blocked {entry.tool_name} before delegation.",
            ToolStatus.NOT_EXECUTED,
        )

    @staticmethod
    def _recovered_session_result(entry: EffectJournalEntry, verification: EffectVerification) -> dict[str, object]:
        return {
            "tool_name": entry.tool_name,
            "result": entry.result or str(EffectJournalStore._blocked_result(entry)),
            "tool_status": ToolStatus.NOT_EXECUTED.value,
            "interrupted": True,
            "recovery": {
                "source": "effect_journal_verified",
                "verifier": verification.verifier,
                "evidence_event_ids": list(verification.evidence_refs),
            },
        }

    def _control_block_verification(self, entry: EffectJournalEntry) -> EffectVerification | None:
        try:
            session = self.session_store.replay(entry.session_id)
        except (OSError, TypeError, ValueError):
            return None
        if (
            session.repository_root is None
            or entry.effect is ToolEffect.READ
            or any(
                not warning.startswith("Excluded incomplete tool round from model history:")
                for warning in session.warnings
            )
            or len({event.event_id for event in session.events}) != len(session.events)
        ):
            return None
        requests = [
            (index, event) for index, event in enumerate(session.events)
            if event.event_type == RuntimeEventType.TOOL_REQUESTED.value and event.tool_call_id == entry.tool_call_id
        ]
        controls = [
            (index, event) for index, event in enumerate(session.events)
            if event.event_type == RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value
            and event.tool_call_id == entry.tool_call_id
        ]
        if len(requests) != 1 or len(controls) != 1:
            return None
        request_index, request = requests[0]
        control_index, control = controls[0]
        if (
            request_index >= control_index
            or request.turn_id != control.turn_id
            or request.round_index != control.round_index
            or request.payload.get("tool_name") != entry.tool_name
            or request.payload.get("arguments") != entry.arguments
            or control.payload.get("tool_name") != entry.tool_name
            or control.payload.get("required_control") != "block"
        ):
            return None
        try:
            request_at = datetime.fromisoformat(request.created_at)
            started_at = datetime.fromisoformat(entry.started_at)
            control_at = datetime.fromisoformat(control.created_at)
            if (
                request_at.tzinfo is None
                or started_at.tzinfo is None
                or control_at.tzinfo is None
                or not request_at <= started_at <= control_at
            ):
                return None
        except ValueError:
            return None
        verification = EffectVerification(
            verifier=_CONTROL_BLOCK_VERIFIER,
            evidence_refs=(request.event_id, control.event_id),
            detail="Durable Runtime execution control blocked this Tool Call before delegation.",
        )
        completions = [
            event for event in session.events
            if event.event_type == RuntimeEventType.TOOL_COMPLETED.value
            and event.tool_call_id == entry.tool_call_id
        ]
        if len(completions) > 1 or (
            completions and (
                completions[0].turn_id != request.turn_id
                or completions[0].round_index != request.round_index
                or session.events.index(completions[0]) <= control_index
                or completions[0].payload != self._recovered_session_result(entry, verification)
            )
        ):
            return None
        return verification

    def path_for(self, session_id: str) -> Path:
        safe_id = self._normalize_session_id(session_id)
        path = (self.directory / f"{safe_id}.jsonl").resolve()
        if path.parent != self.directory:
            raise ValueError("Invalid Session id")
        return path

    @contextmanager
    def _recovery_lease(self, session_id: str) -> Iterator[None]:
        """Serialize replay-and-append recovery across Host processes."""

        lock_path = self.path_for(session_id).with_suffix(".recovery.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as stream:
            _lock_recovery_file(stream)
            try:
                yield
            finally:
                _unlock_recovery_file(stream)

    def _required_started(self, session_id: str, tool_call_id: str) -> EffectJournalEntry:
        entry = self.replay(session_id).entries.get(self._normalize_tool_call_id(tool_call_id))
        if entry is None or entry.state is not EffectJournalState.STARTED:
            raise ValueError("Effect Journal entry is not started")
        return entry

    def _append(self, value: dict[str, object]) -> None:
        session_id = value["session_id"]
        if not isinstance(session_id, str):
            raise TypeError("Effect Journal event requires session_id")
        path = self.path_for(session_id)
        line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._lock, path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise EffectJournalError(f"Could not append Effect Journal event to {path}") from error

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        if not isinstance(session_id, str):
            raise TypeError("Session id must be a string")
        value = _SAFE_SESSION_ID.sub("-", session_id.strip()).strip(".-_")
        if not value:
            raise ValueError("Invalid Session id")
        return value[:100]

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: str) -> str:
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError("Tool call id must be a non-empty string")
        return tool_call_id.strip()

    def _apply(self, projection: EffectJournalProjection, value: dict[str, object]) -> None:
        schema_version = value.get("schema_version")
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("Unsupported Effect Journal schema")
        if value.get("session_id") != projection.session_id:
            raise ValueError("Effect Journal session mismatch")
        tool_call_id = value.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise TypeError("Effect Journal event requires tool_call_id")
        if value.get("event_type") == "effect_started":
            if tool_call_id in projection.entries:
                raise ValueError("Duplicate effect_started event")
            tool_name = value.get("tool_name")
            arguments = value.get("arguments")
            effect = value.get("effect")
            if not isinstance(tool_name, str) or not isinstance(arguments, dict) or not isinstance(effect, str):
                raise TypeError("Invalid effect_started event")
            projection.entries[tool_call_id] = EffectJournalEntry(
                session_id=projection.session_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                effect=ToolEffect(effect),
                started_at=str(value.get("created_at", "")),
            )
            return
        event_type = value.get("event_type")
        if event_type not in {"effect_finished", "effect_reconciled", "effect_verified"}:
            raise ValueError("Unknown Effect Journal event")
        prior = projection.entries.get(tool_call_id)
        state = value.get("state")
        result = value.get("result")
        status = value.get("tool_status")
        if (
            prior is None
            or prior.state is not EffectJournalState.STARTED
            or not isinstance(state, str)
            or not isinstance(result, str)
            or not isinstance(status, str)
        ):
            raise TypeError(f"Invalid {event_type} event")
        terminal_state = EffectJournalState(state)
        terminal_status = ToolStatus(status)
        if terminal_state not in {EffectJournalState.FINISHED, EffectJournalState.NOT_EXECUTED}:
            raise ValueError(f"Invalid terminal Effect Journal state: {terminal_state.value}")
        if event_type == "effect_finished" and (
            (terminal_state is EffectJournalState.NOT_EXECUTED and terminal_status not in {ToolStatus.DENIED, ToolStatus.NOT_EXECUTED})
            or (terminal_state is EffectJournalState.FINISHED and terminal_status not in {ToolStatus.COMPLETED, ToolStatus.ERROR})
        ):
            raise ValueError("Effect Journal terminal state and Tool status disagree")
        reconciliation_reason = None
        verification = None
        if event_type == "effect_reconciled":
            if schema_version != 1:
                raise ValueError("New operator reconciliation events are not supported")
            reason = value.get("reconciliation_reason")
            if (
                not isinstance(reason, str)
                or not reason.strip()
                or terminal_status is not (
                    ToolStatus.COMPLETED
                    if terminal_state is EffectJournalState.FINISHED
                    else ToolStatus.NOT_EXECUTED
                )
            ):
                raise TypeError("Invalid effect_reconciled event")
            reconciliation_reason = reason.strip()
        elif event_type == "effect_verified":
            if schema_version != _SCHEMA_VERSION:
                raise ValueError("Effect verification requires the current Effect Journal schema")
            verification = EffectVerification.from_payload(value.get("verification"))
            if (
                terminal_state is not EffectJournalState.NOT_EXECUTED
                or terminal_status is not ToolStatus.NOT_EXECUTED
                or result != str(self._blocked_result(prior))
                or verification != self._control_block_verification(prior)
            ):
                raise ValueError("Effect verification is not backed by matching Runtime control evidence")
        projection.entries[tool_call_id] = EffectJournalEntry(
            session_id=prior.session_id,
            tool_call_id=prior.tool_call_id,
            tool_name=prior.tool_name,
            arguments=prior.arguments,
            effect=prior.effect,
            state=terminal_state,
            result=result,
            tool_status=terminal_status,
            started_at=prior.started_at,
            completed_at=str(value.get("created_at", "")),
            reconciliation_reason=reconciliation_reason,
            verification=verification,
        )


class EffectJournalExecutor:
    """Decorator that makes non-read Tool Calls recoverable before delegation."""

    def __init__(self, delegate: object | None, store: EffectJournalStore) -> None:
        self.delegate = delegate
        self.store = store
        self._turn_stop_message: str | None = None

    def begin_turn(self) -> None:
        self._turn_stop_message = None
        begin = getattr(self.delegate, "begin_turn", None)
        if callable(begin):
            begin()

    def consume_turn_stop_message(self) -> str | None:
        consume = getattr(self.delegate, "consume_turn_stop_message", None)
        delegated = consume() if callable(consume) else None
        return delegated or self._turn_stop_message

    def prepare_call(self, tool: Tool, arguments: dict[str, Any]) -> PreparedToolCall | None:
        prepare = getattr(self.delegate, "prepare_call", None)
        return prepare(tool, dict(arguments)) if callable(prepare) else None

    def describe_call(self, tool: Tool, arguments: dict[str, Any]) -> ToolExecutionDescription | None:
        describe = getattr(self.delegate, "describe_call", None)
        return describe(tool, dict(arguments)) if callable(describe) else None

    def execute(self, tool: Tool, arguments: dict[str, Any]) -> str:
        return self.execute_call(tool, arguments, tool_call_id="uncorrelated-tool-call")

    def execute_call(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        *,
        tool_call_id: str,
        execution_context: ToolExecutionContext | None = None,
        prepared_call: PreparedToolCall | None = None,
    ) -> str:
        description = self._description(tool, arguments, prepared_call)
        if description.effect is ToolEffect.READ:
            return self._delegate(tool, arguments, tool_call_id, execution_context, prepared_call)
        if execution_context is None:
            return ToolResult(
                "[effect unknown] Durable execution requires a Runtime ToolExecutionContext; tool was not started.",
                ToolStatus.EFFECT_UNKNOWN,
            )
        projection = self.store.replay(execution_context.session_id)
        if projection.warnings:
            return self._stop_unknown(projection.ambiguous_entries[0])
        existing = projection.entries.get(tool_call_id)
        if existing is not None:
            if existing.tool_name != tool.name or existing.arguments != dict(arguments) or existing.effect is not description.effect:
                return self._stop_unknown(existing)
            if existing.state is EffectJournalState.STARTED:
                verified = self.store.verify_non_execution(execution_context.session_id, tool_call_id)
                if verified is None:
                    return self._stop_unknown(existing)
                return ToolResult(verified.result or "", verified.tool_status or ToolStatus.ERROR)
            return ToolResult(existing.result or "", existing.tool_status or ToolStatus.ERROR)
        if projection.ambiguous_entries:
            for entry in projection.ambiguous_entries:
                self.store.verify_non_execution(execution_context.session_id, entry.tool_call_id)
            remaining = self.store.replay(execution_context.session_id).ambiguous_entries
            if remaining:
                return self._stop_unknown(remaining[0])
        self.store.start(
            session_id=execution_context.session_id,
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            arguments=dict(arguments),
            effect=description.effect,
        )
        result = self._delegate(tool, arguments, tool_call_id, execution_context, prepared_call)
        outcome = tool_result(result)
        if outcome.status is ToolStatus.EFFECT_UNKNOWN:
            return self._stop_unknown(None)
        self.store.finish(execution_context.session_id, tool_call_id, outcome)
        if outcome.status in {ToolStatus.DENIED, ToolStatus.NOT_EXECUTED, ToolStatus.ERROR}:
            self._turn_stop_message = "A durable Tool effect did not complete successfully; no further tools were executed."
        return outcome

    def _description(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        prepared_call: PreparedToolCall | None,
    ) -> ToolExecutionDescription:
        if (
            isinstance(prepared_call, PreparedToolCall)
            and prepared_call.tool_name == tool.name
            and dict(prepared_call.requested_arguments) == dict(arguments)
        ):
            return prepared_call.description
        described = self.describe_call(tool, arguments)
        if isinstance(described, ToolExecutionDescription):
            return described
        return default_tool_description(tool.name, dict(arguments))

    def _delegate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        tool_call_id: str,
        execution_context: ToolExecutionContext | None,
        prepared_call: PreparedToolCall | None,
    ) -> str:
        execute_call = getattr(self.delegate, "execute_call", None)
        if callable(execute_call):
            kwargs: dict[str, object] = {"tool_call_id": tool_call_id}
            if self._accepts_keyword(execute_call, "execution_context"):
                kwargs["execution_context"] = execution_context
            if prepared_call is not None and self._accepts_keyword(execute_call, "prepared_call"):
                kwargs["prepared_call"] = prepared_call
            return execute_call(tool, dict(arguments), **kwargs)
        execute = getattr(self.delegate, "execute", None)
        if callable(execute):
            return execute(tool, dict(arguments))
        return tool.execute(**arguments)

    @staticmethod
    def _accepts_keyword(callback: object, name: str) -> bool:
        """Keep compatibility with older ToolExecutor decorators.

        ``prepared_call`` was added after the first executor protocol.  An
        executor that does not know it must still receive the durable wrapper,
        rather than failing before it can record an effect boundary.
        """

        try:
            parameters = inspect.signature(callback).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _stop_unknown(self, entry: EffectJournalEntry | None) -> ToolResult:
        self._turn_stop_message = "Effect recovery required: inspect the previous operation before any further side effects."
        detail = (
            f"Previous effect {entry.tool_call_id!r} has no durable outcome."
            if entry is not None
            else "The current effect outcome is unknown."
        )
        return ToolResult(f"[effect unknown] {detail}", ToolStatus.EFFECT_UNKNOWN)


def _lock_recovery_file(stream: BinaryIO) -> None:
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
        os.fsync(stream.fileno())
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _unlock_recovery_file(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
