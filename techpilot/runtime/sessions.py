"""Versioned append-only event sessions for TechPilot Chat."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from techpilot.engine.events import EventSink, RuntimeEvent, RuntimeEventType

from .contracts import TaskRuntimeResult

SESSION_SCHEMA_VERSION = 1
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SessionStoreError(OSError):
    """A required Session persistence operation could not complete."""


@dataclass(frozen=True)
class SessionEvent:
    """One durable, JSON-serializable fact in a Chat Session."""

    event_type: str
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    round_index: int = 0
    tool_call_id: str | None = None
    schema_version: int = SESSION_SCHEMA_VERSION
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        json.dumps(self.payload, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "round_index": self.round_index,
            "tool_call_id": self.tool_call_id,
            "created_at": self.created_at,
            "payload": copy.deepcopy(self.payload),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SessionEvent:
        if not isinstance(value, dict):
            raise TypeError("Session event must be an object")
        version = value.get("schema_version", 0)
        if type(version) is not int:
            raise TypeError("Session event schema version must be an integer")
        if version == 0:
            # C1 RuntimeEvent JSON did not carry an explicit schema version.
            # It is losslessly readable as the first C4 event version.
            version = SESSION_SCHEMA_VERSION
        if version != SESSION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Session event schema version: {version}")
        payload = value.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("Session event payload must be an object")
        event_type = value.get("event_type")
        session_id = value.get("session_id")
        if not isinstance(event_type, str) or not event_type.strip() or not isinstance(session_id, str) or not session_id.strip():
            raise TypeError("Session event requires event_type and session_id")
        event_id = value.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise TypeError("Session event requires a durable event_id")
        created_at = value.get("created_at")
        if not isinstance(created_at, str) or not created_at.strip():
            raise TypeError("Session event requires created_at")
        turn_id = value.get("turn_id")
        if turn_id is not None and (not isinstance(turn_id, str) or not turn_id.strip()):
            raise TypeError("Session event turn_id must be a non-empty string or null")
        round_index = value.get("round_index", 0)
        if type(round_index) is not int or round_index < 0:
            raise TypeError("Session event round_index must be a non-negative integer")
        tool_call_id = value.get("tool_call_id")
        if tool_call_id is not None and (not isinstance(tool_call_id, str) or not tool_call_id.strip()):
            raise TypeError("Session event tool_call_id must be a non-empty string or null")
        if (
            event_type in {RuntimeEventType.TOOL_REQUESTED.value, RuntimeEventType.TOOL_COMPLETED.value}
            and (turn_id is None or tool_call_id is None)
        ):
            raise TypeError("Tool event requires turn_id and tool_call_id")
        if event_type == RuntimeEventType.TOOL_REQUESTED.value:
            if not isinstance(payload.get("tool_name"), str) or not payload["tool_name"].strip():
                raise TypeError("Tool request requires a tool_name")
            if not isinstance(payload.get("arguments", {}), dict):
                raise TypeError("Tool request arguments must be an object")
        if event_type == "session_created":
            root = payload.get("repository_root")
            if not isinstance(root, str) or not root.strip() or not Path(root).is_absolute():
                raise TypeError("Session creation requires an absolute repository_root")
        return cls(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            turn_id=turn_id,
            round_index=round_index,
            tool_call_id=tool_call_id,
            schema_version=version,
            event_id=event_id,
            created_at=created_at,
        )


@dataclass
class SessionProjection:
    """Recoverable model view plus the immutable event history behind it."""

    session_id: str
    repository_root: Path | None = None
    model: str | None = None
    mode: str = "chat"
    status: str = "active"
    task_id: str | None = None
    run_id: str | None = None
    source_repository_root: Path | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    model_messages: list[dict[str, Any]] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provider_calls: int = 0
    tool_rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_result: TaskRuntimeResult | None = None
    pending_isolation_requests: list[dict[str, Any]] = field(default_factory=list)
    review_artifact_paths: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class SessionStore:
    """Append and replay event sessions rooted in one local repository."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self._lock = threading.Lock()

    @classmethod
    def for_repository(cls, repository: Path, directory: Path | None = None) -> SessionStore:
        return cls(directory if directory is not None else repository / ".techpilot" / "sessions")

    def create(
        self,
        session_id: str,
        *,
        repository_root: Path,
        model: str,
        mode: str = "chat",
        task_id: str | None = None,
        run_id: str | None = None,
        source_repository_root: Path | None = None,
    ) -> SessionProjection:
        safe_id = self._normalize_session_id(session_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path_for(safe_id)
        if path.exists():
            return self.replay(safe_id)
        self.append(
            SessionEvent(
                event_type="session_created",
                session_id=safe_id,
                payload={
                    "repository_root": str(repository_root.resolve()),
                    "model": model,
                    "mode": mode,
                    "task_id": task_id,
                    "run_id": run_id,
                    "source_repository_root": str((source_repository_root or repository_root).resolve()),
                },
            )
        )
        return self.replay(safe_id)

    def append_runtime(self, event: RuntimeEvent) -> None:
        self.append(SessionEvent(
            event_type=event.event_type.value,
            session_id=event.session_id,
            turn_id=event.turn_id,
            round_index=event.round_index,
            tool_call_id=event.tool_call_id,
            event_id=event.event_id,
            created_at=event.created_at.isoformat(),
            payload=event.payload,
        ))

    def append(self, event: SessionEvent) -> None:
        path = self._path_for(event.session_id)
        line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._lock, path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise SessionStoreError(f"Could not append Session event to {path}") from error

    def replay(self, session_id: str) -> SessionProjection:
        safe_id = self._normalize_session_id(session_id)
        path = self._path_for(safe_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {safe_id!r} does not exist")
        projection = SessionProjection(session_id=safe_id)
        events, warnings = self._read_events(path)
        projection.events = events
        projection.warnings.extend(warnings)
        pending_calls: dict[tuple[str | None, int], list[dict[str, Any]]] = {}
        emitted_tool_messages: set[tuple[str | None, int]] = set()
        model_messages: list[dict[str, Any]] = []
        model_pending_calls: dict[tuple[str | None, int], dict[str, dict[str, Any]]] = {}
        model_call_order: dict[tuple[str | None, int], list[str]] = {}
        model_completed_calls: dict[tuple[str | None, int], dict[str, str]] = {}
        model_assistant_content: dict[tuple[str | None, int], str | None] = {}

        for event in events:
            payload = event.payload
            terminal_result = TaskRuntimeResult.from_terminal_event(event.event_type, payload)
            if terminal_result is not None:
                projection.last_result = terminal_result
            if event.event_type == "session_created":
                root = payload.get("repository_root")
                projection.repository_root = Path(root).resolve() if isinstance(root, str) else None
                projection.model = payload.get("model") if isinstance(payload.get("model"), str) else None
                projection.mode = payload.get("mode") if isinstance(payload.get("mode"), str) else "chat"
                projection.task_id = payload.get("task_id") if isinstance(payload.get("task_id"), str) else None
                projection.run_id = payload.get("run_id") if isinstance(payload.get("run_id"), str) else None
                source_root = payload.get("source_repository_root")
                projection.source_repository_root = (
                    Path(source_root).resolve()
                    if isinstance(source_root, str)
                    else projection.repository_root
                )
            elif event.event_type == "session_resumed":
                projection.status = "active"
            elif event.event_type == "session_model_changed":
                if isinstance(payload.get("model"), str):
                    projection.model = payload["model"]
            elif event.event_type == "runtime_result_recorded":
                try:
                    projection.last_result = TaskRuntimeResult.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    projection.warnings.append("Ignored invalid Runtime result event")
            elif event.event_type == RuntimeEventType.TURN_STARTED.value:
                user_input = payload.get("user_input")
                if isinstance(user_input, str):
                    projection.messages.append({"role": "user", "content": user_input})
                    model_messages.append({"role": "user", "content": user_input})
            elif event.event_type == RuntimeEventType.PROVIDER_STARTED.value:
                projection.provider_calls += 1
            elif event.event_type == RuntimeEventType.TOOL_REQUESTED.value:
                key = (event.turn_id, event.round_index)
                pending_calls.setdefault(key, []).append({
                    "id": event.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": payload.get("tool_name", "unknown"),
                        "arguments": json.dumps(payload.get("arguments", {}), ensure_ascii=False),
                    },
                    "content": payload.get("assistant_content") or None,
                })
                if not isinstance(event.tool_call_id, str) or not event.tool_call_id:
                    projection.warnings.append("Excluded invalid tool request from model history")
                else:
                    calls = model_pending_calls.setdefault(key, {})
                    order = model_call_order.setdefault(key, [])
                    assistant_content = payload.get("assistant_content")
                    if key not in model_assistant_content or model_assistant_content[key] is None:
                        model_assistant_content[key] = (
                            assistant_content if isinstance(assistant_content, str) and assistant_content else None
                        )
                    if event.tool_call_id in calls:
                        projection.warnings.append(
                            f"Excluded duplicate tool request {event.tool_call_id!r} from model history"
                        )
                    else:
                        calls[event.tool_call_id] = {
                            "id": event.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": payload.get("tool_name", "unknown"),
                                "arguments": json.dumps(payload.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                        order.append(event.tool_call_id)
            elif event.event_type == RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value:
                if payload.get("required_control") == "isolate":
                    projection.pending_isolation_requests.append({
                        "task_id": payload.get("task_id"),
                        "session_id": event.session_id,
                        "turn_id": event.turn_id,
                        "round_index": event.round_index,
                        "tool_call_id": event.tool_call_id,
                        "tool_name": payload.get("tool_name"),
                        "summary": payload.get("normalized_summary"),
                        "reasons": copy.deepcopy(payload.get("reasons", [])),
                    })
            elif event.event_type == "isolation_pending":
                tool_call_id = payload.get("tool_call_id")
                for index, request in enumerate(projection.pending_isolation_requests):
                    if request.get("tool_call_id") == tool_call_id:
                        projection.pending_isolation_requests[index] = copy.deepcopy(payload)
                        break
            elif event.event_type in {"isolation_upgrade_completed", "isolation_cancelled"}:
                tool_call_id = payload.get("tool_call_id")
                projection.pending_isolation_requests = [
                    request
                    for request in projection.pending_isolation_requests
                    if request.get("tool_call_id") != tool_call_id
                ]
                if event.event_type == "isolation_upgrade_completed":
                    for key in ("patch", "validation", "report", "events"):
                        value = payload.get(key)
                        if isinstance(value, str) and value not in projection.review_artifact_paths:
                            projection.review_artifact_paths.append(value)
            elif event.event_type == RuntimeEventType.TOOL_COMPLETED.value:
                key = (event.turn_id, event.round_index)
                calls = pending_calls.get(key, [])
                if key not in emitted_tool_messages and calls:
                    content = next((call.pop("content") for call in calls if call.get("content")), None)
                    for call in calls:
                        call.pop("content", None)
                    projection.messages.append({"role": "assistant", "content": content, "tool_calls": calls})
                    emitted_tool_messages.add(key)
                    projection.tool_rounds += 1
                projection.messages.append({
                    "role": "tool",
                    "tool_call_id": event.tool_call_id,
                    "content": str(payload.get("result", "")),
                })
                model_calls = model_pending_calls.get(key, {})
                if not isinstance(event.tool_call_id, str) or event.tool_call_id not in model_calls:
                    projection.warnings.append(
                        f"Excluded orphan tool result {event.tool_call_id!r} from model history"
                    )
                else:
                    completed = model_completed_calls.setdefault(key, {})
                    if event.tool_call_id in completed:
                        projection.warnings.append(
                            f"Excluded duplicate tool result {event.tool_call_id!r} from model history"
                        )
                    else:
                        completed[event.tool_call_id] = str(payload.get("result", ""))
                        order = model_call_order[key]
                        if all(call_id in completed for call_id in order):
                            model_messages.append({
                                "role": "assistant",
                                "content": model_assistant_content[key],
                                "tool_calls": [copy.deepcopy(model_calls[call_id]) for call_id in order],
                            })
                            model_messages.extend({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": completed[call_id],
                            } for call_id in order)
                            del model_pending_calls[key]
                            del model_call_order[key]
                            del model_completed_calls[key]
                            del model_assistant_content[key]
            elif event.event_type == RuntimeEventType.TURN_COMPLETED.value:
                content = payload.get("content")
                if isinstance(content, str):
                    projection.messages.append({"role": "assistant", "content": content})
                    model_messages.append({"role": "assistant", "content": content})
                projection.prompt_tokens += _as_non_negative_int(payload.get("prompt_tokens"))
                projection.completion_tokens += _as_non_negative_int(payload.get("completion_tokens"))
            elif event.event_type == RuntimeEventType.TURN_INTERRUPTED.value:
                projection.status = "interrupted"
            elif event.event_type == RuntimeEventType.TURN_FAILED.value:
                projection.status = "failed"
            elif event.event_type == RuntimeEventType.TURN_LIMIT_REACHED.value:
                projection.status = "limit_reached"
            elif event.event_type == RuntimeEventType.CONTEXT_COMPRESSED.value:
                candidate = payload.get("message_projection")
                if _is_provider_message_list(candidate):
                    # A valid compression projection is a trusted checkpoint:
                    # it replaces the prior model view even if earlier JSONL
                    # events contained an unreadable gap. Later events still
                    # need their own request/result pairing before they join it.
                    model_messages = copy.deepcopy(candidate)
                    model_pending_calls.clear()
                    model_call_order.clear()
                    model_completed_calls.clear()
                    model_assistant_content.clear()
                else:
                    projection.warnings.append("Ignored invalid context compression projection")

        for key, order in model_call_order.items():
            unresolved = [call_id for call_id in order if call_id not in model_completed_calls.get(key, {})]
            if unresolved:
                projection.warnings.append(
                    "Excluded incomplete tool round from model history: "
                    f"turn={key[0]!r}, round={key[1]}, pending={unresolved!r}"
                )
        projection.model_messages = model_messages
        return projection

    def list(self) -> list[SessionProjection]:
        if not self.directory.exists():
            return []
        projections: list[SessionProjection] = []
        for path in self.directory.glob("*.jsonl"):
            try:
                projections.append(self.replay(path.stem))
            except (OSError, ValueError):
                continue
        return sorted(
            projections,
            key=lambda item: item.events[-1].created_at if item.events else "",
            reverse=True,
        )

    def _read_events(self, path: Path) -> tuple[list[SessionEvent], list[str]]:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise SessionStoreError(f"Could not read Session events from {path}") from error
        events: list[SessionEvent] = []
        warnings: list[str] = []
        seen_event_ids: set[str] = set()
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines, start=1):
            is_last = index == len(lines)
            if is_last and not line.endswith((b"\n", b"\r")):
                warnings.append("Ignored incomplete trailing Session event")
                break
            try:
                value = json.loads(line.decode("utf-8"))
                event = SessionEvent.from_dict(value)
                if event.session_id != path.stem:
                    raise ValueError(f"Session event belongs to {event.session_id!r}, expected {path.stem!r}")
                if event.event_type == "session_created" and index != 1:
                    raise ValueError("Session creation event must be the first record")
                if event.event_id in seen_event_ids:
                    raise ValueError(f"Duplicate Session event_id: {event.event_id!r}")
                seen_event_ids.add(event.event_id)
                events.append(event)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                warnings.append(f"Ignored invalid Session event on line {index}: {error}")
        return events, warnings

    def _path_for(self, session_id: str) -> Path:
        path = (self.directory / f"{self._normalize_session_id(session_id)}.jsonl").resolve()
        if path.parent != self.directory:
            raise ValueError("Invalid Session id")
        return path

    def path_for(self, session_id: str) -> Path:
        """Return the canonical event path for one normalized Session id."""

        return self._path_for(session_id)

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        name = session_id.strip().replace("\\", "/").split("/")[-1]
        name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
        if not name:
            raise ValueError("Invalid Session id")
        return name[:100]


class SessionEventSink:
    """Persist RuntimeEvents before forwarding them to a live observer."""

    def __init__(self, store: SessionStore, downstream: EventSink) -> None:
        self.store = store
        self.downstream = downstream
        self.persistence_error: Exception | None = None
        self.last_result: TaskRuntimeResult | None = None

    @property
    def last_turn_streamed(self) -> bool:
        return bool(getattr(self.downstream, "last_turn_streamed", False))

    def emit(self, event: RuntimeEvent) -> None:
        terminal_result = TaskRuntimeResult.from_terminal_event(
            event.event_type.value,
            event.payload,
        )
        if terminal_result is not None:
            self.last_result = terminal_result
        try:
            self.store.append_runtime(event)
        except Exception as error:
            self.persistence_error = self.persistence_error or error
        self.downstream.emit(event)

    def record(self, event_type: str, session_id: str, payload: dict[str, Any] | None = None) -> None:
        try:
            self.store.append(SessionEvent(event_type=event_type, session_id=session_id, payload=payload or {}))
        except Exception as error:
            self.persistence_error = self.persistence_error or error

    def record_result(self, session_id: str, result: TaskRuntimeResult) -> None:
        """Persist an orchestration-level result and expose it to the live Runtime."""

        self.last_result = result
        self.record("runtime_result_recorded", session_id, result.to_dict())

    def ensure_persisted(self) -> None:
        if self.persistence_error is not None:
            raise SessionStoreError("Session persistence failed; the latest turn may not be recoverable") from self.persistence_error
        ensure_downstream = getattr(self.downstream, "ensure_persisted", None)
        if callable(ensure_downstream):
            ensure_downstream()


def _as_non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _is_provider_message_list(value: object) -> bool:
    """Accept only snapshots that can be sent back to a tool-calling Provider."""

    if not isinstance(value, list):
        return False
    pending_tool_calls: set[str] = set()
    for message in value:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            return False
        role = message["role"]
        if pending_tool_calls:
            if role != "tool":
                return False
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or tool_call_id not in pending_tool_calls:
                return False
            pending_tool_calls.remove(tool_call_id)
            continue
        if role == "tool" or role not in {"system", "user", "assistant"}:
            return False
        if role != "assistant" or "tool_calls" not in message:
            continue
        tool_calls = message["tool_calls"]
        if not isinstance(tool_calls, list) or not tool_calls:
            return False
        call_ids: set[str] = set()
        for call in tool_calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                return False
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                return False
            if call["id"] in call_ids:
                return False
            call_ids.add(call["id"])
        pending_tool_calls = call_ids
    return not pending_tool_calls
