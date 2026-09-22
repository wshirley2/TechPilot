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
        version = value.get("schema_version", 0)
        if version == 0:
            # C1 RuntimeEvent JSON did not carry an explicit schema version.
            # It is losslessly readable as the first C4 event version.
            version = SESSION_SCHEMA_VERSION
        if version != SESSION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Session event schema version: {version}")
        payload = value.get("payload", {})
        if not isinstance(payload, dict):
            raise TypeError("Session event payload must be an object")
        event_type = value.get("event_type")
        session_id = value.get("session_id")
        if not isinstance(event_type, str) or not isinstance(session_id, str):
            raise TypeError("Session event requires event_type and session_id")
        return cls(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            turn_id=value.get("turn_id"),
            round_index=int(value.get("round_index", 0)),
            tool_call_id=value.get("tool_call_id"),
            schema_version=version,
            event_id=str(value.get("event_id") or uuid4().hex),
            created_at=str(value.get("created_at") or datetime.now(timezone.utc).isoformat()),
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
        compressed_projection: list[dict[str, Any]] | None = None

        for event in events:
            message_count_before = len(projection.messages)
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
            elif event.event_type == RuntimeEventType.TURN_COMPLETED.value:
                content = payload.get("content")
                if isinstance(content, str):
                    projection.messages.append({"role": "assistant", "content": content})
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
                if _is_message_list(candidate):
                    compressed_projection = copy.deepcopy(candidate)

            # A compression event is a checkpoint for the model-visible
            # projection, not the end of the Session.  Continue rebuilding
            # messages produced after that checkpoint so a resumed Agent sees
            # the later context as well as the compressed history.
            if (
                compressed_projection is not None
                and event.event_type != RuntimeEventType.CONTEXT_COMPRESSED.value
                and len(projection.messages) > message_count_before
            ):
                compressed_projection.extend(
                    copy.deepcopy(projection.messages[message_count_before:])
                )

        projection.model_messages = compressed_projection or copy.deepcopy(projection.messages)
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
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines, start=1):
            is_last = index == len(lines)
            if is_last and not line.endswith((b"\n", b"\r")):
                warnings.append("Ignored incomplete trailing Session event")
                break
            try:
                value = json.loads(line.decode("utf-8"))
                events.append(SessionEvent.from_dict(value))
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


def _is_message_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)
