"""Durable, append-only state for recoverable long-running Tasks.

This module deliberately does not execute tools or grant permissions.  It records
the facts a future Task orchestrator needs to decide whether a side effect is
safe to start, skip, or reconcile after an interruption.  Chat Session events
remain the source of truth for the conversation itself.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from .sessions import SessionStore

LONG_TASK_SCHEMA_VERSION = 1
_SAFE_TASK_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


class LongTaskStoreError(OSError):
    """A required long-task persistence operation could not complete."""


class LongTaskStateError(ValueError):
    """A requested state transition is unsafe for the current Task state."""


class LongTaskLeaseError(LongTaskStateError):
    """Another process owns the exclusive execution lease for this Task."""


class LongTaskStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LIMIT_REACHED = "limit_reached"
    RECOVERY_REQUIRED = "recovery_required"


class LongTaskActionStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EffectDisposition(str, Enum):
    """The only safe choices for a side effect during normal run or recovery."""

    EXECUTE = "execute"
    SKIP_SUCCEEDED = "skip_succeeded"
    RECONCILE_REQUIRED = "reconcile_required"
    BLOCKED_CANCELLED = "blocked_cancelled"
    BLOCKED_LIMIT = "blocked_limit"
    BLOCKED_DEPENDENCY = "blocked_dependency"
    BLOCKED_FAILED = "blocked_failed"


@dataclass(frozen=True, slots=True)
class LongTaskBudget:
    """First long-task budget slice; turn/token budgets remain owned by Runtime."""

    max_effects: int | None = None

    def __post_init__(self) -> None:
        if self.max_effects is not None and self.max_effects <= 0:
            raise ValueError("max_effects must be positive")

    def to_dict(self) -> dict[str, int | None]:
        return {"max_effects": self.max_effects}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LongTaskBudget:
        maximum = value.get("max_effects")
        if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int)):
            raise TypeError("max_effects must be an integer or null")
        return cls(max_effects=maximum)


@dataclass(frozen=True, slots=True)
class LongTaskEvent:
    """One append-only durable fact about a long Task."""

    task_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LONG_TASK_SCHEMA_VERSION
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        json.dumps(self.payload, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "payload": copy.deepcopy(self.payload),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LongTaskEvent:
        if value.get("schema_version") != LONG_TASK_SCHEMA_VERSION:
            raise ValueError("Unsupported long Task event schema version")
        task_id = value.get("task_id")
        event_type = value.get("event_type")
        payload = value.get("payload", {})
        if not isinstance(task_id, str) or not task_id or not isinstance(event_type, str) or not event_type:
            raise TypeError("Long Task event requires task_id and event_type")
        if not isinstance(payload, dict):
            raise TypeError("Long Task event payload must be an object")
        return cls(
            task_id=task_id,
            event_type=event_type,
            payload=payload,
            event_id=str(value.get("event_id") or uuid4().hex),
            created_at=str(value.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass(frozen=True, slots=True)
class LongTaskAction:
    """A planned action; only actions with an effect id can cause side effects."""

    action_id: str
    kind: str
    dependencies: tuple[str, ...] = ()
    effect_id: str | None = None
    status: LongTaskActionStatus = LongTaskActionStatus.PENDING
    result: str | None = None


@dataclass(frozen=True, slots=True)
class TaskCheckpoint:
    """A recovery index over Task facts, never a replacement for them."""

    checkpoint_id: str
    event_cursor: str | None
    session_event_cursor: str | None
    message_projection: list[dict[str, Any]]
    completed_effect_ids: tuple[str, ...]
    pending_action_ids: tuple[str, ...]
    recovery_reason: str
    effects_used: int
    max_effects: int | None


@dataclass
class LongTaskProjection:
    """Current view reconstructed exclusively from durable Task events."""

    task_id: str
    goal: str = ""
    repository_root: Path | None = None
    session_id: str | None = None
    status: LongTaskStatus = LongTaskStatus.RUNNING
    budget: LongTaskBudget = field(default_factory=LongTaskBudget)
    events: list[LongTaskEvent] = field(default_factory=list)
    actions: dict[str, LongTaskAction] = field(default_factory=dict)
    checkpoints: list[TaskCheckpoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recovery_blockers: list[str] = field(default_factory=list)

    @property
    def effects_used(self) -> int:
        return sum(action.status is LongTaskActionStatus.STARTED or action.status is LongTaskActionStatus.SUCCEEDED for action in self.actions.values() if action.effect_id is not None)

    @property
    def completed_effect_ids(self) -> tuple[str, ...]:
        return tuple(sorted(action.effect_id for action in self.actions.values() if action.effect_id and action.status is LongTaskActionStatus.SUCCEEDED))

    @property
    def pending_action_ids(self) -> tuple[str, ...]:
        return tuple(sorted(action_id for action_id, action in self.actions.items() if action.status is LongTaskActionStatus.PENDING))

    @property
    def ambiguous_effect_ids(self) -> tuple[str, ...]:
        return tuple(sorted(action.effect_id for action in self.actions.values() if action.effect_id and action.status is LongTaskActionStatus.STARTED))


@dataclass
class LongTaskLease:
    """One process-owned OS file lock protecting side-effect starts for a Task."""

    task_id: str
    owner_id: str
    _stream: BinaryIO
    _released: bool = False

    @property
    def active(self) -> bool:
        return not self._released and not self._stream.closed

    def release(self) -> None:
        if self._released:
            return
        try:
            _unlock_file(self._stream)
        finally:
            self._stream.close()
            self._released = True

    def __enter__(self) -> LongTaskLease:  # noqa: PYI034 - Python 3.10 compatibility
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.release()


class LongTaskStore:
    """Append and replay long-task events under ``.techpilot/tasks``.

    The caller owns user approval and actual tool execution.  The store only
    permits an effect start after checking the durable effect ledger.
    """

    def __init__(self, directory: Path, *, session_store: SessionStore | None = None) -> None:
        self.directory = directory.resolve()
        self.session_store = session_store
        self._lock = threading.Lock()

    @classmethod
    def for_repository(
        cls,
        repository: Path,
        directory: Path | None = None,
        *,
        session_directory: Path | None = None,
    ) -> LongTaskStore:
        return cls(
            directory if directory is not None else repository / ".techpilot" / "tasks",
            session_store=SessionStore.for_repository(repository, session_directory),
        )

    def create(
        self,
        task_id: str,
        *,
        goal: str,
        repository_root: Path,
        session_id: str,
        budget: LongTaskBudget | None = None,
    ) -> LongTaskProjection:
        safe_id = self._normalize_task_id(task_id)
        if not goal.strip():
            raise ValueError("Long Task goal cannot be empty")
        if not session_id.strip():
            raise ValueError("Long Task session id cannot be empty")
        path = self.path_for(safe_id)
        if path.exists():
            return self.replay(safe_id)
        effective_budget = budget or LongTaskBudget()
        self.append(LongTaskEvent(
            task_id=safe_id,
            event_type="task_created",
            payload={
                "goal": goal,
                "repository_root": str(repository_root.resolve()),
                "session_id": session_id,
                "budget": effective_budget.to_dict(),
            },
        ))
        return self.replay(safe_id)

    def plan_action(
        self,
        task_id: str,
        *,
        action_id: str,
        kind: str,
        effect_id: str | None = None,
        dependencies: tuple[str, ...] = (),
    ) -> LongTaskProjection:
        projection = self.replay(task_id)
        self._require_running(projection)
        safe_action_id = self._normalize_id(action_id, "action id")
        if safe_action_id in projection.actions:
            raise LongTaskStateError(f"Action already exists: {safe_action_id}")
        if not kind.strip():
            raise ValueError("Long Task action kind cannot be empty")
        safe_dependencies = tuple(self._normalize_id(item, "dependency id") for item in dependencies)
        if len(set(safe_dependencies)) != len(safe_dependencies):
            raise ValueError("Long Task action dependencies must be unique")
        missing = sorted(set(safe_dependencies) - set(projection.actions))
        if missing:
            raise LongTaskStateError(f"Action dependencies are not planned: {', '.join(missing)}")
        safe_effect_id = self._normalize_id(effect_id, "effect id") if effect_id is not None else None
        if safe_effect_id and any(action.effect_id == safe_effect_id for action in projection.actions.values()):
            raise LongTaskStateError(f"Effect id already exists: {safe_effect_id}")
        self.append(LongTaskEvent(
            task_id=projection.task_id,
            event_type="action_planned",
            payload={
                "action_id": safe_action_id,
                "kind": kind,
                "effect_id": safe_effect_id,
                "dependencies": list(safe_dependencies),
            },
        ))
        return self.replay(projection.task_id)

    def checkpoint(
        self,
        task_id: str,
        *,
        message_projection: list[dict[str, Any]],
        recovery_reason: str,
        session_event_cursor: str,
        checkpoint_id: str | None = None,
    ) -> TaskCheckpoint:
        projection = self.replay(task_id)
        if projection.status in {LongTaskStatus.CANCELLED, LongTaskStatus.SUCCEEDED, LongTaskStatus.FAILED}:
            raise LongTaskStateError(f"Cannot checkpoint terminal Task: {projection.status.value}")
        if not recovery_reason.strip():
            raise ValueError("Checkpoint recovery reason cannot be empty")
        if not _is_message_projection(message_projection):
            raise TypeError("Checkpoint message projection must be a list of objects")
        safe_session_cursor = self._validate_session_cursor(projection, session_event_cursor)
        safe_checkpoint_id = self._normalize_id(checkpoint_id or uuid4().hex, "checkpoint id")
        cursor = projection.events[-1].event_id if projection.events else None
        self.append(LongTaskEvent(
            task_id=projection.task_id,
            event_type="checkpoint_recorded",
            payload={
                "checkpoint_id": safe_checkpoint_id,
                "event_cursor": cursor,
                "session_event_cursor": safe_session_cursor,
                "message_projection": copy.deepcopy(message_projection),
                "completed_effect_ids": list(projection.completed_effect_ids),
                "pending_action_ids": list(projection.pending_action_ids),
                "recovery_reason": recovery_reason,
                "effects_used": projection.effects_used,
                "max_effects": projection.budget.max_effects,
            },
        ))
        return self.replay(projection.task_id).checkpoints[-1]

    def acquire_lease(self, task_id: str, *, owner_id: str | None = None) -> LongTaskLease:
        """Acquire the single-process execution lease for one existing Task.

        The OS owns release-on-crash behavior.  No historical lease record is
        trusted for recovery because a dead process cannot append its release.
        """

        projection = self.replay(task_id)
        if projection.status in {LongTaskStatus.CANCELLED, LongTaskStatus.SUCCEEDED, LongTaskStatus.FAILED}:
            raise LongTaskLeaseError(f"Cannot lease terminal Task: {projection.status.value}")
        lease_path = self._lease_path(projection.task_id)
        stream: BinaryIO | None = None
        try:
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            stream = lease_path.open("a+b")
            _lock_file(stream)
        except OSError as error:
            if stream is not None:
                stream.close()
            raise LongTaskLeaseError(f"Task is already running: {projection.task_id}") from error
        return LongTaskLease(
            task_id=projection.task_id,
            owner_id=self._normalize_id(owner_id or uuid4().hex, "lease owner id"),
            _stream=stream,
        )

    def pause(self, task_id: str, *, reason: str) -> LongTaskProjection:
        projection = self.replay(task_id)
        self._require_running(projection)
        self.append(LongTaskEvent(task_id=projection.task_id, event_type="task_paused", payload={"reason": reason or "paused"}))
        return self.replay(projection.task_id)

    def complete_action(self, task_id: str, action_id: str, *, result: str = "") -> LongTaskProjection:
        """Complete a non-effect action such as one durable Agent turn."""

        projection = self.replay(task_id)
        self._require_running(projection)
        action = self._action(projection, action_id)
        if action.effect_id is not None or action.status is not LongTaskActionStatus.PENDING:
            raise LongTaskStateError("Only a pending non-effect action can complete")
        self.append(LongTaskEvent(
            task_id=projection.task_id,
            event_type="action_completed",
            payload={"action_id": action.action_id, "result": result},
        ))
        return self.replay(projection.task_id)

    def cancel(self, task_id: str, *, reason: str) -> LongTaskProjection:
        projection = self.replay(task_id)
        if projection.status in {LongTaskStatus.SUCCEEDED, LongTaskStatus.FAILED, LongTaskStatus.CANCELLED}:
            raise LongTaskStateError(f"Cannot cancel terminal Task: {projection.status.value}")
        self.append(LongTaskEvent(task_id=projection.task_id, event_type="task_cancelled", payload={"reason": reason or "cancelled"}))
        return self.replay(projection.task_id)

    def resume(self, task_id: str, *, reason: str) -> LongTaskProjection:
        projection = self.replay(task_id)
        if projection.status not in {LongTaskStatus.RUNNING, LongTaskStatus.PAUSED, LongTaskStatus.RECOVERY_REQUIRED}:
            raise LongTaskStateError(f"Only interrupted Tasks can resume, got: {projection.status.value}")
        if projection.recovery_blockers or projection.ambiguous_effect_ids:
            self.append(LongTaskEvent(
                task_id=projection.task_id,
                event_type="task_recovery_required",
                payload={
                    "reason": reason or "recovery requires reconciliation",
                    "effect_ids": list(projection.ambiguous_effect_ids),
                    "blockers": list(projection.recovery_blockers),
                },
            ))
        else:
            self.append(LongTaskEvent(task_id=projection.task_id, event_type="task_resumed", payload={"reason": reason or "resumed"}))
        return self.replay(projection.task_id)

    def succeed(self, task_id: str, *, result: str = "") -> LongTaskProjection:
        """Record a Task terminal success only after every planned action is durable."""

        projection = self.replay(task_id)
        self._require_running(projection)
        unfinished = sorted(action_id for action_id, action in projection.actions.items() if action.status is not LongTaskActionStatus.SUCCEEDED)
        if unfinished:
            raise LongTaskStateError(f"Cannot succeed with unfinished actions: {', '.join(unfinished)}")
        self.append(LongTaskEvent(task_id=projection.task_id, event_type="task_succeeded", payload={"result": result}))
        return self.replay(projection.task_id)

    def fail(self, task_id: str, *, reason: str) -> LongTaskProjection:
        """Record an explicit terminal failure; recovery cannot start new effects afterwards."""

        projection = self.replay(task_id)
        if projection.status in {LongTaskStatus.CANCELLED, LongTaskStatus.SUCCEEDED, LongTaskStatus.FAILED}:
            raise LongTaskStateError(f"Cannot fail terminal Task: {projection.status.value}")
        if not reason.strip():
            raise ValueError("Task failure reason cannot be empty")
        self.append(LongTaskEvent(task_id=projection.task_id, event_type="task_failed", payload={"reason": reason}))
        return self.replay(projection.task_id)

    def effect_disposition(self, task_id: str, action_id: str) -> EffectDisposition:
        projection = self.replay(task_id)
        action = self._action(projection, action_id)
        if action.effect_id is None:
            raise LongTaskStateError(f"Action does not describe a side effect: {action.action_id}")
        if action.status is LongTaskActionStatus.SUCCEEDED:
            return EffectDisposition.SKIP_SUCCEEDED
        if action.status is LongTaskActionStatus.STARTED:
            return EffectDisposition.RECONCILE_REQUIRED
        if action.status is LongTaskActionStatus.FAILED:
            return EffectDisposition.BLOCKED_FAILED
        if projection.status is LongTaskStatus.CANCELLED:
            return EffectDisposition.BLOCKED_CANCELLED
        if projection.status is LongTaskStatus.LIMIT_REACHED:
            return EffectDisposition.BLOCKED_LIMIT
        if projection.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Task is not running: {projection.status.value}")
        if any(projection.actions[dependency].status is not LongTaskActionStatus.SUCCEEDED for dependency in action.dependencies):
            return EffectDisposition.BLOCKED_DEPENDENCY
        if projection.budget.max_effects is not None and projection.effects_used >= projection.budget.max_effects:
            self.append(LongTaskEvent(
                task_id=projection.task_id,
                event_type="task_limit_reached",
                payload={"limit": "max_effects", "actual": projection.effects_used, "maximum": projection.budget.max_effects},
            ))
            return EffectDisposition.BLOCKED_LIMIT
        return EffectDisposition.EXECUTE

    def start_effect(self, task_id: str, action_id: str, *, lease: LongTaskLease) -> LongTaskProjection:
        self._require_active_lease(task_id, lease)
        disposition = self.effect_disposition(task_id, action_id)
        if disposition is not EffectDisposition.EXECUTE:
            raise LongTaskStateError(f"Effect cannot start: {disposition.value}")
        projection = self.replay(task_id)
        action = self._action(projection, action_id)
        self.append(LongTaskEvent(
            task_id=projection.task_id,
            event_type="effect_started",
            payload={"action_id": action.action_id, "effect_id": action.effect_id},
        ))
        return self.replay(projection.task_id)

    def complete_effect(self, task_id: str, action_id: str, *, result: str = "") -> LongTaskProjection:
        projection = self.replay(task_id)
        action = self._action(projection, action_id)
        if action.effect_id is None or action.status is not LongTaskActionStatus.STARTED:
            raise LongTaskStateError("Only a durably started side effect can complete")
        self.append(LongTaskEvent(
            task_id=projection.task_id,
            event_type="effect_completed",
            payload={"action_id": action.action_id, "effect_id": action.effect_id, "result": result},
        ))
        return self.replay(projection.task_id)

    def fail_effect(self, task_id: str, action_id: str, *, reason: str) -> LongTaskProjection:
        """Record an explicitly rejected or blocked effect as not executed.

        Unknown execution failures stay ``STARTED`` and require reconciliation;
        only a policy result proving the effect did not start may use this path.
        """

        projection = self.replay(task_id)
        action = self._action(projection, action_id)
        if action.effect_id is None or action.status is not LongTaskActionStatus.STARTED:
            raise LongTaskStateError("Only a durably started side effect can fail")
        if not reason.strip():
            raise ValueError("Effect failure reason cannot be empty")
        self.append(LongTaskEvent(
            task_id=projection.task_id,
            event_type="effect_not_executed",
            payload={"action_id": action.action_id, "effect_id": action.effect_id, "reason": reason},
        ))
        return self.replay(projection.task_id)

    def append(self, event: LongTaskEvent) -> None:
        path = self.path_for(event.task_id)
        line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise LongTaskStoreError(f"Could not append long Task event to {path}") from error

    def replay(self, task_id: str) -> LongTaskProjection:
        safe_id = self._normalize_task_id(task_id)
        path = self.path_for(safe_id)
        if not path.exists():
            raise FileNotFoundError(f"Long Task {safe_id!r} does not exist")
        events, warnings = self._read_events(path)
        projection = LongTaskProjection(task_id=safe_id, events=events, warnings=warnings)
        for event in events:
            if event.task_id != safe_id:
                projection.warnings.append("Ignored Task event for a different task id")
                continue
            self._apply_event(projection, event)
        if any(warning.startswith("Ignored invalid long Task event") for warning in warnings):
            projection.recovery_blockers.append("event_log_invalid")
            projection.status = LongTaskStatus.RECOVERY_REQUIRED
        if any(checkpoint.session_event_cursor is None for checkpoint in projection.checkpoints):
            projection.recovery_blockers.append("checkpoint_session_cursor_missing")
            projection.status = LongTaskStatus.RECOVERY_REQUIRED
        return projection

    def path_for(self, task_id: str) -> Path:
        safe_id = self._normalize_task_id(task_id)
        task_directory = (self.directory / safe_id).resolve()
        if task_directory.parent != self.directory:
            raise ValueError("Invalid long Task id")
        return task_directory / "events.jsonl"

    def _lease_path(self, task_id: str) -> Path:
        return self.path_for(task_id).with_name("execution.lock")

    def _apply_event(self, projection: LongTaskProjection, event: LongTaskEvent) -> None:
        payload = event.payload
        if event.event_type == "task_created":
            projection.goal = _required_string(payload, "goal")
            projection.session_id = _required_string(payload, "session_id")
            root = _required_string(payload, "repository_root")
            projection.repository_root = Path(root).resolve()
            budget = payload.get("budget", {})
            if not isinstance(budget, dict):
                raise TypeError("Long Task budget must be an object")
            projection.budget = LongTaskBudget.from_dict(budget)
        elif event.event_type == "action_planned":
            action_id = _required_string(payload, "action_id")
            dependencies = payload.get("dependencies", [])
            if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
                raise TypeError("Long Task dependencies must be a list of strings")
            projection.actions[action_id] = LongTaskAction(
                action_id=action_id,
                kind=_required_string(payload, "kind"),
                effect_id=payload.get("effect_id") if isinstance(payload.get("effect_id"), str) else None,
                dependencies=tuple(dependencies),
            )
        elif event.event_type == "effect_started":
            self._replace_action(projection, _required_string(payload, "action_id"), LongTaskActionStatus.STARTED)
        elif event.event_type == "effect_completed" or event.event_type == "action_completed":
            self._replace_action(
                projection,
                _required_string(payload, "action_id"),
                LongTaskActionStatus.SUCCEEDED,
                result=payload.get("result") if isinstance(payload.get("result"), str) else "",
            )
        elif event.event_type == "effect_not_executed":
            self._replace_action(
                projection,
                _required_string(payload, "action_id"),
                LongTaskActionStatus.FAILED,
                result=payload.get("reason") if isinstance(payload.get("reason"), str) else "",
            )
        elif event.event_type == "task_paused":
            projection.status = LongTaskStatus.PAUSED
        elif event.event_type == "task_resumed":
            projection.status = LongTaskStatus.RUNNING
        elif event.event_type == "task_cancelled":
            projection.status = LongTaskStatus.CANCELLED
        elif event.event_type == "task_succeeded":
            projection.status = LongTaskStatus.SUCCEEDED
        elif event.event_type == "task_failed":
            projection.status = LongTaskStatus.FAILED
        elif event.event_type == "task_limit_reached":
            projection.status = LongTaskStatus.LIMIT_REACHED
        elif event.event_type == "task_recovery_required":
            projection.status = LongTaskStatus.RECOVERY_REQUIRED
        elif event.event_type == "checkpoint_recorded":
            projection.checkpoints.append(TaskCheckpoint(
                checkpoint_id=_required_string(payload, "checkpoint_id"),
                event_cursor=payload.get("event_cursor") if isinstance(payload.get("event_cursor"), str) else None,
                session_event_cursor=(
                    payload.get("session_event_cursor")
                    if isinstance(payload.get("session_event_cursor"), str)
                    else None
                ),
                message_projection=copy.deepcopy(payload.get("message_projection", [])),
                completed_effect_ids=tuple(_string_list(payload, "completed_effect_ids")),
                pending_action_ids=tuple(_string_list(payload, "pending_action_ids")),
                recovery_reason=_required_string(payload, "recovery_reason"),
                effects_used=_required_int(payload, "effects_used"),
                max_effects=payload.get("max_effects") if isinstance(payload.get("max_effects"), int) else None,
            ))

    @staticmethod
    def _replace_action(
        projection: LongTaskProjection,
        action_id: str,
        status: LongTaskActionStatus,
        *,
        result: str | None = None,
    ) -> None:
        action = projection.actions.get(action_id)
        if action is None:
            raise LongTaskStateError(f"Effect refers to unknown action: {action_id}")
        projection.actions[action_id] = LongTaskAction(
            action_id=action.action_id,
            kind=action.kind,
            dependencies=action.dependencies,
            effect_id=action.effect_id,
            status=status,
            result=action.result if result is None else result,
        )

    def _read_events(self, path: Path) -> tuple[list[LongTaskEvent], list[str]]:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise LongTaskStoreError(f"Could not read long Task events from {path}") from error
        events: list[LongTaskEvent] = []
        warnings: list[str] = []
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines, start=1):
            if index == len(lines) and not line.endswith((b"\n", b"\r")):
                warnings.append("Ignored incomplete trailing long Task event")
                break
            try:
                events.append(LongTaskEvent.from_dict(json.loads(line.decode("utf-8"))))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                warnings.append(f"Ignored invalid long Task event on line {index}: {error}")
        return events, warnings

    @staticmethod
    def _require_running(projection: LongTaskProjection) -> None:
        if projection.status is not LongTaskStatus.RUNNING:
            raise LongTaskStateError(f"Task is not running: {projection.status.value}")

    def _validate_session_cursor(self, projection: LongTaskProjection, session_event_cursor: str) -> str:
        if self.session_store is None:
            raise LongTaskStateError("Long Task Session store is not configured")
        if not isinstance(session_event_cursor, str) or not session_event_cursor.strip():
            raise ValueError("Checkpoint Session event cursor cannot be empty")
        if projection.session_id is None or projection.repository_root is None:
            raise LongTaskStateError("Long Task is missing its Session or repository identity")
        session = self.session_store.replay(projection.session_id)
        if session.repository_root != projection.repository_root:
            raise LongTaskStateError("Checkpoint Session belongs to a different repository")
        if not any(event.event_id == session_event_cursor for event in session.events):
            raise LongTaskStateError("Checkpoint Session event cursor does not exist")
        return session_event_cursor

    @staticmethod
    def _require_active_lease(task_id: str, lease: LongTaskLease) -> None:
        if not isinstance(lease, LongTaskLease) or not lease.active:
            raise LongTaskLeaseError("A live Task execution lease is required before starting an effect")
        if lease.task_id != LongTaskStore._normalize_task_id(task_id):
            raise LongTaskLeaseError("Task execution lease belongs to a different Task")

    @staticmethod
    def _action(projection: LongTaskProjection, action_id: str) -> LongTaskAction:
        action = projection.actions.get(action_id)
        if action is None:
            raise LongTaskStateError(f"Unknown Task action: {action_id}")
        return action

    @staticmethod
    def _normalize_id(value: str | None, label: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{label} must be a string")
        normalized = _SAFE_TASK_ID_RE.sub("-", value.strip()).strip(".-_")
        if not normalized:
            raise ValueError(f"Invalid {label}")
        return normalized[:100]

    @classmethod
    def _normalize_task_id(cls, task_id: str) -> str:
        return cls._normalize_id(task_id, "long Task id")


def _is_message_projection(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise TypeError(f"Long Task event requires string {key}")
    return item


def _required_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise TypeError(f"Long Task event requires non-negative integer {key}")
    return item


def _string_list(value: dict[str, Any], key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list) or not all(isinstance(element, str) for element in item):
        raise TypeError(f"Long Task event requires string list {key}")
    return item


def _lock_file(stream: BinaryIO) -> None:
    """Acquire one cross-process byte lock; the OS releases it on process exit."""

    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
        os.fsync(stream.fileno())
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
