"""Stable identity and result contracts shared by TechPilot runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class RuntimeResultStatus(str, Enum):
    """Terminal outcomes for one Chat or Goal turn."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LIMIT_REACHED = "limit_reached"
    ESCALATION_REQUIRED = "escalation_required"


@dataclass(frozen=True, slots=True)
class TaskRuntimePaths:
    """Canonical persistence and working-directory boundaries for one Runtime."""

    working_directory: Path
    session_directory: Path

    def __post_init__(self) -> None:
        working_directory = self.working_directory.resolve()
        session_directory = self.session_directory.resolve()
        object.__setattr__(self, "working_directory", working_directory)
        object.__setattr__(self, "session_directory", session_directory)

    @classmethod
    def for_runtime(
        cls,
        working_directory: Path,
        session_directory: Path | None = None,
    ) -> TaskRuntimePaths:
        """Build the default Session path layout for Chat."""

        working = working_directory.resolve()
        sessions = (
            session_directory.resolve()
            if session_directory is not None
            else working / ".techpilot" / "sessions"
        )
        return cls(working, sessions)

    def to_dict(self) -> dict[str, str]:
        return {
            "working_directory": str(self.working_directory),
            "session_directory": str(self.session_directory),
        }


@dataclass(frozen=True, slots=True)
class TaskRuntimeResult:
    """Immutable terminal result for one Chat or Goal turn."""

    status: RuntimeResultStatus
    response: str = ""
    reason: str | None = None
    error_type: str | None = None
    limit: str | None = None
    actual: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        status = RuntimeResultStatus(self.status)
        if status is RuntimeResultStatus.SUCCEEDED:
            if self.error_type is not None or self.limit is not None:
                raise ValueError("A successful Runtime result cannot contain an error or limit")
        elif not self.reason:
            raise ValueError("A non-success Runtime result requires a reason")
        if status is RuntimeResultStatus.FAILED and not self.error_type:
            raise ValueError("A failed Runtime result requires an error type")
        if status is RuntimeResultStatus.LIMIT_REACHED and not self.limit:
            raise ValueError("A limit-reached Runtime result requires a limit name")
        if status is not RuntimeResultStatus.LIMIT_REACHED and (
            self.limit is not None or self.actual is not None or self.maximum is not None
        ):
            raise ValueError("Limit facts are only valid for a limit-reached Runtime result")
        object.__setattr__(self, "status", status)

    @classmethod
    def from_terminal_event(
        cls,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> TaskRuntimeResult | None:
        """Project an existing terminal Runtime Event into the shared contract."""

        if event_type == "turn_completed":
            return cls(
                status=RuntimeResultStatus.SUCCEEDED,
                response=_string(payload.get("content")),
            )
        if event_type == "turn_interrupted":
            return cls(
                status=RuntimeResultStatus.CANCELLED,
                reason=_string(payload.get("reason")) or "Runtime turn interrupted",
            )
        if event_type == "turn_failed":
            return cls(
                status=RuntimeResultStatus.FAILED,
                reason=_string(payload.get("error")) or "Runtime turn failed",
                error_type=_string(payload.get("error_type")) or "RuntimeError",
            )
        if event_type == "turn_limit_reached":
            limit = _string(payload.get("limit")) or "tool_rounds"
            return cls(
                status=RuntimeResultStatus.LIMIT_REACHED,
                response=_string(payload.get("result")),
                reason=f"Runtime limit reached: {limit}",
                limit=limit,
                actual=_number(payload.get("actual", payload.get("max_rounds"))),
                maximum=_number(payload.get("maximum", payload.get("max_rounds"))),
            )
        return None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskRuntimeResult:
        return cls(
            status=RuntimeResultStatus(str(value["status"])),
            response=_string(value.get("response")),
            reason=_optional_string(value.get("reason")),
            error_type=_optional_string(value.get("error_type")),
            limit=_optional_string(value.get("limit")),
            actual=_number(value.get("actual")),
            maximum=_number(value.get("maximum")),
        )

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "status": self.status.value,
            "response": self.response,
            "reason": self.reason,
            "error_type": self.error_type,
            "limit": self.limit,
            "actual": self.actual,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class TaskRuntimeIdentity:
    """Immutable identity facts for one Chat Runtime."""

    session_id: str
    working_directory: Path

    def __post_init__(self) -> None:
        session_id = self.session_id.strip()
        if not session_id:
            raise ValueError("Task Runtime requires a non-empty session id")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "working_directory", self.working_directory.resolve())

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "working_directory": str(self.working_directory),
        }


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None
