"""Bounded, user-authored stage contracts for durable Goals.

These contracts describe work. They never grant Tool permissions or prove that
an outcome was completed; execution evidence remains in Runtime and Session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

_STEP_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
MAX_GOAL_STEPS = 12


def _text(value: object, label: str, *, limit: int = 500) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Goal plan {label} must be a string")
    result = value.strip()
    if not result or len(result) > limit:
        raise ValueError(f"Goal plan {label} must contain 1–{limit} characters")
    return result


class GoalStageCheckKind(str, Enum):
    FILE_EXISTS = "file_exists"
    FILE_CONTAINS = "file_contains"
    TEST_COMMAND = "test_command"


@dataclass(frozen=True, slots=True)
class GoalStageCheck:
    kind: GoalStageCheckKind
    path: str | None = None
    text: str | None = None
    command: str | None = None

    @classmethod
    def from_dict(cls, raw: object) -> GoalStageCheck:
        if not isinstance(raw, dict):
            raise TypeError("Goal stage check must be an object")
        kind = GoalStageCheckKind(_text(raw.get("kind"), "check kind", limit=32))
        fields = {
            GoalStageCheckKind.FILE_EXISTS: {"kind", "path"},
            GoalStageCheckKind.FILE_CONTAINS: {"kind", "path", "text"},
            GoalStageCheckKind.TEST_COMMAND: {"kind", "command"},
        }[kind]
        if set(raw) != fields:
            raise ValueError(f"Goal stage {kind.value} check requires: {', '.join(sorted(fields))}")
        if kind is GoalStageCheckKind.TEST_COMMAND:
            command = _text(raw["command"], "test command", limit=4_000)
            return cls(kind=kind, command=command)
        path = _text(raw["path"], "check path", limit=300).replace("\\", "/")
        parsed = Path(path)
        if parsed.is_absolute() or parsed.drive or any(part in {"", ".."} for part in parsed.parts):
            raise ValueError("Goal stage check path must stay inside the repository")
        return cls(kind=kind, path=path, text=_text(raw["text"], "expected text") if kind is GoalStageCheckKind.FILE_CONTAINS else None)

    def to_dict(self) -> dict[str, str]:
        if self.kind is GoalStageCheckKind.TEST_COMMAND:
            return {"kind": self.kind.value, "command": self.command or ""}
        result = {"kind": self.kind.value, "path": self.path or ""}
        if self.kind is GoalStageCheckKind.FILE_CONTAINS:
            result["text"] = self.text or ""
        return result

    def __str__(self) -> str:
        if self.kind is GoalStageCheckKind.TEST_COMMAND:
            return f"exact test command exits 0: {self.command}"
        if self.kind is GoalStageCheckKind.FILE_CONTAINS:
            return f"file {self.path} contains confirmed text"
        return f"file exists: {self.path}"


class GoalStageAssessmentStatus(str, Enum):
    VERIFIED = "verified"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class GoalStageAssessment:
    plan_version: int
    step_id: str
    checkpoint_id: str
    session_event_cursor: str
    status: GoalStageAssessmentStatus
    evidence_event_ids: tuple[str, ...]
    tool_call_ids: tuple[str, ...]
    detail: str
    next_action: str
    source: str = "evaluator"

    @classmethod
    def from_payload(cls, payload: object) -> GoalStageAssessment:
        if not isinstance(payload, dict) or set(payload) != {
            "plan_version", "step_id", "checkpoint_id", "session_event_cursor", "status",
            "evidence_event_ids", "tool_call_ids", "detail", "next_action", "source",
        }:
            raise ValueError("Goal stage assessment has invalid fields")
        version = payload["plan_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("Goal stage assessment plan_version must be positive")
        step_id = _text(payload["step_id"], "assessment step id", limit=64)
        if not _STEP_ID.fullmatch(step_id):
            raise ValueError("Goal stage assessment step id is invalid")
        evidence = payload["evidence_event_ids"]
        calls = payload["tool_call_ids"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 32:
            raise ValueError("Goal stage assessment requires 1–32 Session evidence IDs")
        if not isinstance(calls, list) or len(calls) > 32:
            raise ValueError("Goal stage assessment accepts at most 32 Tool Call IDs")
        evidence_ids = tuple(_text(item, "Session evidence ID", limit=100) for item in evidence)
        call_ids = tuple(_text(item, "Tool Call ID", limit=100) for item in calls)
        if len(set(evidence_ids)) != len(evidence_ids) or len(set(call_ids)) != len(call_ids):
            raise ValueError("Goal stage assessment evidence IDs must be unique")
        source = _text(payload["source"], "assessment source", limit=32)
        if source not in {"evaluator", "user_review"}:
            raise ValueError("Goal stage assessment source is invalid")
        return cls(
            plan_version=version,
            step_id=step_id,
            checkpoint_id=_text(payload["checkpoint_id"], "assessment checkpoint ID", limit=100),
            session_event_cursor=_text(payload["session_event_cursor"], "assessment Session cursor", limit=100),
            status=GoalStageAssessmentStatus(payload["status"]),
            evidence_event_ids=evidence_ids,
            tool_call_ids=call_ids,
            detail=_text(payload["detail"], "assessment detail", limit=2_000),
            next_action=_text(payload["next_action"], "assessment next action"),
            source=source,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "step_id": self.step_id,
            "checkpoint_id": self.checkpoint_id,
            "session_event_cursor": self.session_event_cursor,
            "status": self.status.value,
            "evidence_event_ids": list(self.evidence_event_ids),
            "tool_call_ids": list(self.tool_call_ids),
            "detail": self.detail,
            "next_action": self.next_action,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class GoalStep:
    step_id: str
    goal: str
    dependencies: tuple[str, ...]
    expected_output: str
    work_scope: tuple[str, ...]
    evidence: str
    check: str | GoalStageCheck
    next_action: str

    @classmethod
    def from_dict(cls, raw: object) -> GoalStep:
        if not isinstance(raw, dict):
            raise TypeError("Goal plan step must be an object")
        required = {"id", "goal", "dependencies", "expected_output", "work_scope", "evidence", "check", "next_action"}
        if set(raw) != required:
            raise ValueError(f"Goal plan step fields must be: {', '.join(sorted(required))}")
        step_id = _text(raw["id"], "step id", limit=64)
        if not _STEP_ID.fullmatch(step_id):
            raise ValueError("Goal plan step id must start with a letter and use only letters, digits, _ or -")
        dependencies = raw["dependencies"]
        if not isinstance(dependencies, list) or len(dependencies) > MAX_GOAL_STEPS:
            raise TypeError("Goal plan dependencies must be a bounded list")
        dependency_ids = tuple(_text(item, "dependency id", limit=64) for item in dependencies)
        if any(not _STEP_ID.fullmatch(item) for item in dependency_ids) or len(set(dependency_ids)) != len(dependency_ids):
            raise ValueError("Goal plan dependency ids must be valid and unique")
        scope = raw["work_scope"]
        if not isinstance(scope, list) or not 1 <= len(scope) <= 8:
            raise ValueError("Goal plan work_scope must contain 1–8 paths or resources")
        safe_scope = tuple(_text(item, "work scope item", limit=200) for item in scope)
        if len(set(safe_scope)) != len(safe_scope):
            raise ValueError("Goal plan work_scope items must be unique")
        return cls(
            step_id=step_id,
            goal=_text(raw["goal"], "stage goal"),
            dependencies=dependency_ids,
            expected_output=_text(raw["expected_output"], "expected output"),
            work_scope=safe_scope,
            evidence=_text(raw["evidence"], "evidence"),
            check=GoalStageCheck.from_dict(raw["check"]) if isinstance(raw["check"], dict) else _text(raw["check"], "check"),
            next_action=_text(raw["next_action"], "next action"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "goal": self.goal,
            "dependencies": list(self.dependencies),
            "expected_output": self.expected_output,
            "work_scope": list(self.work_scope),
            "evidence": self.evidence,
            "check": self.check.to_dict() if isinstance(self.check, GoalStageCheck) else self.check,
            "next_action": self.next_action,
        }


@dataclass(frozen=True, slots=True)
class GoalPlan:
    version: int
    reason: str
    steps: tuple[GoalStep, ...]

    @classmethod
    def from_payload(cls, payload: object) -> GoalPlan:
        if not isinstance(payload, dict):
            raise TypeError("Goal plan payload must be an object")
        if set(payload) != {"version", "reason", "steps"}:
            raise ValueError("Goal plan requires version, reason and steps")
        version = payload["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("Goal plan version must be a positive integer")
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_GOAL_STEPS:
            raise ValueError(f"Goal plan must contain 1–{MAX_GOAL_STEPS} steps")
        steps = tuple(GoalStep.from_dict(item) for item in raw_steps)
        by_id = {step.step_id: step for step in steps}
        if len(by_id) != len(steps):
            raise ValueError("Goal plan step ids must be unique")
        for step in steps:
            missing = set(step.dependencies) - set(by_id)
            if missing:
                raise ValueError(f"Goal plan step {step.step_id} has missing dependencies: {', '.join(sorted(missing))}")
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("Goal plan dependencies must not contain a cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in by_id[step_id].dependencies:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step in steps:
            visit(step.step_id)
        return cls(version=version, reason=_text(payload["reason"], "revision reason"), steps=steps)

    def to_payload(self) -> dict[str, Any]:
        return {"version": self.version, "reason": self.reason, "steps": [step.to_dict() for step in self.steps]}

    def step(self, step_id: str) -> GoalStep | None:
        return next((step for step in self.steps if step.step_id == step_id), None)
