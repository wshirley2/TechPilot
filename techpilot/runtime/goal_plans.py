"""Bounded, user-authored stage contracts for durable Goals.

These contracts describe work. They never grant Tool permissions or prove that
an outcome was completed; execution evidence remains in Runtime and Session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class GoalStep:
    step_id: str
    goal: str
    dependencies: tuple[str, ...]
    expected_output: str
    work_scope: tuple[str, ...]
    evidence: str
    check: str
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
            check=_text(raw["check"], "check"),
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
            "check": self.check,
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
