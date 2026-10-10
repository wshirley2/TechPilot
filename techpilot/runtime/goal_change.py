"""Read-only classification of a user's change to a running Goal contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect
from techpilot.engine.tool_results import ToolResult, ToolStatus
from techpilot.engine.tools.base import Tool

from .goal_plans import GoalPlan, GoalStageCheck
from .goal_setup import GoalSetupTool

GOAL_CHANGE_TOOL_NAME = "propose_goal_change"
_KINDS = frozenset({
    "continue", "clarify", "constraint_add", "constraint_remove",
    "criterion_add", "criterion_remove", "plan_replace",
})


@dataclass(frozen=True, slots=True)
class GoalChangeDecision:
    kind: str
    summary: str
    data: dict[str, Any]


class GoalChangeTool(Tool):
    """Classify a human turn before any Goal work tools become available."""

    name = GOAL_CHANGE_TOOL_NAME
    description = (
        "Classify a user message for a running Goal. Use continue for ordinary work, "
        "clarify if intent is ambiguous, or propose exactly one contract change. "
        "This tool never executes work or changes the Goal; changes need user confirmation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": sorted(_KINDS)},
            "summary": {"type": "string"},
            "data": {"type": "object"},
        },
        "required": ["kind", "summary", "data"],
        "additionalProperties": False,
    }
    execution_effect = ToolEffect.READ
    execution_concurrency = ToolConcurrency.EXCLUSIVE

    def __init__(self, submit: Callable[[GoalChangeDecision], None]) -> None:
        self._submit = submit

    def execute(self, kind: str, summary: str, data: dict[str, Any]) -> ToolResult:
        try:
            decision = self.validate(kind, summary, data)
            self._submit(decision)
        except (TypeError, ValueError) as error:
            return ToolResult(f"Error: invalid Goal change proposal: {error}", ToolStatus.ERROR)
        return ToolResult("Goal input classified; awaiting Host decision.", ToolStatus.COMPLETED)

    @staticmethod
    def validate(kind: str, summary: str, data: dict[str, Any]) -> GoalChangeDecision:
        if kind not in _KINDS:
            raise ValueError("unknown Goal change kind")
        if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 500:
            raise ValueError("summary must contain 1–500 characters")
        if not isinstance(data, dict):
            raise TypeError("data must be an object")
        expected = {
            "continue": set(), "clarify": set(),
            "constraint_add": {"text"}, "constraint_remove": {"constraint_id"},
            "criterion_add": {"criterion"}, "criterion_remove": {"criterion_id"},
            "plan_replace": {"plan"},
        }[kind]
        if set(data) != expected:
            raise ValueError(f"{kind} requires exactly {sorted(expected)}")
        if kind in {"constraint_add", "constraint_remove", "criterion_remove"}:
            key = next(iter(expected))
            if not isinstance(data[key], str) or not data[key].strip() or len(data[key]) > 2_000:
                raise ValueError(f"{key} must be nonempty text")
        if kind == "criterion_add":
            validated = GoalSetupTool._validate("Goal change", [], [data["criterion"]], None)
            data = {"criterion": validated.criteria[0]}
        if kind == "plan_replace":
            plan = data["plan"]
            if not isinstance(plan, dict) or set(plan) != {"reason", "steps"}:
                raise ValueError("plan requires reason and steps")
            checked = GoalPlan.from_payload({**plan, "version": 1})
            if any(not isinstance(step.check, GoalStageCheck) for step in checked.steps):
                raise ValueError("every replacement stage needs a structured Runtime check")
        return GoalChangeDecision(kind=kind, summary=summary.strip(), data=dict(data))
