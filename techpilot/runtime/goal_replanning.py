"""Read-only model proposal for revising a durable Goal stage plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect
from techpilot.engine.tool_results import ToolResult, ToolStatus
from techpilot.engine.tools.base import Tool

from .goal_plans import GoalPlan, GoalStageCheck

_TEXT = {"type": "string", "minLength": 1}
_CHECK = {
    "oneOf": [
        {
            "type": "object", "properties": {"kind": {"enum": ["file_exists"]}, "path": _TEXT},
            "required": ["kind", "path"], "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"kind": {"enum": ["file_contains"]}, "path": _TEXT, "text": _TEXT},
            "required": ["kind", "path", "text"], "additionalProperties": False,
        },
        {
            "type": "object", "properties": {"kind": {"enum": ["test_command"]}, "command": _TEXT},
            "required": ["kind", "command"], "additionalProperties": False,
        },
    ],
}
_STEP = {
    "type": "object",
    "properties": {
        "id": _TEXT, "goal": _TEXT, "dependencies": {"type": "array", "items": _TEXT},
        "expected_output": _TEXT, "work_scope": {"type": "array", "minItems": 1, "items": _TEXT},
        "evidence": _TEXT, "check": _CHECK, "next_action": _TEXT,
    },
    "required": [
        "id", "goal", "dependencies", "expected_output", "work_scope",
        "evidence", "check", "next_action",
    ],
    "additionalProperties": False,
}
_PLAN = {
    "oneOf": [
        {"type": "null"},
        {
            "type": "object",
            "properties": {
                "reason": _TEXT,
                "steps": {"type": "array", "minItems": 2, "maxItems": 7, "items": _STEP},
            },
            "required": ["reason", "steps"], "additionalProperties": False,
        },
    ],
}


@dataclass(frozen=True, slots=True)
class GoalReplanDecision:
    action: str
    summary: str
    plan: dict[str, Any] | None


class GoalReplanTool(Tool):
    """Collect one proposal; only the Host may persist an accepted revision."""

    name = "propose_goal_replan"
    description = (
        "After inspecting Goal facts, propose exactly one planning outcome: "
        "replace with a complete structured stage plan, keep the current plan, "
        "or unavailable with a concrete missing fact. This never changes Goal facts or permissions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["replace", "keep", "unavailable"]},
            "summary": {"type": "string", "minLength": 1},
            "plan": _PLAN,
        },
        "required": ["action", "summary", "plan"],
        "additionalProperties": False,
    }
    execution_effect = ToolEffect.READ
    execution_concurrency = ToolConcurrency.EXCLUSIVE

    def __init__(self, submit: Callable[[GoalReplanDecision], None]) -> None:
        self._submit = submit

    def execute(self, action: str, summary: str, plan: dict[str, Any] | None) -> ToolResult:
        try:
            decision = self.validate(action, summary, plan)
            self._submit(decision)
        except (TypeError, ValueError) as error:
            return ToolResult(f"Error: invalid Goal replan proposal: {error}", ToolStatus.ERROR)
        return ToolResult("Goal planning outcome proposed for Host validation.", ToolStatus.COMPLETED)

    @staticmethod
    def validate(action: str, summary: str, plan: dict[str, Any] | None) -> GoalReplanDecision:
        if action not in {"replace", "keep", "unavailable"}:
            raise ValueError("unknown planning action")
        if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 500:
            raise ValueError("summary must contain 1–500 characters")
        if action == "replace":
            if not isinstance(plan, dict) or set(plan) != {"reason", "steps"}:
                raise ValueError("replacement requires reason and steps")
            checked = GoalPlan.from_payload({**plan, "version": 1})
            if not 2 <= len(checked.steps) <= 7:
                raise ValueError("a model-proposed plan requires 2–7 stages")
            if any(not isinstance(step.check, GoalStageCheck) for step in checked.steps):
                raise ValueError("every proposed stage needs a structured Runtime check")
        elif plan is not None:
            raise ValueError("keep and unavailable must use plan=null")
        return GoalReplanDecision(action, summary.strip(), plan)
