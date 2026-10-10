"""Ephemeral, read-only proposal for a Goal's acceptance contract.

The proposal is not a Task fact. Chat shows it to the user and only records
the contract after explicit confirmation in ordinary conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect
from techpilot.engine.tool_results import ToolResult, ToolStatus
from techpilot.engine.tools.base import Tool

from .goal_plans import GoalPlan, GoalStageCheck
from .long_tasks import CompletionCriterionKind

GOAL_SETUP_TOOL_NAME = "propose_goal_setup"


@dataclass(frozen=True, slots=True)
class GoalSetupProposal:
    summary: str
    constraints: tuple[str, ...]
    criteria: tuple[dict[str, str], ...]
    plan: dict[str, Any] | None


class GoalSetupTool(Tool):
    """Collect a proposal without changing durable Goal state."""

    name = GOAL_SETUP_TOOL_NAME
    description = (
        "Propose a verifiable Goal contract for user confirmation. "
        "This does not authorize work, record constraints, or start the Goal."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": [kind.value for kind in CompletionCriterionKind]},
                        "path": {"type": "string"},
                        "expected_text": {"type": "string"},
                        "command": {"type": "string"},
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                },
            },
            "plan": {"type": ["object", "null"]},
        },
        "required": ["summary", "constraints", "criteria", "plan"],
        "additionalProperties": False,
    }
    execution_effect = ToolEffect.READ
    execution_concurrency = ToolConcurrency.EXCLUSIVE

    def __init__(self, submit: Callable[[GoalSetupProposal], None]) -> None:
        self._submit = submit

    def execute(
        self,
        summary: str,
        constraints: list[str],
        criteria: list[dict[str, str]],
        plan: dict[str, Any] | None,
    ) -> ToolResult:
        try:
            proposal = self._validate(summary, constraints, criteria, plan)
            self._submit(proposal)
        except (TypeError, ValueError) as error:
            return ToolResult(f"Error: invalid Goal setup proposal: {error}", ToolStatus.ERROR)
        return ToolResult("Goal setup proposed; awaiting user confirmation before execution.", ToolStatus.COMPLETED)

    @staticmethod
    def _validate(
        summary: str,
        constraints: list[str],
        criteria: list[dict[str, str]],
        plan: dict[str, Any] | None,
    ) -> GoalSetupProposal:
        if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 500:
            raise ValueError("summary must contain 1–500 characters")
        if not isinstance(constraints, list) or len(constraints) > 12:
            raise ValueError("at most 12 constraints are allowed")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 2_000 for item in constraints):
            raise ValueError("constraints must be nonempty text")
        if not isinstance(criteria, list) or not 1 <= len(criteria) <= 12:
            raise ValueError("1–12 completion criteria are required")
        validated: list[dict[str, str]] = []
        for criterion in criteria:
            if not isinstance(criterion, dict) or "kind" not in criterion:
                raise ValueError("each criterion needs a kind")
            kind = CompletionCriterionKind(criterion["kind"])
            required = {
                CompletionCriterionKind.FILE_EXISTS: {"kind", "path"},
                CompletionCriterionKind.FILE_CONTAINS: {"kind", "path", "expected_text"},
                CompletionCriterionKind.TEST_COMMAND: {"kind", "command"},
            }[kind]
            if set(criterion) != required or any(not isinstance(value, str) or not value.strip() for value in criterion.values()):
                raise ValueError(f"{kind.value} has missing or extra fields")
            GoalStageCheck.from_dict({
                "kind": kind.value,
                **({"command": criterion["command"]} if kind is CompletionCriterionKind.TEST_COMMAND else {
                    "path": criterion["path"],
                    **({"text": criterion["expected_text"]} if kind is CompletionCriterionKind.FILE_CONTAINS else {}),
                }),
            })
            validated.append(dict(criterion))
        if plan is not None:
            if not isinstance(plan, dict):
                raise ValueError("plan must be an object or null")
            checked_plan = GoalPlan.from_payload({**plan, "version": 1})
            if any(not isinstance(step.check, GoalStageCheck) for step in checked_plan.steps):
                raise ValueError("every proposed stage needs a structured Runtime check")
        return GoalSetupProposal(
            summary=summary.strip(),
            constraints=tuple(item.strip() for item in constraints),
            criteria=tuple(validated),
            plan=plan,
        )
