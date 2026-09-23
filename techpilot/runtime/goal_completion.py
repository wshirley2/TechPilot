"""Goal-scoped completion proposal Tool.

The Tool is intentionally ephemeral: ``LongTaskWorkflow`` exposes it only
while a Goal step is active.  It records an in-memory proposal; the Workflow
later verifies that proposal against durable Session events before appending a
Task completion claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect
from techpilot.engine.tool_results import ToolResult, ToolStatus
from techpilot.engine.tools.base import Tool

GOAL_COMPLETION_TOOL_NAME = "submit_goal_completion"


@dataclass(frozen=True, slots=True)
class GoalCompletionProposal:
    summary: str
    evidence_tool_call_ids: tuple[str, ...]


class GoalCompletionTool(Tool):
    """Let a model propose completion without granting it a terminal transition."""

    name = GOAL_COMPLETION_TOOL_NAME
    description = (
        "Propose that the current Goal is ready for user confirmation. "
        "Reference completed Tool Call IDs from this Goal step. This does not complete the Goal."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "What the cited evidence verifies."},
            "evidence_tool_call_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Completed Tool Call IDs from this Goal step.",
            },
        },
        "required": ["summary", "evidence_tool_call_ids"],
        "additionalProperties": False,
    }
    # The proposal changes only in-memory Workflow state. It must not be
    # treated as an external effect, but it stays an exclusive barrier so a
    # model cannot mix its declaration into a concurrent read wave.
    execution_effect = ToolEffect.READ
    execution_concurrency = ToolConcurrency.EXCLUSIVE

    def __init__(self, submit: Callable[[GoalCompletionProposal], None]) -> None:
        self._submit = submit

    def execute(self, summary: str, evidence_tool_call_ids: list[str]) -> str:
        if not isinstance(summary, str) or not summary.strip():
            return ToolResult("Error: completion summary cannot be empty", ToolStatus.ERROR)
        if (
            not isinstance(evidence_tool_call_ids, list)
            or not evidence_tool_call_ids
            or not all(isinstance(call_id, str) and call_id.strip() for call_id in evidence_tool_call_ids)
        ):
            return ToolResult("Error: completion proposal requires Tool Call evidence", ToolStatus.ERROR)
        try:
            self._submit(GoalCompletionProposal(
                summary=summary.strip(),
                evidence_tool_call_ids=tuple(call_id.strip() for call_id in evidence_tool_call_ids),
            ))
        except ValueError as error:
            return ToolResult(f"Error: completion proposal rejected: {error}", ToolStatus.ERROR)
        return ToolResult(
            "Completion proposal received. Runtime will verify the cited Session evidence after this Goal step; "
            "the Goal remains running until the user confirms it.",
            ToolStatus.COMPLETED,
        )
