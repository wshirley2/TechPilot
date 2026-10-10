"""Contract tests for the read-only running-Goal input classifier."""

from __future__ import annotations

import pytest

from techpilot.engine.tool_results import ToolStatus
from techpilot.runtime.goal_change import GoalChangeDecision, GoalChangeTool


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        ("continue", {}),
        ("clarify", {}),
        ("constraint_add", {"text": "Do not use network"}),
        ("constraint_remove", {"constraint_id": "constraint-1"}),
        ("criterion_add", {"criterion": {"kind": "file_exists", "path": "result.md"}}),
        ("criterion_remove", {"criterion_id": "criterion-1"}),
    ],
)
def test_goal_change_tool_accepts_only_the_documented_data_shape(kind, data):
    submitted: list[GoalChangeDecision] = []

    result = GoalChangeTool(submitted.append).execute(kind, "A valid change", data)

    assert result.status is ToolStatus.COMPLETED
    assert [(decision.kind, decision.data) for decision in submitted] == [(kind, data)]


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        ("continue", {"requested_action": "continue"}),
        ("clarify", {"question": "Which file?"}),
        ("constraint_add", {"rule": "Do not use network", "scope": "task"}),
        ("constraint_remove", {"id": "constraint-1", "reason": "obsolete"}),
        ("criterion_add", {"description": "result exists", "id": "criterion-1"}),
        ("criterion_remove", {"id": "criterion-1"}),
    ],
)
def test_goal_change_tool_rejects_plausible_but_undocumented_model_fields(kind, data):
    submitted: list[GoalChangeDecision] = []

    result = GoalChangeTool(submitted.append).execute(kind, "A malformed change", data)

    assert result.status is ToolStatus.ERROR
    assert submitted == []


def test_goal_change_schema_binds_every_kind_to_one_exact_data_variant():
    variants = GoalChangeTool.parameters["oneOf"]

    assert {variant["properties"]["kind"]["enum"][0] for variant in variants} == {
        "continue", "clarify", "constraint_add", "constraint_remove",
        "criterion_add", "criterion_remove", "plan_replace",
    }
    assert variants[0]["properties"]["data"] == {
        "type": "object", "properties": {}, "additionalProperties": False,
    }
    assert variants[2]["properties"]["data"]["required"] == ["text"]
    assert variants[4]["properties"]["data"]["required"] == ["criterion"]
