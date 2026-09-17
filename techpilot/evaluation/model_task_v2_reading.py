"""Behavioral read-only cards used to restore code-reading coverage in M0 v2."""

from __future__ import annotations

from .model_tasks import (
    BehaviorCheck,
    ModelTaskAcceptanceLevel,
    ModelTaskCard,
    ModelTaskFamily,
    ModelTaskKind,
    ModelTaskNegativeExample,
    ModelTaskQualityEvidence,
    ModelTaskReviewState,
    ModelTaskSource,
    ModelTaskSplit,
    require_valid_model_task_deck,
)

MODEL_CODING_V2_READING_COVERAGE_SUITE = "model-coding-v2-reading-coverage"


def build_model_coding_v2_reading_coverage_cards() -> tuple[ModelTaskCard, ...]:
    """Two development, three frozen Holdout, and one regression read-only cards."""

    return require_valid_model_task_deck((
        _read("coding-v2r-dev-level-149", "levels.py", "level_text", "value: int", "return f'level={value}'", (BehaviorCheck("one", "levels.py", "level_text", (1,), "level=1"), BehaviorCheck("two", "levels.py", "level_text", (2,), "level=2")), "What does level_text return for 1?", "level=1", ModelTaskSplit.DEVELOPMENT, "manual read contract: derive a visible function result without modifying its source"),
        _read("coding-v2r-dev-separator-150", "export.py", "separator", "value: str", "return f'item\\t{value}'", (BehaviorCheck("value", "export.py", "separator", ("a",), "item\ta"), BehaviorCheck("empty", "export.py", "separator", ("",), "item\t")), "What separator appears between item and its value?", "tab", ModelTaskSplit.DEVELOPMENT, "manual read contract: identify the separator in a public formatter"),
        _read("coding-v2r-holdout-limit-151", "limits.py", "limit_text", "value: int", "return f'limit={value}'", (BehaviorCheck("ten", "limits.py", "limit_text", (10,), "limit=10"), BehaviorCheck("zero", "limits.py", "limit_text", (0,), "limit=0")), "What does limit_text return for 10?", "limit=10", ModelTaskSplit.HOLDOUT, "frozen read contract: recover a visible function result from source"),
        _read("coding-v2r-holdout-prefix-152", "prefix.py", "render", "value: str", "return f'Alert: {value}'", (BehaviorCheck("value", "prefix.py", "render", ("x",), "Alert: x"), BehaviorCheck("empty", "prefix.py", "render", ("",), "Alert: ")), "What prefix does render add before its value?", "alert", ModelTaskSplit.HOLDOUT, "frozen read contract: identify a public output prefix"),
        _read("coding-v2r-holdout-region-153", "region.py", "region_label", "value: str", "return f'region={value}'", (BehaviorCheck("cn", "region.py", "region_label", ("cn-shenzhen",), "region=cn-shenzhen"), BehaviorCheck("us", "region.py", "region_label", ("us-east",), "region=us-east")), "What does region_label return for cn-shenzhen?", "region=cn-shenzhen", ModelTaskSplit.HOLDOUT, "frozen read contract: derive a configured region presentation"),
        _read("coding-v2r-regression-retention-154", "retention.py", "days_text", "value: int", "return f'days={value}'", (BehaviorCheck("thirty", "retention.py", "days_text", (30,), "days=30"), BehaviorCheck("seven", "retention.py", "days_text", (7,), "days=7")), "What does days_text return for 30?", "days=30", ModelTaskSplit.REGRESSION, "observed regression surrogate: a code-reading answer omitted the unit-bearing visible result"),
    ))


def _read(card_id: str, filename: str, function: str, parameters: str, body: str, checks: tuple[BehaviorCheck, ...], question: str, fact: str, split: ModelTaskSplit, source_reference: str) -> ModelTaskCard:
    source = f"def {function}({parameters}):\n    {body}\n"
    return ModelTaskCard(
        id=card_id, suite=MODEL_CODING_V2_READING_COVERAGE_SUITE, kind=ModelTaskKind.READ_ONLY,
        prompts=(f"Read {filename}. {question} Do not modify files.",), initial_files={filename: source},
        required_response_facts=(fact,), family=ModelTaskFamily.CODE_READING,
        source=ModelTaskSource.MANUAL_CONTRACT if split is not ModelTaskSplit.REGRESSION else ModelTaskSource.OBSERVED_REGRESSION,
        source_reference=source_reference, template_id="v2-read-function-result", split=split,
        acceptance_level=ModelTaskAcceptanceLevel.BEHAVIORAL, behavior_checks=checks,
        quality=ModelTaskQualityEvidence(reference_response=fact, negative_examples=(ModelTaskNegativeExample("semantic-wrong-answer", response="unknown", rationale="A plausible response that does not state the requested visible fact."),), reviewed_by="techpilot-maintainer", review_note="Read-only behavior and answer fact are independently checked without workspace mutation.", review_state=ModelTaskReviewState.FROZEN if split is ModelTaskSplit.HOLDOUT else ModelTaskReviewState.REVIEWED),
    )
