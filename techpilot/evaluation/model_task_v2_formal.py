"""The assembled, QA-gated 100-card M0 v2 Coding evaluation deck."""

from __future__ import annotations

from dataclasses import replace

from .model_task_batch_03 import build_model_coding_v1_working_100_cards
from .model_task_v2_batch_01 import build_model_coding_v2_batch_01_cards
from .model_task_v2_batch_02 import build_model_coding_v2_batch_02_cards
from .model_task_v2_reading import build_model_coding_v2_reading_coverage_cards
from .model_tasks import (
    MODEL_CODING_V1_FORMAL_CANDIDATE_IDS,
    ModelTaskCard,
    ModelTaskSplit,
    require_valid_model_task_deck,
)

MODEL_CODING_V2_FORMAL_100_SUITE = "model-coding-v2-formal-100"


def build_model_coding_v2_formal_100_cards() -> tuple[ModelTaskCard, ...]:
    """Return the frozen-shape M0 deck: 60 development / 25 Holdout / 15 regression."""

    v1_candidates = tuple(
        card
        for card in build_model_coding_v1_working_100_cards()
        if card.id in MODEL_CODING_V1_FORMAL_CANDIDATE_IDS
    )
    replaced = {
        "coding-v2b1-dev-mode-default-112", "coding-v2b1-holdout-label-118",
        "coding-v2b2-dev-code-upper-121", "coding-v2b2-holdout-name-136",
        "coding-v2b2-holdout-active-137", "coding-v2b2-regression-limit-146",
    }
    expansion_cards = tuple(
        card
        for card in build_model_coding_v2_batch_01_cards() + build_model_coding_v2_batch_02_cards()
        if card.id not in replaced
    )
    cards = v1_candidates + expansion_cards + build_model_coding_v2_reading_coverage_cards()
    if len(cards) != 100:
        raise ValueError(f"formal M0 v2 requires exactly 100 cards, got {len(cards)}")
    expected_splits = {
        ModelTaskSplit.DEVELOPMENT: 60,
        ModelTaskSplit.HOLDOUT: 25,
        ModelTaskSplit.REGRESSION: 15,
    }
    actual_splits = {split: sum(card.split is split for card in cards) for split in expected_splits}
    if actual_splits != expected_splits:
        raise ValueError(f"formal M0 v2 split mismatch: {actual_splits}")
    return require_valid_model_task_deck(tuple(replace(card, suite=MODEL_CODING_V2_FORMAL_100_SUITE) for card in cards))
