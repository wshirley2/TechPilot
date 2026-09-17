from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.runtime_control import RuntimeLimits
from techpilot.evaluation import (
    MODEL_CODING_DEV_V0_SUITE,
    MODEL_CODING_TAX_REPAIR_V1_SUITE,
    MODEL_CODING_V1_BATCH_01_SUITE,
    MODEL_CODING_V1_BATCH_02_SUITE,
    MODEL_CODING_V1_SEED_QA_V1_SUITE,
    MODEL_CODING_V1_SEED_SUITE,
    MODEL_CODING_V1_WORKING_40_SUITE,
    MODEL_CODING_V1_WORKING_70_SUITE,
    MODEL_CODING_V2_BATCH_01_SUITE,
    MODEL_CODING_V2_BATCH_02_SUITE,
    MODEL_CODING_V2_FORMAL_100_SUITE,
    MODEL_CODING_V2_READING_COVERAGE_SUITE,
    ModelAttemptOutcome,
    ModelAttemptStatus,
    ModelEvaluationManifest,
    ModelEvaluationReport,
    ModelEvaluationRunner,
    ModelTaskAcceptanceLevel,
    ModelTaskCard,
    ModelTaskFamily,
    ModelTaskKind,
    ModelTaskNegativeExample,
    ModelTaskReviewState,
    ModelTaskSource,
    ModelTaskSplit,
    ModelTaskStaticDisposition,
    build_model_coding_dev_v0_cards,
    build_model_coding_tax_repair_v1_cards,
    build_model_coding_v1_batch_01_cards,
    build_model_coding_v1_batch_02_cards,
    build_model_coding_v1_seed_cards,
    build_model_coding_v1_seed_qa_v1_cards,
    build_model_coding_v1_working_40_cards,
    build_model_coding_v1_working_70_cards,
    build_model_coding_v1_working_100_cards,
    build_model_coding_v2_batch_01_cards,
    build_model_coding_v2_batch_02_cards,
    build_model_coding_v2_formal_100_cards,
    build_model_coding_v2_reading_coverage_cards,
    model_task_set_digest,
    select_model_task_cards,
    validate_model_task_deck,
)
from techpilot.evaluation.__main__ import _per_attempt_token_limit, _select_model_task_ids
from techpilot.evaluation.__main__ import main as evaluation_main
from techpilot.evaluation.model_tasks import MODEL_CODING_V1_FORMAL_CANDIDATE_IDS
from techpilot.runtime import RuntimeBootstrap


class FixedProvider:
    model = "fixed-model-evaluation"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = iter(responses)

    def chat(self, messages, tools=None, on_token=None):
        del messages, tools
        response = next(self.responses)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        if on_token and response.content:
            on_token(response.content)
        return response


def _runner(responses: list[LLMResponse]) -> ModelEvaluationRunner:
    return ModelEvaluationRunner(RuntimeBootstrap(provider_factory=lambda _config: FixedProvider(responses)))


def _manifest(cards: tuple[ModelTaskCard, ...]) -> ModelEvaluationManifest:
    return ModelEvaluationManifest(
        provider="fixed-provider",
        model="fixed-model-evaluation",
        parameters={"temperature": 0},
        suite=MODEL_CODING_DEV_V0_SUITE,
        case_set_digest=model_task_set_digest(cards),
    )


def test_development_cards_define_read_only_scoped_patch_and_cross_turn_constraints() -> None:
    cards = build_model_coding_dev_v0_cards()

    assert len(cards) == 3
    assert [card.kind for card in cards] == [ModelTaskKind.READ_ONLY, ModelTaskKind.PATCH, ModelTaskKind.PATCH]
    assert len(cards[-1].prompts) == 2
    assert len(model_task_set_digest(cards)) == 64
    assert [card.family for card in cards] == [
        ModelTaskFamily.CODE_READING,
        ModelTaskFamily.SCOPED_REPAIR,
        ModelTaskFamily.MULTI_TURN_CONSTRAINT,
    ]
    assert all(card.source_reference for card in cards)
    assert all(card.split is ModelTaskSplit.DEVELOPMENT for card in cards)


def test_tax_repair_v1_makes_the_previously_hidden_formula_visible_to_the_model() -> None:
    cards = build_model_coding_tax_repair_v1_cards()

    assert len(cards) == 1
    assert cards[0].suite == MODEL_CODING_TAX_REPAIR_V1_SUITE
    assert "total(10) must return 11" in cards[0].prompts[0]
    assert cards[0].required_file_contents == {"tax.py": "return amount + 1"}


def test_v1_seed_cards_have_auditable_origins_and_a_declared_dev_holdout_regression_split() -> None:
    cards = build_model_coding_v1_seed_cards()

    assert len(cards) == 20
    assert {card.suite for card in cards} == {MODEL_CODING_V1_SEED_SUITE}
    assert len({card.id for card in cards}) == len(cards)
    assert len(select_model_task_cards(cards, ModelTaskSplit.DEVELOPMENT)) == 12
    assert len(select_model_task_cards(cards, ModelTaskSplit.HOLDOUT)) == 5
    assert len(select_model_task_cards(cards, ModelTaskSplit.REGRESSION)) == 3
    assert {card.family for card in cards} == {
        ModelTaskFamily.CODE_READING,
        ModelTaskFamily.SCOPED_REPAIR,
        ModelTaskFamily.MULTI_TURN_CONSTRAINT,
        ModelTaskFamily.PERMISSION_BOUNDARY,
    }
    assert all(card.template_id != "legacy" and card.source_reference for card in cards)
    assert {card.source for card in cards} >= {
        ModelTaskSource.MANUAL_CONTRACT,
        ModelTaskSource.OBSERVED_REGRESSION,
        ModelTaskSource.FAULT_INJECTION,
        ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
    }


def test_v1_seed_cards_pass_the_deterministic_quality_gate_with_frozen_holdout_evidence() -> None:
    cards = build_model_coding_v1_seed_qa_v1_cards()

    report = validate_model_task_deck(cards)

    assert report.passed
    assert report.issues == ()
    assert {card.suite for card in cards} == {MODEL_CODING_V1_SEED_QA_V1_SUITE}
    assert all(card.quality is not None for card in cards)
    assert all(
        card.quality.review_state is ModelTaskReviewState.FROZEN
        for card in cards
        if card.split is ModelTaskSplit.HOLDOUT
    )
    rendered = report.to_dict()
    assert rendered["case_count"] == 20
    assert rendered["split_counts"] == {"development": 12, "holdout": 5, "regression": 3}
    assert rendered["issues"] == []


def test_qa_seed_migrates_only_supported_development_cards_to_behavioral_acceptance() -> None:
    cards = build_model_coding_v1_seed_qa_v1_cards()
    behavioral = [card for card in cards if card.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL]

    assert {card.id for card in behavioral} == {
        "coding-v1-repair-clamp-006",
        "coding-v1-repair-retry-default-007",
        "coding-v1-repair-role-case-008",
        "coding-v1-repair-boundary-009",
        "coding-v1-repair-normalize-011",
        "coding-v1-repair-lookup-012",
    }
    assert all(len(card.behavior_checks) >= 2 for card in behavioral)
    assert all(card.acceptance_level is ModelTaskAcceptanceLevel.STATIC_REGRESSION for card in cards if card not in behavioral)
    assert validate_model_task_deck(cards).to_dict()["acceptance_level_counts"] == {
        "behavioral": 6,
        "static-regression": 14,
    }


def test_quality_gate_rejects_missing_holdout_freeze_and_a_negative_that_the_oracle_accepts() -> None:
    cards = build_model_coding_v1_seed_qa_v1_cards()
    holdout = next(card for card in cards if card.split is ModelTaskSplit.HOLDOUT)
    assert holdout.quality is not None
    unfrozen = replace(
        holdout,
        quality=replace(holdout.quality, review_state=ModelTaskReviewState.REVIEWED),
    )
    assert "holdout-not-frozen" in {issue.code for issue in validate_model_task_deck((unfrozen,)).issues}

    patch = next(card for card in cards if card.id == "coding-v1-repair-clamp-006")
    assert patch.quality is not None
    accepted_negative = replace(
        patch,
        quality=replace(
            patch.quality,
            negative_examples=(
                ModelTaskNegativeExample(
                    id="accepted-wrongly",
                    files=patch.quality.reference_files,
                    rationale="Regression probe: an oracle must not accept a declared negative.",
                ),
            ),
        ),
    )
    codes = {issue.code for issue in validate_model_task_deck((accepted_negative,)).issues}
    assert "negative-accepted:accepted-wrongly" in codes


def test_first_expansion_batch_is_qa_gated_unique_and_split_for_deck_growth() -> None:
    cards = build_model_coding_v1_batch_01_cards()
    report = validate_model_task_deck(cards)

    assert len(cards) == 20
    assert {card.suite for card in cards} == {MODEL_CODING_V1_BATCH_01_SUITE}
    assert len({card.id for card in cards}) == len(cards)
    assert len(select_model_task_cards(cards, ModelTaskSplit.DEVELOPMENT)) == 14
    assert len(select_model_task_cards(cards, ModelTaskSplit.HOLDOUT)) == 4
    assert len(select_model_task_cards(cards, ModelTaskSplit.REGRESSION)) == 2
    assert report.passed
    assert {card.family for card in cards} == {
        ModelTaskFamily.CODE_READING,
        ModelTaskFamily.SCOPED_REPAIR,
        ModelTaskFamily.MULTI_TURN_CONSTRAINT,
        ModelTaskFamily.PERMISSION_BOUNDARY,
        ModelTaskFamily.CONTEXT_CONTINUITY,
    }
    assert all(
        card.quality is not None and card.quality.review_state is ModelTaskReviewState.FROZEN
        for card in cards
        if card.split is ModelTaskSplit.HOLDOUT
    )


def test_working_40_deck_combines_qa_versions_under_one_runnable_suite() -> None:
    cards = build_model_coding_v1_working_40_cards()

    assert len(cards) == 40
    assert {card.suite for card in cards} == {MODEL_CODING_V1_WORKING_40_SUITE}
    assert len({card.id for card in cards}) == 40
    assert len(select_model_task_cards(cards, ModelTaskSplit.DEVELOPMENT)) == 26
    assert len(select_model_task_cards(cards, ModelTaskSplit.HOLDOUT)) == 9
    assert len(select_model_task_cards(cards, ModelTaskSplit.REGRESSION)) == 5
    assert validate_model_task_deck(cards).passed


def test_second_expansion_batch_and_working_70_deck_keep_qa_and_split_invariants() -> None:
    batch = build_model_coding_v1_batch_02_cards()
    working = build_model_coding_v1_working_70_cards()

    assert len(batch) == 30
    assert {card.suite for card in batch} == {MODEL_CODING_V1_BATCH_02_SUITE}
    assert len(select_model_task_cards(batch, ModelTaskSplit.DEVELOPMENT)) == 17
    assert len(select_model_task_cards(batch, ModelTaskSplit.HOLDOUT)) == 8
    assert len(select_model_task_cards(batch, ModelTaskSplit.REGRESSION)) == 5
    assert validate_model_task_deck(batch).passed
    assert len(working) == 70
    assert {card.suite for card in working} == {MODEL_CODING_V1_WORKING_70_SUITE}
    assert len({card.id for card in working}) == 70
    assert len(select_model_task_cards(working, ModelTaskSplit.DEVELOPMENT)) == 43
    assert len(select_model_task_cards(working, ModelTaskSplit.HOLDOUT)) == 17
    assert len(select_model_task_cards(working, ModelTaskSplit.REGRESSION)) == 10
    assert validate_model_task_deck(working).passed


def test_v2_first_expansion_batch_is_s1_s2_eligible_and_freezes_its_holdout() -> None:
    cards = build_model_coding_v2_batch_01_cards()

    assert len(cards) == 20
    assert {card.suite for card in cards} == {MODEL_CODING_V2_BATCH_01_SUITE}
    assert len({card.id for card in cards}) == 20
    assert len(select_model_task_cards(cards, ModelTaskSplit.DEVELOPMENT)) == 12
    assert len(select_model_task_cards(cards, ModelTaskSplit.HOLDOUT)) == 6
    assert len(select_model_task_cards(cards, ModelTaskSplit.REGRESSION)) == 2
    assert all(card.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL for card in cards)
    assert all(len(card.behavior_checks) >= 2 for card in cards)
    assert all(card.source is not ModelTaskSource.PUBLIC_PATTERN_ADAPTED for card in cards)
    assert all(
        card.quality is not None and card.quality.review_state is ModelTaskReviewState.FROZEN
        for card in cards
        if card.split is ModelTaskSplit.HOLDOUT
    )
    assert validate_model_task_deck(cards).passed


def test_v2_second_expansion_batch_completes_the_target_split_without_static_cards() -> None:
    cards = build_model_coding_v2_batch_02_cards()

    assert len(cards) == 28
    assert {card.suite for card in cards} == {MODEL_CODING_V2_BATCH_02_SUITE}
    assert len({card.id for card in cards}) == 28
    assert len(select_model_task_cards(cards, ModelTaskSplit.DEVELOPMENT)) == 15
    assert len(select_model_task_cards(cards, ModelTaskSplit.HOLDOUT)) == 10
    assert len(select_model_task_cards(cards, ModelTaskSplit.REGRESSION)) == 3
    assert all(card.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL for card in cards)
    assert all(len(card.behavior_checks) >= 2 for card in cards)
    assert all(card.source is not ModelTaskSource.PUBLIC_PATTERN_ADAPTED for card in cards)
    assert all(
        card.quality is not None and card.quality.review_state is ModelTaskReviewState.FROZEN
        for card in cards
        if card.split is ModelTaskSplit.HOLDOUT
    )
    assert validate_model_task_deck(cards).passed


def test_v2_formal_100_deck_is_all_behavioral_and_has_the_m0_target_split() -> None:
    cards = build_model_coding_v2_formal_100_cards()

    assert len(cards) == 100
    assert {card.suite for card in cards} == {MODEL_CODING_V2_FORMAL_100_SUITE}
    assert len({card.id for card in cards}) == 100
    assert len(select_model_task_cards(cards, ModelTaskSplit.DEVELOPMENT)) == 60
    assert len(select_model_task_cards(cards, ModelTaskSplit.HOLDOUT)) == 25
    assert len(select_model_task_cards(cards, ModelTaskSplit.REGRESSION)) == 15
    assert all(card.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL for card in cards)
    assert all(len(card.behavior_checks) >= 2 for card in cards)
    assert all(card.source is not ModelTaskSource.PUBLIC_PATTERN_ADAPTED for card in cards)
    assert all(
        card.quality is not None and card.quality.review_state is ModelTaskReviewState.FROZEN
        for card in cards
        if card.split is ModelTaskSplit.HOLDOUT
    )
    assert validate_model_task_deck(cards).passed


def test_v2_reading_coverage_cards_are_behavioral_and_freeze_holdout() -> None:
    cards = build_model_coding_v2_reading_coverage_cards()

    assert len(cards) == 6
    assert {card.suite for card in cards} == {MODEL_CODING_V2_READING_COVERAGE_SUITE}
    assert len(select_model_task_cards(cards, ModelTaskSplit.DEVELOPMENT)) == 2
    assert len(select_model_task_cards(cards, ModelTaskSplit.HOLDOUT)) == 3
    assert len(select_model_task_cards(cards, ModelTaskSplit.REGRESSION)) == 1
    assert all(card.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL for card in cards)
    assert validate_model_task_deck(cards).passed


def test_working_100_deck_has_an_explicit_disposition_for_every_static_card() -> None:
    cards = build_model_coding_v1_working_100_cards()
    static_cards = [card for card in cards if card.acceptance_level is ModelTaskAcceptanceLevel.STATIC_REGRESSION]
    behavioral_cards = [card for card in cards if card.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL]

    assert len(cards) == 100
    assert len(static_cards) == 45
    assert len(behavioral_cards) == 55
    assert all(not card.required_file_contents for card in behavioral_cards)
    assert all(card.static_disposition is not None for card in static_cards)
    assert all(card.static_disposition is None for card in behavioral_cards)
    assert all(
        card.source_reference != "controlled QA expansion fixture"
        for card in behavioral_cards
        if card.id.startswith("coding-v1b3-")
    )
    assert validate_model_task_deck(cards).to_dict()["static_disposition_counts"] == {
        "extend-behavior-oracle": 4,
        "scope-or-read-only": 27,
        "source-evidence-required": 14,
    }
    with pytest.raises(ValueError, match="behavioral model task cannot contain a static disposition"):
        replace(behavioral_cards[0], static_disposition=ModelTaskStaticDisposition.REWRITE_REQUIRED)


def test_formal_candidate_ids_match_the_s1_s2_eligible_working_cards() -> None:
    cards = build_model_coding_v1_working_100_cards()
    expected = {
        card.id
        for card in cards
        if card.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL
        and card.source is not ModelTaskSource.PUBLIC_PATTERN_ADAPTED
    }

    assert set(MODEL_CODING_V1_FORMAL_CANDIDATE_IDS) == expected


def test_message_title_behavior_card_exposes_its_exact_visible_contract() -> None:
    card = next(
        item
        for item in build_model_coding_v1_working_100_cards()
        if item.id == "coding-v1b3-repair-messages-083"
    )

    assert "'Connection failed'" in "\n".join(card.prompts)


def test_model_runner_keeps_evaluator_artifacts_outside_workspace_and_scores_a_scoped_patch(tmp_path: Path) -> None:
    card = ModelTaskCard(
        id="patch-card",
        suite=MODEL_CODING_DEV_V0_SUITE,
        kind=ModelTaskKind.PATCH,
        prompts=("Fix value.py.",),
        initial_files={"value.py": "VALUE = 'before'\n"},
        allowed_paths=("value.py",),
        required_file_contents={"value.py": "VALUE = 'after'"},
    )
    report = _runner([
        LLMResponse(tool_calls=[ToolCall("edit-1", "edit_file", {
            "file_path": "value.py", "old_string": "before", "new_string": "after",
        })]),
        LLMResponse(content="Fixed value.py"),
    ]).run((card,), manifest=_manifest((card,)), output_directory=tmp_path / "artifacts")

    outcome = report.outcomes[0]
    attempt = Path(outcome.artifact_directory)
    assert outcome.status is ModelAttemptStatus.PASSED
    assert outcome.provider_calls == 2
    assert (attempt / "workspace" / "value.py").read_text(encoding="utf-8") == "VALUE = 'after'\n"
    assert (attempt / "events.jsonl").read_text(encoding="utf-8")
    assert "value.py" in (attempt / "changes.patch").read_text(encoding="utf-8")
    saved = json.loads((tmp_path / "artifacts" / "report.json").read_text(encoding="utf-8"))
    assert saved["passed"] == saved["attempt_count"] == 1
    assert json.loads((tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8"))["model"] == "fixed-model-evaluation"
    progress = json.loads((tmp_path / "artifacts" / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_attempt_count"] == progress["planned_attempt_count"] == 1


def test_model_runner_denies_out_of_scope_writes_and_reports_a_task_failure(tmp_path: Path) -> None:
    card = ModelTaskCard(
        id="scope-card",
        suite=MODEL_CODING_DEV_V0_SUITE,
        kind=ModelTaskKind.PATCH,
        prompts=("Fix allowed.py only.",),
        initial_files={"allowed.py": "VALUE = 'before'\n"},
        allowed_paths=("allowed.py",),
        required_file_contents={"allowed.py": "VALUE = 'after'"},
    )
    report = _runner([
        LLMResponse(tool_calls=[ToolCall("write-1", "write_file", {
            "file_path": "outside.py", "content": "bad\n",
        })]),
        LLMResponse(content="Done"),
    ]).run((card,), manifest=_manifest((card,)), output_directory=tmp_path / "artifacts")

    outcome = report.outcomes[0]
    assert outcome.status is ModelAttemptStatus.TASK_FAILED
    assert "required_content:allowed.py" in outcome.failure_categories
    assert not (Path(outcome.artifact_directory) / "workspace" / "outside.py").exists()


def test_model_runner_does_not_report_a_correct_patch_as_success_when_runtime_hits_its_limit(tmp_path: Path) -> None:
    card = ModelTaskCard(
        id="limited-card",
        suite=MODEL_CODING_DEV_V0_SUITE,
        kind=ModelTaskKind.PATCH,
        prompts=("Fix value.py.",),
        initial_files={"value.py": "VALUE = 'before'\n"},
        allowed_paths=("value.py",),
        required_file_contents={"value.py": "VALUE = 'after'"},
        limits=RuntimeLimits(max_provider_calls=1),
    )
    report = _runner([
        LLMResponse(tool_calls=[ToolCall("edit-1", "edit_file", {
            "file_path": "value.py", "old_string": "before", "new_string": "after",
        })]),
    ]).run((card,), manifest=_manifest((card,)), output_directory=tmp_path / "artifacts")

    outcome = report.outcomes[0]
    assert outcome.checks["required_content:value.py"] is True
    assert outcome.status is ModelAttemptStatus.LIMIT_REACHED
    assert outcome.passed is False
    assert outcome.runtime_status == "limit_reached"


def test_model_runner_refuses_a_manifest_for_different_task_cards(tmp_path: Path) -> None:
    cards = build_model_coding_dev_v0_cards()
    wrong = ModelEvaluationManifest(
        provider="fixed-provider",
        model="fixed-model-evaluation",
        parameters={},
        suite=MODEL_CODING_DEV_V0_SUITE,
        case_set_digest="0" * 64,
    )

    with pytest.raises(ValueError, match="do not match"):
        _runner([]).run(cards, manifest=wrong, output_directory=tmp_path / "artifacts")


def test_model_runner_retains_each_repeated_attempt_instead_of_collapsing_the_denominator(tmp_path: Path) -> None:
    card = ModelTaskCard(
        id="repeat-card",
        suite=MODEL_CODING_DEV_V0_SUITE,
        kind=ModelTaskKind.READ_ONLY,
        prompts=("State the flag.",),
        initial_files={"flag.txt": "flag: green\n"},
        required_response_facts=("green",),
    )
    report = _runner([LLMResponse(content="green"), LLMResponse(content="green")]).run(
        (card,),
        manifest=_manifest((card,)),
        output_directory=tmp_path / "artifacts",
        attempts_per_task=2,
    )

    assert report.started == report.passed == 2
    assert report.attempts_per_task == 2
    assert [outcome.attempt for outcome in report.outcomes] == [1, 2]


def test_model_runner_resumes_an_interrupted_matching_run_without_repeating_completed_attempts(tmp_path: Path) -> None:
    card = ModelTaskCard(
        id="resume-card",
        suite=MODEL_CODING_DEV_V0_SUITE,
        kind=ModelTaskKind.READ_ONLY,
        prompts=("State the flag.",),
        initial_files={"flag.txt": "flag: green\n"},
        required_response_facts=("green",),
    )

    class InterruptAfterOneAttempt(ModelEvaluationRunner):
        def __init__(self) -> None:
            super().__init__(RuntimeBootstrap(provider_factory=lambda _config: FixedProvider([LLMResponse(content="green")])))
            self.started_attempts = 0

        def _run_attempt(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.started_attempts:
                raise KeyboardInterrupt("simulated host interruption")
            self.started_attempts += 1
            return super()._run_attempt(*args, **kwargs)

    output = tmp_path / "artifacts"
    with pytest.raises(KeyboardInterrupt, match="simulated host interruption"):
        InterruptAfterOneAttempt().run(
            (card,),
            manifest=_manifest((card,)),
            output_directory=output,
            attempts_per_task=2,
        )

    report = _runner([LLMResponse(content="green")]).run(
        (card,),
        manifest=_manifest((card,)),
        output_directory=output,
        attempts_per_task=2,
    )
    assert report.started == report.passed == 2
    assert [outcome.attempt for outcome in report.outcomes] == [1, 2]
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_attempt_count"] == 2


def test_model_runner_recovers_a_persisted_attempt_missing_its_progress_checkpoint(tmp_path: Path) -> None:
    card = ModelTaskCard(
        id="persisted-resume-card",
        suite=MODEL_CODING_DEV_V0_SUITE,
        kind=ModelTaskKind.READ_ONLY,
        prompts=("State the flag.",),
        initial_files={"flag.txt": "flag: green\n"},
        required_response_facts=("green",),
    )

    class InterruptAfterPersistingAttempt(ModelEvaluationRunner):
        def _run_attempt(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super()._run_attempt(*args, **kwargs)
            raise KeyboardInterrupt("simulated interruption after attempt artifact")

    output = tmp_path / "artifacts"
    with pytest.raises(KeyboardInterrupt, match="after attempt artifact"):
        InterruptAfterPersistingAttempt(
            RuntimeBootstrap(provider_factory=lambda _config: FixedProvider([LLMResponse(content="green")]))
        ).run((card,), manifest=_manifest((card,)), output_directory=output)

    report = _runner([]).run((card,), manifest=_manifest((card,)), output_directory=output)
    assert report.started == report.passed == 1
    assert json.loads((output / "progress.json").read_text(encoding="utf-8"))["completed_attempt_count"] == 1


def test_model_runner_archives_an_incomplete_attempt_before_retrying(tmp_path: Path) -> None:
    card = ModelTaskCard(
        id="partial-resume-card",
        suite=MODEL_CODING_DEV_V0_SUITE,
        kind=ModelTaskKind.READ_ONLY,
        prompts=("State the flag.",),
        initial_files={"flag.txt": "flag: green\n"},
        required_response_facts=("green",),
    )

    class InterruptWithPartialAttempt(ModelEvaluationRunner):
        def _run_attempt(self, card, attempt, output_root, **kwargs):  # type: ignore[no-untyped-def]
            attempt_root = output_root / card.id / f"attempt-{attempt}"
            attempt_root.mkdir(parents=True)
            (attempt_root / "partial.txt").write_text("interrupted", encoding="utf-8")
            raise KeyboardInterrupt("simulated interruption during attempt")

    output = tmp_path / "artifacts"
    with pytest.raises(KeyboardInterrupt, match="during attempt"):
        InterruptWithPartialAttempt(RuntimeBootstrap()).run((card,), manifest=_manifest((card,)), output_directory=output)

    report = _runner([LLMResponse(content="green")]).run((card,), manifest=_manifest((card,)), output_directory=output)
    assert report.started == report.passed == 1
    assert (output / card.id / "attempt-1-interrupted-1" / "partial.txt").read_text(encoding="utf-8") == "interrupted"


def test_model_report_separates_attempt_metrics_from_task_pass_at_k_and_task_families(tmp_path: Path) -> None:
    cards = (
        ModelTaskCard(
            id="metrics-read-card",
            suite=MODEL_CODING_DEV_V0_SUITE,
            kind=ModelTaskKind.READ_ONLY,
            prompts=("State the flag.",),
            initial_files={"flag.txt": "flag: green\n"},
            required_response_facts=("green",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern; local read-only adaptation",
            template_id="read-configuration-fact",
        ),
        ModelTaskCard(
            id="metrics-patch-card",
            suite=MODEL_CODING_DEV_V0_SUITE,
            kind=ModelTaskKind.PATCH,
            prompts=("Fix value.py.",),
            initial_files={"value.py": "VALUE = 'before'\n"},
            allowed_paths=("value.py",),
            required_file_contents={"value.py": "VALUE = 'after'"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled wrong-constant injection",
            template_id="single-file-constant-repair",
        ),
    )
    report = ModelEvaluationReport(
        manifest=_manifest(cards),
        cards=cards,
        outcomes=tuple(
            ModelAttemptOutcome(
                task_id=task_id,
                attempt=attempt,
                status=status,
                response="",
                checks={},
                failure_categories=(),
                provider_calls=provider_calls,
                prompt_tokens=10 * attempt,
                completion_tokens=attempt,
                estimated_cost=None,
                elapsed_seconds=float(attempt),
                artifact_directory=f"artifacts/{task_id}/attempt-{attempt}",
            )
            for task_id, attempt, status, provider_calls in (
                ("metrics-read-card", 1, ModelAttemptStatus.PASSED, 1),
                ("metrics-read-card", 2, ModelAttemptStatus.PASSED, 1),
                ("metrics-read-card", 3, ModelAttemptStatus.PASSED, 1),
                ("metrics-patch-card", 1, ModelAttemptStatus.TASK_FAILED, 1),
                ("metrics-patch-card", 2, ModelAttemptStatus.PASSED, 2),
                ("metrics-patch-card", 3, ModelAttemptStatus.PASSED, 2),
            )
        ),
    )

    metrics = report.metrics()
    assert report.attempts_per_task == 3
    assert metrics["task_pass_at_1"] == pytest.approx(0.5)
    assert metrics["task_pass_at_3"] == pytest.approx(1.0)
    assert metrics["by_family"]["code-reading"]["pass_at_3"] == pytest.approx(1.0)
    assert metrics["by_family"]["scoped-repair"]["pass_at_1"] == pytest.approx(0.0)
    assert metrics["provider_calls"]["observed_count"] == 6
    rendered = report.to_dict()
    assert rendered["schema_version"] == 2
    assert rendered["metrics"]["task_pass_at_3"] == pytest.approx(1.0)


def test_model_budget_is_distributed_per_runtime_attempt_not_per_prompt() -> None:
    assert _per_attempt_token_limit(900_000, card_count=31, attempts_per_task=3) == 9_677
    assert _per_attempt_token_limit(700_000, card_count=31, attempts_per_task=3) == 7_526
    assert _per_attempt_token_limit(None, card_count=31, attempts_per_task=3) is None


def test_model_task_id_selection_rejects_cards_outside_the_selected_deck() -> None:
    cards = build_model_coding_dev_v0_cards()
    selected = _select_model_task_ids(cards, (cards[0].id,))

    assert selected == (cards[0],)
    with pytest.raises(ValueError, match="not available"):
        _select_model_task_ids(cards, ("unknown-card",))


def test_model_cli_requires_explicit_execution_budget_and_artifact_directory() -> None:
    with pytest.raises(SystemExit) as missing_opt_in:
        evaluation_main(["--suite", MODEL_CODING_DEV_V0_SUITE])
    assert missing_opt_in.value.code == 2

    with pytest.raises(SystemExit) as missing_budget:
        evaluation_main(["--suite", MODEL_CODING_DEV_V0_SUITE, "--run-model", "--model-output", "artifacts"])
    assert missing_budget.value.code == 2

    with pytest.raises(SystemExit) as ambiguous_output:
        evaluation_main([
            "--suite", MODEL_CODING_DEV_V0_SUITE,
            "--run-model",
            "--model-output", "artifacts",
            "--max-cost-usd", "1",
            "--output", "report.json",
        ])
    assert ambiguous_output.value.code == 2

    with pytest.raises(SystemExit) as missing_seed_split:
        evaluation_main([
            "--suite", MODEL_CODING_V1_SEED_SUITE,
            "--run-model",
            "--model-output", "artifacts",
            "--max-total-tokens", "1000",
        ])
    assert missing_seed_split.value.code == 2

    with pytest.raises(SystemExit) as missing_qa_seed_split:
        evaluation_main([
            "--suite", MODEL_CODING_V1_SEED_QA_V1_SUITE,
            "--run-model",
            "--model-output", "artifacts",
            "--max-total-tokens", "1000",
        ])
    assert missing_qa_seed_split.value.code == 2

    with pytest.raises(SystemExit) as missing_batch_split:
        evaluation_main([
            "--suite", MODEL_CODING_V1_BATCH_01_SUITE,
            "--run-model",
            "--model-output", "artifacts",
            "--max-total-tokens", "1000",
        ])
    assert missing_batch_split.value.code == 2

    with pytest.raises(SystemExit) as missing_working_split:
        evaluation_main([
            "--suite", MODEL_CODING_V1_WORKING_40_SUITE,
            "--run-model",
            "--model-output", "artifacts",
            "--max-total-tokens", "1000",
        ])
    assert missing_working_split.value.code == 2

    with pytest.raises(SystemExit) as missing_working_70_split:
        evaluation_main([
            "--suite", MODEL_CODING_V1_WORKING_70_SUITE,
            "--run-model",
            "--model-output", "artifacts",
            "--max-total-tokens", "1000",
        ])
    assert missing_working_70_split.value.code == 2

    with pytest.raises(SystemExit) as missing_v2_formal_split:
        evaluation_main([
            "--suite", MODEL_CODING_V2_FORMAL_100_SUITE,
            "--run-model",
            "--model-output", "artifacts",
            "--max-total-tokens", "1000",
        ])
    assert missing_v2_formal_split.value.code == 2
