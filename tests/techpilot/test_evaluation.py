from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from techpilot.evaluation import (
    CORE_V0_CASE_COUNT,
    CORE_V0_CASE_SET_DIGEST,
    HOLDOUT_SCHEMA_VERSION,
    HOLDOUT_SUITE,
    LONG_TASK_HOLDOUT_FORMAT,
    LONG_TASK_HOLDOUT_SCHEMA_VERSION,
    LONG_TASK_HOLDOUT_SUITE,
    LONG_TASK_OBSERVABILITY_V0_CASE_COUNT,
    LONG_TASK_OBSERVABILITY_V0_CASE_SET_DIGEST,
    LONG_TASK_RUNTIME_SMOKE_CASE_COUNT,
    LONG_TASK_RUNTIME_SMOKE_CASE_SET_DIGEST,
    LONG_TASK_RUNTIME_V0_CASE_COUNT,
    LONG_TASK_RUNTIME_V0_CASE_SET_DIGEST,
    ROLE_RUNTIME_VALIDATION_CASE_COUNT,
    ROLE_RUNTIME_VALIDATION_CASE_SET_DIGEST,
    RUNNER_VALIDATION_CASE_COUNT,
    BaselineReference,
    ExtendedCaseProvenance,
    ExtendedCaseSource,
    HoldoutFormatError,
    ModelEvaluationManifest,
    ReplayCase,
    ReplayCaseOrigin,
    ReplayCategory,
    ReplayRunner,
    build_core_v0_cases,
    build_long_task_observability_v0_cases,
    build_long_task_runtime_smoke_cases,
    build_long_task_runtime_v0_cases,
    build_role_runtime_validation_cases,
    build_runner_validation_cases,
    case_set_digest,
    holdout_case_set_metadata,
    inspect_holdout_case_schema,
    inspect_long_task_holdout_design,
    load_holdout_suite,
    load_long_task_holdout_suite,
    long_task_holdout_case_set_metadata,
    run_holdout,
    run_long_task_holdout,
    write_holdout_summary,
)
from techpilot.evaluation.__main__ import main as evaluation_main


def test_core_v0_has_144_unique_cases_with_separate_runtime_categories() -> None:
    cases = build_core_v0_cases()

    assert len(cases) == CORE_V0_CASE_COUNT == 144
    assert case_set_digest(cases) == CORE_V0_CASE_SET_DIGEST
    assert len({case.id for case in cases}) == CORE_V0_CASE_COUNT
    assert {case.category for case in cases} == set(ReplayCategory)
    assert {category: sum(item.category is category for item in cases) for category in ReplayCategory} == {
        ReplayCategory.TOOL: 16,
        ReplayCategory.SCHEDULING: 64,
        ReplayCategory.CONTEXT: 16,
        ReplayCategory.PERSISTENCE: 16,
        ReplayCategory.CONTRACT: 16,
        ReplayCategory.INSTRUCTION: 16,
    }


def test_runner_validation_deck_has_24_cases_across_every_handler() -> None:
    cases = build_runner_validation_cases()

    assert len(cases) == RUNNER_VALIDATION_CASE_COUNT == 24
    assert {case.category for case in cases} == set(ReplayCategory)
    assert all(case.suite == "runner-validation-v0" for case in cases)
    assert all(case.origin is ReplayCaseOrigin.RUNNER_VALIDATION for case in cases)


def test_role_runtime_validation_suite_exercises_isolation_and_recovery_without_changing_core() -> None:
    cases = build_role_runtime_validation_cases()

    assert len(cases) == ROLE_RUNTIME_VALIDATION_CASE_COUNT == 9
    assert case_set_digest(cases) == ROLE_RUNTIME_VALIDATION_CASE_SET_DIGEST
    assert {case.scenario for case in cases} == {"role-runtime-lifecycle"}
    assert {case.input["mode"] for case in cases} == {
        "registration",
        "disabled",
        "incompatible",
        "configuration",
        "activation",
        "switch",
        "scope-exception",
        "resume",
        "clear-chat",
    }
    assert all(case.origin is ReplayCaseOrigin.RUNNER_VALIDATION for case in cases)
    report = ReplayRunner(Path(__file__).parents[2]).run(cases)
    assert report.passed == report.total == ROLE_RUNTIME_VALIDATION_CASE_COUNT


def test_long_task_runtime_smoke_suite_exercises_fixed_provider_recovery_without_changing_core() -> None:
    cases = build_long_task_runtime_smoke_cases()

    assert len(cases) == LONG_TASK_RUNTIME_SMOKE_CASE_COUNT == 6
    assert case_set_digest(cases) == LONG_TASK_RUNTIME_SMOKE_CASE_SET_DIGEST
    assert {case.scenario for case in cases} == {"long-task-runtime"}
    assert {case.input["mode"] for case in cases} == {
        "normal-write",
        "permission-denied",
        "unknown-effect",
        "completed-effect-skip",
        "checkpoint-cursor",
        "plain-session-control",
    }
    assert all(case.origin is ReplayCaseOrigin.RUNNER_VALIDATION for case in cases)
    report = ReplayRunner(Path(__file__).parents[2]).run(cases)
    assert report.passed == report.total == LONG_TASK_RUNTIME_SMOKE_CASE_COUNT


def test_long_task_v0_cards_are_frozen_before_remaining_runner_modes_are_implemented() -> None:
    cards = build_long_task_runtime_v0_cases()
    implemented_modes = {
        "normal-write", "normal-read", "normal-command", "permission-denied", "policy-blocked",
        "interrupt-before-effect", "unknown-effect", "completed-effect-skip", "read-then-interrupt",
        "command-completed-skip", "same-round-read-write", "same-round-write-write", "same-round-read-read",
        "opaque-tool-exclusive", "checkpoint-cursor", "plain-session-control",
        "session-mismatch", "repository-mismatch", "lease-conflict", "effect-budget", "cancelled-task",
        "two-turn-read-write", "two-turn-write-read", "corrupt-event-log",
    }
    candidate = tuple(card for card in cards if card.input["mode"] in implemented_modes)

    assert len(cards) == LONG_TASK_RUNTIME_V0_CASE_COUNT == 24
    assert case_set_digest(cards) == LONG_TASK_RUNTIME_V0_CASE_SET_DIGEST
    assert len(candidate) == 24
    report = ReplayRunner(Path(__file__).parents[2]).run(candidate)
    assert report.passed == report.total == 24


def test_long_task_observability_extension_records_real_tool_bodies_without_rewriting_v0() -> None:
    cases = build_long_task_observability_v0_cases()

    assert len(cases) == LONG_TASK_OBSERVABILITY_V0_CASE_COUNT == 4
    assert case_set_digest(cases) == LONG_TASK_OBSERVABILITY_V0_CASE_SET_DIGEST
    assert {case.scenario for case in cases} == {"long-task-observability"}
    assert {case.input["mode"] for case in cases} == {
        "command-effect-recovery",
        "read-read-overlap",
        "read-write-order",
        "write-write-exclusive",
    }
    assert all(case.origin is ReplayCaseOrigin.EXTENDED for case in cases)
    assert all(case.provenance is not None for case in cases)
    report = ReplayRunner(Path(__file__).parents[2]).run(cases)
    assert report.passed == report.total == LONG_TASK_OBSERVABILITY_V0_CASE_COUNT


def test_extended_cases_require_evidence_backed_provenance() -> None:
    source = build_core_v0_cases()[0]

    with pytest.raises(ValueError, match="require evidence-backed provenance"):
        replace(source, id="extended-without-evidence", suite="extended-v0", origin=ReplayCaseOrigin.EXTENDED)

    provenance = ExtendedCaseProvenance(
        source=ExtendedCaseSource.REAL_DEFECT,
        evidence_id="issue-123",
        first_observed_commit="abc123",
        pre_fix_failure="tool event was missing from the session projection",
        rationale="Protect the observed regression from recurring.",
    )
    with pytest.raises(ValueError, match=r"must use an extended-\* suite"):
        replace(source, id="extended-wrong-suite", origin=ReplayCaseOrigin.EXTENDED, provenance=provenance)

    case = replace(
        source,
        id="extended-real-defect-001",
        suite="extended-v0",
        origin=ReplayCaseOrigin.EXTENDED,
        provenance=provenance,
    )

    assert case.provenance is not None
    assert case.to_dict()["provenance"]["source"] == "real-defect"


def test_baseline_comparison_refuses_a_changed_case_deck() -> None:
    runner = ReplayRunner(Path(__file__).parents[2])
    report = runner.run(build_core_v0_cases())
    baseline = BaselineReference.from_report(report)

    assert baseline.compare(report).comparable is True
    changed_case = replace(report.cases[0], description="changed case definition")
    changed_report = replace(report, cases=(changed_case, *report.cases[1:]))
    comparison = baseline.compare(changed_report)

    assert comparison.comparable is False
    assert comparison.reason == "case_set_changed"


def test_candidate_report_cannot_be_used_as_a_baseline(tmp_path: Path) -> None:
    report = ReplayRunner(Path(__file__).parents[2]).run(build_core_v0_cases())
    candidate = ReplayRunner.write_report(report, tmp_path / "candidate.json")

    with pytest.raises(ValueError, match="invalid baseline report"):
        BaselineReference.from_path(candidate)


def test_core_v0_runner_is_offline_deterministic_and_writes_a_manifest(tmp_path: Path) -> None:
    cases = build_core_v0_cases()
    runner = ReplayRunner(Path(__file__).parents[2])

    first = runner.run(cases)
    second = runner.run(cases)
    output = runner.write_report(first, tmp_path / "baseline-v0-candidate.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert first.passed == first.total == 144
    assert second.passed == second.total == 144
    assert first.case_set_digest == second.case_set_digest
    assert payload["case_set_digest"] == first.case_set_digest
    assert isinstance(payload["git_dirty"], bool)
    assert payload["categories"] == {
        "tool": {"passed": 16, "total": 16},
        "scheduling": {"passed": 64, "total": 64},
        "context": {"passed": 16, "total": 16},
        "persistence": {"passed": 16, "total": 16},
        "contract": {"passed": 16, "total": 16},
        "instruction": {"passed": 16, "total": 16},
    }


def test_baseline_rejects_dirty_or_failed_core_runs(tmp_path: Path) -> None:
    runner = ReplayRunner(Path(__file__).parents[2])
    report = runner.run(build_core_v0_cases())

    with pytest.raises(ValueError, match="clean Git worktree"):
        runner.write_baseline(replace(report, git_dirty=True), tmp_path / "baseline-v0.json")
    with pytest.raises(ValueError, match="every core-v0 case to pass"):
        runner.write_baseline(
            replace(report, outcomes=(replace(report.outcomes[0], passed=False), *report.outcomes[1:]), git_dirty=False),
            tmp_path / "baseline-v0.json",
        )
    output = runner.write_baseline(replace(report, git_dirty=False), tmp_path / "baseline-v0.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["git_dirty"] is False
    assert payload["baseline"]["kind"] == "baseline-v0"
    assert BaselineReference.from_path(output).compare(report).comparable is True


def test_replay_report_fails_one_case_without_hiding_its_category() -> None:
    source = build_core_v0_cases()[0]
    broken = ReplayCase(
        id="tool-broken-expectation",
        suite=source.suite,
        category=source.category,
        scenario=source.scenario,
        input=source.input,
        expected={**source.expected, "tool_result": "this must fail"},
        description=source.description,
    )

    report = ReplayRunner(Path(__file__).parents[2]).run((broken,))

    assert report.passed == 0
    assert report.total == 1
    assert report.outcomes[0].category is ReplayCategory.TOOL
    assert report.outcomes[0].failure == "Tool result did not preserve the expected argument"


def test_model_manifest_carries_conditions_but_never_a_fake_score() -> None:
    manifest = ModelEvaluationManifest(
        provider="example-provider",
        model="example-model",
        parameters={"temperature": 0},
        suite="model-core-v0",
        case_set_digest="a" * 64,
    )

    assert manifest.to_dict() == {
        "provider": "example-provider",
        "model": "example-model",
        "parameters": {"temperature": 0},
        "suite": "model-core-v0",
        "case_set_digest": "a" * 64,
        "track": "model",
    }
    with pytest.raises(ValueError, match="requires provider"):
        ModelEvaluationManifest(
            provider="",
            model="example-model",
            parameters={},
            suite="model-core-v0",
            case_set_digest="a" * 64,
        )


def test_private_holdout_loader_validates_integrity_and_writes_only_a_redacted_summary(tmp_path: Path) -> None:
    case = ReplayCase(
        id="holdout-tool-success-001",
        suite=HOLDOUT_SUITE,
        category=ReplayCategory.TOOL,
        scenario="agent-tool-turn",
        input={"mode": "success", "value": "private-marker", "provider_response": "done"},
        expected={"response": "done", "tool_result": "echo:private-marker"},
        origin=ReplayCaseOrigin.HOLDOUT,
        description="Synthetic stand-in; no external holdout content is used in this test.",
    )
    root = tmp_path / "private-holdout"
    root.mkdir()
    (root / "reports").mkdir()
    (root / "cases.jsonl").write_text(json.dumps(case.to_dict()) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "suite": HOLDOUT_SUITE,
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "case_count": 1,
        "case_set_digest": case_set_digest((case,)),
    }), encoding="utf-8")

    loaded = load_holdout_suite(root)
    summary = run_holdout(root, ReplayRunner(Path(__file__).parents[2]))
    output = write_holdout_summary(summary, root / "reports" / "summary.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert loaded.cases == (case,)
    assert summary.passed == summary.total == 1
    assert summary.failed_case_ids == ()
    assert payload["kind"] == "holdout-summary-v0"
    assert payload["failed_case_ids"] == []
    assert "outcomes" not in payload
    assert "observed" not in payload
    assert "private-marker" not in output.read_text(encoding="utf-8")


def test_private_holdout_rejects_a_changed_or_non_holdout_case_deck(tmp_path: Path) -> None:
    root = tmp_path / "private-holdout"
    root.mkdir()
    case = ReplayCase(
        id="holdout-tool-success-001",
        suite=HOLDOUT_SUITE,
        category=ReplayCategory.TOOL,
        scenario="agent-tool-turn",
        input={"mode": "success", "value": "marker", "provider_response": "done"},
        expected={"response": "done", "tool_result": "echo:marker"},
        origin=ReplayCaseOrigin.HOLDOUT,
    )
    altered = case.to_dict() | {"origin": ReplayCaseOrigin.CORE.value}
    (root / "cases.jsonl").write_text(json.dumps(altered) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "suite": HOLDOUT_SUITE,
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "case_count": 1,
        "case_set_digest": case_set_digest((case,)),
    }), encoding="utf-8")

    with pytest.raises(HoldoutFormatError, match="invalid holdout case at line 1"):
        load_holdout_suite(root)


def test_private_holdout_manifest_errors_expose_only_safe_structure_details(tmp_path: Path) -> None:
    root = tmp_path / "private-holdout"
    root.mkdir()

    with pytest.raises(HoldoutFormatError, match="manifest.json is missing"):
        load_holdout_suite(root)

    (root / "manifest.json").write_text(json.dumps({"suite": HOLDOUT_SUITE}), encoding="utf-8")
    with pytest.raises(HoldoutFormatError, match="missing required fields: schema_version, case_count, case_set_digest"):
        load_holdout_suite(root)


def test_private_holdout_case_set_metadata_exposes_only_count_and_digest(tmp_path: Path) -> None:
    case = ReplayCase(
        id="holdout-tool-success-001",
        suite=HOLDOUT_SUITE,
        category=ReplayCategory.TOOL,
        scenario="agent-tool-turn",
        input={"mode": "success", "value": "private-marker", "provider_response": "done"},
        expected={"response": "done", "tool_result": "echo:private-marker"},
        origin=ReplayCaseOrigin.HOLDOUT,
    )
    root = tmp_path / "private-holdout"
    root.mkdir()
    (root / "cases.jsonl").write_text(json.dumps(case.to_dict()) + "\n", encoding="utf-8")

    assert holdout_case_set_metadata(root) == (1, case_set_digest((case,)))


def test_private_holdout_schema_inspection_exposes_field_names_without_values(tmp_path: Path) -> None:
    root = tmp_path / "private-holdout"
    root.mkdir()
    (root / "cases.jsonl").write_text(json.dumps({
        "opaque_id": "private-marker",
        "assertions": ["private expected output"],
    }) + "\n", encoding="utf-8")

    schema = inspect_holdout_case_schema(root)

    assert schema.case_count == 1
    assert schema.fields == ("assertions", "opaque_id")
    assert "private-marker" not in repr(schema)


def test_long_task_holdout_design_inspection_validates_only_its_fixed_structure(tmp_path: Path) -> None:
    root = tmp_path / "long-task-holdout-design"
    root.mkdir()
    card = {
        "id": "private-long-task-001",
        "category": "persistence",
        "title": "Private interruption case",
        "initial_state": {"files": {"private.txt": "private evidence"}},
        "provider_script": [{"content": "private fixed response"}],
        "interruption_point": "after-first-step",
        "recovery_action": "resume",
        "assertions": ["private assertion"],
        "expected_support": "supported",
        "runtime_requirement": "fixed-provider",
        "why_independent": "authored outside the Runtime implementation",
    }
    (root / "cases.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")

    metadata = inspect_long_task_holdout_design(root)

    assert metadata.case_count == 1
    assert metadata.fields == (
        "assertions", "category", "expected_support", "id", "initial_state", "interruption_point",
        "provider_script", "recovery_action", "runtime_requirement", "title", "why_independent",
    )
    assert len(metadata.case_set_digest) == 64
    assert "private evidence" not in repr(metadata)


def test_long_task_holdout_design_rejects_non_design_or_incomplete_cards(tmp_path: Path) -> None:
    root = tmp_path / "long-task-holdout-design"
    root.mkdir()
    (root / "cases.jsonl").write_text(json.dumps({"id": "private-long-task-001"}) + "\n", encoding="utf-8")

    with pytest.raises(HoldoutFormatError, match="invalid long-task holdout design case at line 1"):
        inspect_long_task_holdout_design(root)


def test_long_task_holdout_design_cli_prints_only_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "long-task-holdout-design"
    root.mkdir()
    card = {
        "id": "private-long-task-001",
        "category": "persistence",
        "title": "Private interruption case",
        "initial_state": {"files": {"private.txt": "private evidence"}},
        "provider_script": [{"content": "private fixed response"}],
        "interruption_point": "after-first-step",
        "recovery_action": "resume",
        "assertions": ["private assertion"],
        "expected_support": "supported",
        "runtime_requirement": "fixed-provider",
        "why_independent": "authored outside the Runtime implementation",
    }
    (root / "cases.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")

    assert evaluation_main(["--long-task-holdout-design-metadata", str(root)]) == 0

    rendered = capsys.readouterr().out
    assert "case_count: 1" in rendered
    assert "status: design_only" in rendered
    assert "private evidence" not in rendered


def test_private_long_task_holdout_runs_the_fixed_provider_contract_without_writing_case_detail(tmp_path: Path) -> None:
    root = tmp_path / "private-long-task-holdout"
    root.mkdir()
    card = {
        "id": "private-long-task-001",
        "category": "persistence",
        "title": "Private completed effect case",
        "initial_state": {
            "files": {},
            "user_input": "write private result",
            "task": {"id": "private-long-task-001", "goal": "write a private result", "max_effects": 1},
            "permission": "allow",
        },
        "provider_script": {
            "initial": [
                {"tool_calls": [{"id": "write-1", "name": "write_file", "arguments": {"file_path": "result.txt", "content": "private result\\n"}}]},
                {"content": "complete"},
            ],
            "resume": [],
        },
        "interruption_point": {"phase": "none"},
        "recovery_action": {"kind": "none"},
        "assertions": {
            "task_status": "succeeded",
            "initial_executor_calls": 1,
            "resume_executor_calls": 0,
            "completed_effect_ids": ["effect-write-1"],
            "session_cursor_advanced": True,
            "files": {"result.txt": {"exists": True}},
        },
        "expected_support": "supported",
        "runtime_requirement": "fixed-provider-v0",
        "why_independent": "Designed without Runtime implementation access.",
    }
    (root / "cases.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")
    count, digest = long_task_holdout_case_set_metadata(root)
    (root / "manifest.json").write_text(json.dumps({
        "suite": LONG_TASK_HOLDOUT_SUITE,
        "format": LONG_TASK_HOLDOUT_FORMAT,
        "schema_version": LONG_TASK_HOLDOUT_SCHEMA_VERSION,
        "case_count": count,
        "case_set_digest": digest,
    }), encoding="utf-8")

    loaded = load_long_task_holdout_suite(root)
    summary = run_long_task_holdout(root, ReplayRunner(Path(__file__).parents[2]))
    output = write_holdout_summary(summary, root / "reports" / "summary.json")

    assert len(loaded.cases) == 1
    assert summary.passed == summary.total == 1
    rendered_summary = output.read_text(encoding="utf-8")
    assert summary.case_set_digest == digest
    assert summary.replay_case_set_digest is not None
    assert summary.replay_case_set_digest != digest
    assert "private result" not in rendered_summary
    assert "replay_case_set_digest" in rendered_summary

    mismatch_root = tmp_path / "private-long-task-holdout-mismatch"
    mismatch_root.mkdir()
    mismatched_card = card | {"assertions": card["assertions"] | {"initial_executor_calls": 0}}
    (mismatch_root / "cases.jsonl").write_text(json.dumps(mismatched_card) + "\n", encoding="utf-8")
    mismatch_count, mismatch_digest = long_task_holdout_case_set_metadata(mismatch_root)
    (mismatch_root / "manifest.json").write_text(json.dumps({
        "suite": LONG_TASK_HOLDOUT_SUITE,
        "format": LONG_TASK_HOLDOUT_FORMAT,
        "schema_version": LONG_TASK_HOLDOUT_SCHEMA_VERSION,
        "case_count": mismatch_count,
        "case_set_digest": mismatch_digest,
    }), encoding="utf-8")

    mismatch_summary = run_long_task_holdout(mismatch_root, ReplayRunner(Path(__file__).parents[2]))

    assert mismatch_summary.failure_kind_by_case == {"private-long-task-001": "assertion_mismatch"}
    assert mismatch_summary.observed_by_case["private-long-task-001"] == {
        "mismatch_codes": ["assertion_initial_executor_count_mismatch"],
        "task_status": "succeeded",
        "initial_executor_calls": 1,
        "resume_executor_calls": 0,
        "completed_effect_count": 1,
        "session_cursor_advanced": True,
        "file_assertions": {"matched": 1, "total": 1},
    }


def test_private_long_task_holdout_cli_prints_only_redacted_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "private-long-task-holdout"
    root.mkdir()
    card = {
        "id": "private-long-task-001",
        "category": "persistence",
        "title": "Private read case",
        "initial_state": {
            "files": {"evidence.txt": "private evidence\\n"},
            "user_input": "read private evidence",
            "task": {"id": "private-long-task-001", "goal": "read private evidence", "max_effects": 1},
            "permission": "allow",
        },
        "provider_script": {
            "initial": [
                {"tool_calls": [{"id": "read-1", "name": "read_file", "arguments": {"file_path": "evidence.txt"}}]},
                {"content": "complete"},
            ],
            "resume": [],
        },
        "interruption_point": {"phase": "none"},
        "recovery_action": {"kind": "none"},
        "assertions": {
            "task_status": "succeeded",
            "initial_executor_calls": 1,
            "resume_executor_calls": 0,
            "completed_effect_ids": [],
            "session_cursor_advanced": True,
            "files": {"evidence.txt": {"exists": True}},
        },
        "expected_support": "supported",
        "runtime_requirement": "fixed-provider-v0",
        "why_independent": "Designed without Runtime implementation access.",
    }
    (root / "cases.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")
    count, digest = long_task_holdout_case_set_metadata(root)
    (root / "manifest.json").write_text(json.dumps({
        "suite": LONG_TASK_HOLDOUT_SUITE,
        "format": LONG_TASK_HOLDOUT_FORMAT,
        "schema_version": LONG_TASK_HOLDOUT_SCHEMA_VERSION,
        "case_count": count,
        "case_set_digest": digest,
    }), encoding="utf-8")

    assert evaluation_main(["--long-task-holdout-root", str(root)]) == 0

    rendered = capsys.readouterr().out
    assert "long-task-holdout-v0: 1/1 passed" in rendered
    assert "source_case_set_digest" in rendered
    assert "replay_case_set_digest" in rendered
    assert "private evidence" not in rendered


def test_private_long_task_holdout_reports_only_a_safe_contract_failure_code() -> None:
    case = ReplayCase(
        id="private-long-task-contract-001",
        suite=LONG_TASK_HOLDOUT_SUITE,
        category=ReplayCategory.CONTRACT,
        scenario="long-task-holdout",
        input={},
        expected={},
        origin=ReplayCaseOrigin.HOLDOUT,
    )

    report = ReplayRunner(Path(__file__).parents[2]).run((case,))

    assert report.passed == 0
    assert report.outcomes[0].failure == "long-task-holdout:execution_contract_type_mismatch"
    assert "private" not in report.outcomes[0].failure.removeprefix("long-task-holdout:")


def test_private_holdout_cli_prints_only_the_redacted_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = ReplayCase(
        id="holdout-tool-success-001",
        suite=HOLDOUT_SUITE,
        category=ReplayCategory.TOOL,
        scenario="agent-tool-turn",
        input={"mode": "success", "value": "private-marker", "provider_response": "done"},
        expected={"response": "done", "tool_result": "echo:private-marker"},
        origin=ReplayCaseOrigin.HOLDOUT,
    )
    root = tmp_path / "private-holdout"
    root.mkdir()
    (root / "cases.jsonl").write_text(json.dumps(case.to_dict()) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "suite": HOLDOUT_SUITE,
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "case_count": 1,
        "case_set_digest": case_set_digest((case,)),
    }), encoding="utf-8")

    assert evaluation_main(["--holdout-root", str(root)]) == 0

    rendered = capsys.readouterr().out
    reports = list((root / "reports").glob("*.json"))
    assert "holdout-v0: 1/1 passed" in rendered
    assert "failed_case_ids: none" in rendered
    assert "private-marker" not in rendered
    assert len(reports) == 1
    assert "private-marker" not in reports[0].read_text(encoding="utf-8")
