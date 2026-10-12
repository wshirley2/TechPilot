"""Independent calibration of the append-only Goal pilot protocol."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from techpilot.evaluation import goal_model_protocol as protocol
from techpilot.evaluation.goal_model_oracle import candidate_digest_for_workspace
from techpilot.evaluation.goal_model_v0 import load_deck, prepare_pilot


def _prepare(tmp_path: Path, *, max_total_tokens: int = 100) -> Path:
    root = tmp_path / "pilot"
    prepare_pilot(root, model="calibration-only", max_cost_usd=None, max_total_tokens=max_total_tokens)
    return root


def _observe(
    root: Path,
    case_id: str,
    label: str,
    *,
    event_count: int = 1,
    session_resumes: int = 0,
    verified_stages: list[str] | None = None,
    ready_stages: list[str] | None = None,
    checkpoints: int = 0,
    continuous_batches: int = 0,
) -> Path:
    observation = {
        "schema_version": 1,
        "suite": "goal-model-pilot-v0",
        "track": "model",
        "case_id": case_id,
        "case_set_digest": load_deck().digest,
        "label": label,
        "task_id": f"goal-pilot-{case_id}",
        "session_id": "calibration-session",
        "task_status": "running",
        "task_event_count": event_count,
        "session_event_count": event_count,
        "observed_continuous_batches": continuous_batches,
        "observed_continuous_work_turns": 0,
        "session_resumes": session_resumes,
        "confirmed_constraints": 0,
        "checkpoints": checkpoints,
        "verified_stages": verified_stages or [],
        "active_constraint_texts": [],
        "ready_stages": ready_stages or [],
        "selected_stage": None,
        "stage_assessments": [],
        "task_recovery_blockers": [],
        "ambiguous_effect_call_ids": [],
    }
    path = root / case_id / "observations" / f"{label}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(observation, ensure_ascii=False), encoding="utf-8")
    return path


def test_record_binds_observation_and_rejects_reuse_or_tampering(tmp_path: Path) -> None:
    root = _prepare(tmp_path)
    path = _observe(root, "normal-json-export", "start")
    record_path = protocol.record_boundary(
        root, case_id="normal-json-export", phase="start", observation_label="start",
        reported_suite_tokens=0,
    )
    original_record = record_path.read_bytes()

    with pytest.raises(ValueError, match="already has a start"):
        protocol.record_boundary(
            root, case_id="normal-json-export", phase="start", observation_label="start",
            reported_suite_tokens=0,
        )
    assert record_path.read_bytes() == original_record
    assert len(list((root / "protocol").glob("*.json"))) == 1

    observation = json.loads(path.read_text(encoding="utf-8"))
    observation["task_status"] = "paused"
    path.write_text(json.dumps(observation), encoding="utf-8")
    with pytest.raises(ValueError, match="observation changed"):
        protocol.build_report(root)


def test_budget_review_stops_at_reported_token_ceiling(tmp_path: Path) -> None:
    root = _prepare(tmp_path)
    _observe(root, "normal-json-export", "start")
    protocol.record_boundary(
        root, case_id="normal-json-export", phase="start", observation_label="start",
        reported_suite_tokens=99,
    )
    review = protocol.review_next_batch(root, case_id="normal-json-export")
    assert review["decision"] == "manual_review_required"
    assert review["remaining_reported_tokens"] == 1

    _observe(root, "normal-json-export", "next", event_count=2, continuous_batches=1)
    protocol.record_boundary(
        root, case_id="normal-json-export", phase="after_batch", observation_label="next",
        reported_suite_tokens=100,
    )
    review = protocol.review_next_batch(root, case_id="normal-json-export")
    assert review["decision"] == "stop"
    assert "suite_token_ceiling_reached" in review["reasons"]


def test_final_assessment_is_bound_to_candidate_files(tmp_path: Path) -> None:
    root = _prepare(tmp_path)
    case_id = "normal-json-export"
    _observe(root, case_id, "start")
    protocol.record_boundary(
        root, case_id=case_id, phase="start", observation_label="start",
        reported_suite_tokens=0,
    )
    _observe(root, case_id, "batch", event_count=2, continuous_batches=1)
    protocol.record_boundary(
        root, case_id=case_id, phase="after_batch", observation_label="batch",
        reported_suite_tokens=20,
    )
    workspace = root / case_id / "workspace"
    assessment = {
        "schema_version": 1,
        "suite": "goal-model-pilot-v0",
        "track": "model",
        "case_id": case_id,
        "case_set_digest": load_deck().digest,
        "label": "final",
        "record_type": "external_acceptance",
        "passed": False,
        "candidate_digest": candidate_digest_for_workspace(workspace),
    }
    assessment_path = root / case_id / "assessments" / "final.json"
    assessment_path.parent.mkdir(exist_ok=True)
    assessment_path.write_text(json.dumps(assessment), encoding="utf-8")
    _observe(root, case_id, "final", event_count=3, continuous_batches=1)

    with pytest.raises(ValueError, match="failure attribution"):
        protocol.record_boundary(
            root, case_id=case_id, phase="final", observation_label="final",
            assessment_label="final", reported_suite_tokens=20,
        )
    protocol.record_boundary(
        root, case_id=case_id, phase="final", observation_label="final",
        assessment_label="final", failure_attribution="model_execution",
        reported_suite_tokens=20,
    )
    assert protocol.build_report(root)["cases"][0]["external_acceptance_passed"] is False

    source = workspace / "src" / "cli_data_tool" / "cli.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate changed"):
        protocol.build_report(root)


@pytest.mark.parametrize(
    "hours,mode,day_later", [(23, "day_later", False), (25, "day_later", True), (0, "cross_process", True)]
)
def test_restart_requires_resume_event_and_reports_real_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hours: int, mode: str, day_later: bool,
) -> None:
    root = _prepare(tmp_path)
    case_id = "restart-json-export"
    first_time = datetime(2026, 10, 11, 3, 54, tzinfo=timezone.utc)

    class Clock(datetime):
        current = first_time

        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return cls.current

    monkeypatch.setattr(protocol, "datetime", Clock)
    _observe(root, case_id, "start")
    protocol.record_boundary(
        root, case_id=case_id, phase="start", observation_label="start",
        reported_suite_tokens=0,
    )
    _observe(
        root, case_id, "before", event_count=2, verified_stages=["inspect"],
        ready_stages=["implement"], checkpoints=1,
    )
    protocol.record_boundary(
        root, case_id=case_id, phase="before_restart", observation_label="before",
        process_instance="process-before", restart_acceptance_mode=mode, reported_suite_tokens=20,
    )
    Clock.current = first_time + timedelta(hours=hours)
    _observe(
        root, case_id, "no-resume", event_count=3, verified_stages=["inspect"],
        ready_stages=["implement"], checkpoints=1,
    )
    with pytest.raises(ValueError, match="durable Session resume event"):
        protocol.record_boundary(
            root, case_id=case_id, phase="after_restart", observation_label="no-resume",
            process_instance="process-after", status_card_next_action="Continue implement",
            next_step_accurate=True, reported_suite_tokens=20,
        )

    _observe(
        root, case_id, "after", event_count=3, session_resumes=1,
        verified_stages=["inspect"], ready_stages=["implement"], checkpoints=1,
    )
    protocol.record_boundary(
        root, case_id=case_id, phase="after_restart", observation_label="after",
        process_instance="process-after", status_card_next_action="Continue implement",
        next_step_accurate=True, reported_suite_tokens=20,
    )
    report_case = next(case for case in protocol.build_report(root)["cases"] if case["case_id"] == case_id)
    assert report_case["restart_delay_hours"] == hours
    assert report_case["restart_acceptance_mode"] == mode
    assert ("day_later_resume_evidence" not in report_case["missing_evidence"]) is day_later


@pytest.mark.parametrize("final_checkpoints", [1, 2])
def test_restart_single_turn_execution_requires_work_after_resume(
    tmp_path: Path, final_checkpoints: int,
) -> None:
    root = _prepare(tmp_path)
    case_id = "restart-json-export"
    _observe(root, case_id, "start")
    protocol.record_boundary(
        root, case_id=case_id, phase="start", observation_label="start",
        reported_suite_tokens=0,
    )
    _observe(
        root, case_id, "before", event_count=2, verified_stages=["inspect"],
        ready_stages=["implement"], checkpoints=1,
    )
    protocol.record_boundary(
        root, case_id=case_id, phase="before_restart", observation_label="before",
        process_instance="before", restart_acceptance_mode="cross_process",
        reported_suite_tokens=10,
    )
    _observe(
        root, case_id, "after", event_count=3, session_resumes=1,
        verified_stages=["inspect"], ready_stages=["implement"], checkpoints=1,
    )
    protocol.record_boundary(
        root, case_id=case_id, phase="after_restart", observation_label="after",
        process_instance="after", status_card_next_action="Continue implement",
        next_step_accurate=True, reported_suite_tokens=10,
    )
    workspace = root / case_id / "workspace"
    assessment = {
        "schema_version": 1,
        "suite": "goal-model-pilot-v0",
        "track": "model",
        "case_id": case_id,
        "case_set_digest": load_deck().digest,
        "label": "final",
        "record_type": "external_acceptance",
        "passed": True,
        "candidate_digest": candidate_digest_for_workspace(workspace),
    }
    assessment_path = root / case_id / "assessments" / "final.json"
    assessment_path.parent.mkdir(exist_ok=True)
    assessment_path.write_text(
        json.dumps(assessment), encoding="utf-8",
    )
    _observe(
        root, case_id, "final", event_count=4, session_resumes=1,
        verified_stages=["inspect", "implement"], checkpoints=final_checkpoints,
    )
    protocol.record_boundary(
        root, case_id=case_id, phase="final", observation_label="final",
        assessment_label="final", reported_suite_tokens=20,
    )
    report_case = next(case for case in protocol.build_report(root)["cases"] if case["case_id"] == case_id)
    assert report_case["execution_evidence_mode"] == (
        "post_restart_work_turn" if final_checkpoints == 2 else None
    )
    assert ("continuous_execution_evidence" not in report_case["missing_evidence"]) == (
        final_checkpoints == 2
    )
