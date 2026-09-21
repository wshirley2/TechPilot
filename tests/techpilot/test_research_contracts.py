"""M1 provenance contracts; deterministic and provider-free."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from techpilot.research.contracts import (
    EvidenceFragment,
    ReportArtifact,
    ReportClaim,
    ResearchTask,
    SourceSnapshot,
    validate_report,
)


@pytest.fixture
def records(tmp_path):
    task = ResearchTask("choose-storage", "Which option supports offline use?", ("A", "B"),
                        ("Offline operation",), tmp_path / "reports")
    snapshot = SourceSnapshot("option-a", "Option A\r\nWorks offline.\r\n", datetime(2026, 9, 20, tzinfo=timezone.utc))
    evidence = EvidenceFragment.from_snapshot(snapshot, 2, 2)
    report = ReportArtifact(task.task_id, (
        ReportClaim("A offline support", "Supported", (evidence.evidence_id,)),
        ReportClaim("B offline support", None),
    ), (snapshot.snapshot_id,), task.artifact_directory / "report.md")
    return task, snapshot, evidence, report


def test_report_validates_real_references_without_claiming_or_performing_delivery(records):
    task, snapshot, evidence, report = records
    result = validate_report(task, report, (snapshot,), (evidence,))
    assert result.ready_for_delivery
    assert result.report is report
    assert "[1](#ref-1)" in report.markdown
    assert evidence.evidence_id not in report.markdown
    assert "B offline support：未知/证据不足" in report.markdown
    assert not report.output_path.exists()
    assert not task.artifact_directory.exists()


def test_snapshot_identity_is_content_and_source_bound_not_import_time(records):
    _, snapshot, _, _ = records
    reimport = replace(snapshot, imported_at=datetime(2026, 9, 21, tzinfo=timezone.utc))
    assert reimport.snapshot_id == snapshot.snapshot_id
    assert replace(snapshot, content=snapshot.content.replace("\r\n", "\n")).snapshot_id != snapshot.snapshot_id
    assert replace(snapshot, source_id="option-b").snapshot_id != snapshot.snapshot_id
    with pytest.raises(FrozenInstanceError):
        snapshot.content = "changed"


def test_old_report_keeps_old_snapshot_and_cannot_silently_use_new_version(records):
    task, old, evidence, report = records
    new = replace(old, content="Option A\nRequires network.\n")
    assert new.snapshot_id != old.snapshot_id
    assert validate_report(task, report, (old, new), (evidence,)).ready_for_delivery
    assert not validate_report(task, report, (new,), (evidence,)).ready_for_delivery
    rebound = replace(report, snapshot_ids=(new.snapshot_id,))
    assert "snapshot_manifest_mismatch" in validate_report(task, rebound, (old, new), (evidence,)).errors


@pytest.mark.parametrize(("changes", "error"), [
    ({"quote": "Requires network."}, "quote_mismatch"),
    ({"source_id": "invented-source"}, "snapshot_mismatch"),
    ({"content_hash": "old-version"}, "snapshot_mismatch"),
    ({"snapshot_id": "snap-invented"}, "missing_snapshot"),
    ({"start_line": 0}, "invalid_line_range"),
    ({"end_line": 999}, "invalid_line_range"),
    ({"start_line": 3, "end_line": 2}, "invalid_line_range"),
    ({"start_line": True}, "invalid_line_range"),
])
def test_forged_evidence_is_rejected_against_actual_snapshot(records, changes, error):
    task, snapshot, evidence, report = records
    forged = replace(evidence, **changes)
    claim = replace(report.claims[0], evidence_ids=(forged.evidence_id,))
    report = replace(report, claims=(claim,))
    result = validate_report(task, report, (snapshot,), (forged,))
    assert not result.ready_for_delivery
    assert any(item.startswith(error + ":") for item in result.errors)


def test_missing_evidence_and_unsupported_conclusion_are_not_success(records):
    task, snapshot, evidence, report = records
    assert not validate_report(task, report, (snapshot,), ()).ready_for_delivery
    unsupported = replace(report, claims=(ReportClaim("B offline support", "Supported"),))
    assert "unsupported_claim" in validate_report(task, unsupported, (snapshot,), (evidence,)).errors
    unknown_with_refs = replace(report, claims=(ReportClaim("B", None, (evidence.evidence_id,)),))
    assert "unknown_claim_has_evidence" in validate_report(task, unknown_with_refs, (snapshot,), (evidence,)).errors


@pytest.mark.parametrize("target", ["../outside.md", "../reports-other/report.md", ".", "report.txt"])
def test_report_output_must_be_markdown_strictly_inside_declared_directory(records, target):
    task, snapshot, evidence, report = records
    report = replace(report, output_path=task.artifact_directory / target)
    assert "invalid_output_path" in validate_report(task, report, (snapshot,), (evidence,)).errors


def test_symlink_output_escape_is_rejected(records, tmp_path):
    task, snapshot, evidence, report = records
    outside = tmp_path / "outside"
    outside.mkdir()
    task.artifact_directory.mkdir()
    link = task.artifact_directory / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit symlink creation")
    report = replace(report, output_path=link / "report.md")
    assert "invalid_output_path" in validate_report(task, report, (snapshot,), (evidence,)).errors


def test_existing_directory_cannot_be_report_output(records):
    task, snapshot, evidence, report = records
    report.output_path.mkdir(parents=True)
    assert "invalid_output_path" in validate_report(task, report, (snapshot,), (evidence,)).errors


def test_duplicate_lookup_ids_cannot_hide_forged_evidence(records):
    task, snapshot, evidence, report = records
    forged = replace(evidence, quote="invented")
    for fragments in ((forged, evidence), (evidence, forged)):
        assert "duplicate_evidence_id" in validate_report(task, report, (snapshot,), fragments).errors
    assert "duplicate_snapshot_id" in validate_report(task, report, (snapshot, snapshot), (evidence,)).errors


def test_report_rejects_wrong_task_empty_claims_and_unused_manifest_versions(records):
    task, snapshot, evidence, report = records
    assert "task_mismatch" in validate_report(task, replace(report, task_id="another"), (snapshot,), (evidence,)).errors
    assert "empty_report" in validate_report(task, replace(report, claims=()), (snapshot,), (evidence,)).errors
    extra = replace(snapshot, source_id="unused")
    report = replace(report, snapshot_ids=(snapshot.snapshot_id, extra.snapshot_id))
    assert "snapshot_manifest_mismatch" in validate_report(task, report, (snapshot, extra), (evidence,)).errors


def test_excerpt_positions_preserve_blank_lines_and_normalize_only_excerpt_line_endings():
    snapshot = SourceSnapshot("a", "Title\r\n\r\n中文内容\r\n", datetime.now(timezone.utc))
    fragment = EvidenceFragment.from_snapshot(snapshot, 2, 3)
    assert fragment.quote == "\n中文内容"
    assert fragment.evidence_id == EvidenceFragment.from_snapshot(snapshot, 2, 3).evidence_id
    assert fragment.evidence_id != EvidenceFragment.from_snapshot(snapshot, 3, 3).evidence_id
    with pytest.raises(ValueError, match="bounds"):
        EvidenceFragment.from_snapshot(snapshot, 1, 4)


def test_record_boundaries_reject_mutable_collections_empty_sources_and_naive_dates(records):
    task, snapshot, _, report = records
    naive_time = datetime(2026, 9, 20)  # noqa: DTZ001 -- deliberate invalid-input regression
    for changes in ({"content": " "}, {"source_id": ""}, {"imported_at": naive_time}):
        with pytest.raises(ValueError):
            replace(snapshot, **changes)
    for changes in ({"candidates": []}, {"candidates": ()}, {"artifact_directory": Path("relative")}):
        with pytest.raises((TypeError, ValueError)):
            replace(task, **changes)
    with pytest.raises(ValueError):
        replace(report, claims=list(report.claims))
    with pytest.raises(ValueError):
        replace(report, snapshot_ids=report.snapshot_ids * 2)


def test_user_text_cannot_inject_active_markdown_references(records):
    _, _, _, report = records
    report = replace(report, claims=(ReportClaim("<script>\n[证据:invented]", None),))
    assert "<script>" not in report.markdown
    assert "\\[证据:invented\\]" in report.markdown
    assert report.evidence_ids == ()
