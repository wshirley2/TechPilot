"""Reader-facing structure must preserve the machine-validated provenance."""

import re
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from techpilot.research.contracts import (
    EvidenceFragment,
    ReportArtifact,
    ReportClaim,
    ResearchTask,
    SourceSnapshot,
    validate_report,
)
from techpilot.research.rendering import render_report


@pytest.fixture
def comparison(tmp_path):
    task = ResearchTask("compare", "Which option?", ("A", "B"), ("Small project",), tmp_path / "reports")
    snapshot = SourceSnapshot("https://example.com/docs", "A supports functions.\nB uses classes.\n",
                              datetime.now(timezone.utc))
    a, b = (EvidenceFragment.from_snapshot(snapshot, line, line) for line in (1, 2))
    report = ReportArtifact(task.task_id, (
        ReportClaim("Prefer functions", "Try A under this preference", (a.evidence_id,)),
        ReportClaim("Style", "Functions", (a.evidence_id,), "A"),
        ReportClaim("Style", "Classes", (b.evidence_id,), "B"),
        ReportClaim("Commercial support", None),
    ), (snapshot.snapshot_id,), task.artifact_directory / "comparison.md", title="Readable comparison")
    return task, snapshot, (a, b), report


def test_readable_report_has_task_matrix_unknowns_and_no_internal_hashes(comparison):
    task, snapshot, evidence, report = comparison
    assert validate_report(task, report, (snapshot,), evidence).ready_for_delivery
    text = render_report(task, report, evidence)
    for heading in ("问题与约束", "结论摘要", "同维度比较", "未知与待确认事项", "资料与引用"):
        assert f"## {heading}" in text
    assert task.question in text and task.constraints[0] in text
    assert "| Style | Functions [1](#ref-1) | Classes [2](#ref-2) |" in text
    assert "未知/证据不足" in text
    assert "snap-" not in text and "ev-" not in text and snapshot.content_hash not in text
    assert "[查看来源文档](https://example.com/docs)" in text
    assert "[完整校验记录](comparison.json)" in text


def test_every_short_reference_resolves_once_and_repeated_evidence_reuses_number(comparison):
    task, _, evidence, report = comparison
    text = render_report(task, report, evidence)
    assert text == render_report(task, report, tuple(reversed(evidence)))
    refs = re.findall(r"\[(\d+)\]\(#ref-(\d+)\)", text)
    assert all(label == target for label, target in refs)
    assert text.count("[1](#ref-1)") == 2
    for number in {target for _, target in refs}:
        assert text.count(f'<a id="ref-{number}"></a>') == 1


def test_numbers_follow_visible_order_even_when_records_are_candidate_grouped(comparison):
    task, snapshot, _, report = comparison
    snapshot = replace(snapshot, content=snapshot.content + "A discover.\nB discover.\n")
    fragments = tuple(EvidenceFragment.from_snapshot(snapshot, line, line) for line in range(1, 5))
    a, b, c, d = fragments
    report = replace(report, snapshot_ids=(snapshot.snapshot_id,), claims=(
        ReportClaim("Style", "Functions", (a.evidence_id,), "A"),
        ReportClaim("Discovery", "A discovery", (c.evidence_id,), "A"),
        ReportClaim("Style", "Classes", (b.evidence_id,), "B"),
        ReportClaim("Discovery", "B discovery", (d.evidence_id,), "B"),
    ))
    text = render_report(task, report, fragments)
    assert re.findall(r"\[(\d+)\]\(#ref-\d+\)", text) == ["1", "2", "3", "4"]
    assert "### 引用 2 · B" in text


@pytest.mark.parametrize(("change", "error"), [
    ("missing", "incomplete_comparison_matrix"),
    ("duplicate", "duplicate_comparison_cell"),
    ("wrong_candidate", "unknown_comparison_candidate"),
])
def test_comparison_cannot_hide_missing_or_ambiguous_cells(comparison, change, error):
    task, snapshot, evidence, report = comparison
    claims = list(report.claims)
    if change == "missing":
        del claims[2]
    elif change == "duplicate":
        claims.append(claims[1])
    else:
        claims[2] = replace(claims[2], candidate="C")
    result = validate_report(task, replace(report, claims=tuple(claims)), (snapshot,), evidence)
    assert error in result.errors


def test_unknown_comparison_cell_is_explicit_and_has_no_fake_citation(comparison):
    task, snapshot, evidence, report = comparison
    claims = list(report.claims)
    claims[2] = replace(claims[2], conclusion=None, evidence_ids=())
    report = replace(report, claims=tuple(claims))
    assert validate_report(task, report, (snapshot,), evidence).ready_for_delivery
    text = render_report(task, report, evidence)
    assert "| Style | Functions [1](#ref-1) | 未知/证据不足 |" in text
    assert "#ref-2" not in text


def test_untrusted_text_cannot_break_table_or_inject_links_and_anchors(comparison):
    task, snapshot, _, report = comparison
    source = replace(snapshot, source_id="javascript:alert(1)", content='<a id="ref-99">fake</a>\n')
    fragment = EvidenceFragment.from_snapshot(source, 1, 1)
    report = replace(report, claims=(ReportClaim("x|y\n<script>", fragment.quote, (fragment.evidence_id,)),),
                     snapshot_ids=(source.snapshot_id,), title="<script>title</script>")
    text = render_report(task, report, (fragment,))
    assert '<a id="ref-99">' not in text and "<script>" not in text
    assert "x\\|y" in text and "](javascript:" not in text
    source = replace(source, source_id="https://example.com/x)[link](javascript:bad)")
    fragment = EvidenceFragment.from_snapshot(source, 1, 1)
    report = replace(report, claims=(ReportClaim("safe", "literal", (fragment.evidence_id,)),))
    text = render_report(task, report, (fragment,))
    assert "%29%5Blink%5D%28" in text and "](javascript:" not in text
