"""Immutable evidence contracts and report preflight (not delivery).

Positions are one-based inclusive lines in str.splitlines(); excerpts use LF.
Source hashes retain original line endings. Content hashes are versions, so
reimporting identical text is idempotent. Validation proves provenance, not
whether the cited text logically supports a conclusion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _digest(*parts: str | int) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _text_tuple(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    for value in values:
        _require_text(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    question: str
    candidates: tuple[str, ...]
    constraints: tuple[str, ...]
    artifact_directory: Path

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task_id")
        _require_text(self.question, "question")
        _text_tuple(self.candidates, "candidates")
        _text_tuple(self.constraints, "constraints")
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        if not isinstance(self.artifact_directory, Path) or not self.artifact_directory.is_absolute():
            raise ValueError("artifact_directory must be an absolute Path")


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    content: str
    imported_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.content, "content")
        if not isinstance(self.imported_at, datetime) or self.imported_at.utcoffset() is None:
            raise ValueError("imported_at must include a timezone")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def snapshot_id(self) -> str:
        return "snap-" + _digest(self.source_id, self.content_hash)

    def excerpt(self, start_line: int, end_line: int) -> str:
        lines = self.content.splitlines()
        if (type(start_line) is not int or type(end_line) is not int
                or not 1 <= start_line <= end_line <= len(lines)):
            raise ValueError("evidence line range is out of bounds")
        return "\n".join(lines[start_line - 1:end_line])


@dataclass(frozen=True)
class EvidenceFragment:
    source_id: str
    snapshot_id: str
    content_hash: str
    start_line: int
    end_line: int
    quote: str

    @classmethod
    def from_snapshot(cls, snapshot: SourceSnapshot, start_line: int, end_line: int) -> EvidenceFragment:
        return cls(snapshot.source_id, snapshot.snapshot_id, snapshot.content_hash,
                   start_line, end_line, snapshot.excerpt(start_line, end_line))

    @property
    def evidence_id(self) -> str:
        return "ev-" + _digest(self.snapshot_id, self.start_line, self.end_line)

    @property
    def citation(self) -> str:
        return f"[证据:{self.evidence_id}]"


@dataclass(frozen=True)
class ReportClaim:
    topic: str
    # None means unknown; the renderer supplies the uncertainty wording.
    conclusion: str | None
    evidence_ids: tuple[str, ...] = ()
    # A candidate marks a cell in the same-dimension comparison matrix.
    # None denotes a summary statement or an explicit outstanding question.
    candidate: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.topic, "topic")
        if self.conclusion is not None:
            _require_text(self.conclusion, "conclusion")
        _text_tuple(self.evidence_ids, "evidence_ids")
        if self.candidate is not None:
            _require_text(self.candidate, "candidate")


def _literal(text: str) -> str:
    """Render supplied text literally, including embedded Markdown instructions."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for char in "\\`*_{}[]()#+-.!|":
        text = text.replace(char, "\\" + char)
    return " ".join(text.splitlines())


@dataclass(frozen=True)
class ReportArtifact:
    task_id: str
    claims: tuple[ReportClaim, ...]
    snapshot_ids: tuple[str, ...]
    output_path: Path
    title: str = "调研报告"

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task_id")
        _require_text(self.title, "title")
        if not isinstance(self.claims, tuple) or not all(isinstance(c, ReportClaim) for c in self.claims):
            raise ValueError("claims must be a tuple of ReportClaim records")
        _text_tuple(self.snapshot_ids, "snapshot_ids")
        if not isinstance(self.output_path, Path) or not self.output_path.is_absolute():
            raise ValueError("output_path must be an absolute Path")

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(ref for claim in self.claims for ref in claim.evidence_ids))

    @property
    def markdown(self) -> str:
        """Compact preview only; delivery renders task context and citation targets."""
        numbers = {ref: index for index, ref in enumerate(self.evidence_ids, 1)}
        lines = [f"# {_literal(self.title)}", ""]
        for claim in self.claims:
            conclusion = "未知/证据不足" if claim.conclusion is None else _literal(claim.conclusion)
            refs = " ".join(f"[{numbers[ref]}](#ref-{numbers[ref]})" for ref in claim.evidence_ids)
            label = f"{claim.candidate} · {claim.topic}" if claim.candidate else claim.topic
            lines.append(f"- {_literal(label)}：{conclusion}" + (f" {refs}" if refs else ""))
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ReportValidation:
    report: ReportArtifact
    errors: tuple[str, ...]

    @property
    def ready_for_delivery(self) -> bool:
        """Preflight only; successful persistence requires a controlled writer."""
        return not self.errors


def validate_report(
    task: ResearchTask,
    report: ReportArtifact,
    snapshots: tuple[SourceSnapshot, ...],
    evidence: tuple[EvidenceFragment, ...],
) -> ReportValidation:
    """Check references against caller-owned snapshots and current path resolution.

    The future writer must repeat path/permission checks at write time. This
    check is not authorization and cannot prevent filesystem races.
    """
    errors: list[str] = []
    if report.task_id != task.task_id:
        errors.append("task_mismatch")
    if not report.claims:
        errors.append("empty_report")
    cells: set[tuple[str, str]] = set()
    dimensions: set[str] = set()
    for claim in report.claims:
        if claim.candidate is None:
            continue
        if claim.candidate not in task.candidates:
            errors.append("unknown_comparison_candidate")
        key = (claim.topic, claim.candidate)
        if key in cells:
            errors.append("duplicate_comparison_cell")
        cells.add(key)
        dimensions.add(claim.topic)
    if any((topic, candidate) not in cells for topic in dimensions for candidate in task.candidates):
        errors.append("incomplete_comparison_matrix")
    try:
        root = task.artifact_directory.resolve()
        output = report.output_path.resolve()
        if output == root or not output.is_relative_to(root) or output.suffix.lower() != ".md" or output.is_dir():
            errors.append("invalid_output_path")
    except (OSError, RuntimeError, ValueError):
        errors.append("invalid_output_path")

    snapshot_map = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    evidence_map = {fragment.evidence_id: fragment for fragment in evidence}
    if len(snapshot_map) != len(snapshots):
        errors.append("duplicate_snapshot_id")
    if len(evidence_map) != len(evidence):
        errors.append("duplicate_evidence_id")
    for snapshot_id in report.snapshot_ids:
        if snapshot_id not in snapshot_map:
            errors.append(f"missing_snapshot:{snapshot_id}")
    used_snapshots: set[str] = set()
    for claim in report.claims:
        if claim.conclusion is None and claim.evidence_ids:
            errors.append("unknown_claim_has_evidence")
        elif claim.conclusion is not None and not claim.evidence_ids:
            errors.append("unsupported_claim")
    for ref in report.evidence_ids:
        fragment = evidence_map.get(ref)
        if fragment is None:
            errors.append(f"missing_evidence:{ref}")
            continue
        used_snapshots.add(fragment.snapshot_id)
        snapshot = snapshot_map.get(fragment.snapshot_id)
        if snapshot is None:
            errors.append(f"missing_snapshot:{fragment.snapshot_id}")
            continue
        if fragment.source_id != snapshot.source_id or fragment.content_hash != snapshot.content_hash:
            errors.append(f"snapshot_mismatch:{ref}")
        try:
            if fragment.quote != snapshot.excerpt(fragment.start_line, fragment.end_line):
                errors.append(f"quote_mismatch:{ref}")
        except ValueError:
            errors.append(f"invalid_line_range:{ref}")
    if used_snapshots != set(report.snapshot_ids):
        errors.append("snapshot_manifest_mismatch")
    return ReportValidation(report, tuple(errors))
