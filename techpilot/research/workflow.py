"""Local M1 snapshot store, deterministic evidence queries and report delivery.

Single-writer workflow. All text I/O goes through RuntimeResearchFiles. A failed
multi-file delivery can leave diagnostic files, but never returns success.
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import EvidenceFragment, ReportArtifact, ResearchTask, SourceSnapshot, validate_report
from .rendering import reference_numbers, render_report
from .runtime_files import ResearchIOError, RuntimeResearchFiles


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


@dataclass(frozen=True)
class ReportDelivery:
    report_path: Path
    receipt_path: Path
    content_hash: str
    snapshot_ids: tuple[str, ...]


class ResearchWorkflow:
    def __init__(self, task: ResearchTask, store_directory: Path, files: RuntimeResearchFiles):
        self.task = task
        self.files = files
        self.store_directory = files.checked_path(store_directory)
        files.checked_path(task.artifact_directory)

    def initialize(self) -> None:
        """Persist explicit business facts separately from Runtime Session events."""
        record = asdict(self.task)
        record["artifact_directory"] = str(self.task.artifact_directory)
        path = self.store_directory / "task.json"
        content = _json(record)
        if path.exists():
            if self.files.read(path) != content:
                raise ResearchIOError("Stored research task differs from this workflow")
        else:
            self.files.write_new(path, content)

    def _snapshot_path(self, snapshot_id: str) -> Path:
        if not re.fullmatch(r"snap-[0-9a-f]{64}", snapshot_id):
            raise ResearchIOError("Invalid snapshot identifier")
        return self.store_directory / "snapshots" / f"{snapshot_id}.json"

    def import_source(self, source_id: str, path: Path) -> SourceSnapshot:
        content = self.files.read(path)
        snapshot = SourceSnapshot(source_id, content, datetime.now(timezone.utc))
        target = self._snapshot_path(snapshot.snapshot_id)
        if target.exists():
            return self.load_snapshot(snapshot.snapshot_id)
        self.files.write_new(target, _json({
            "schema_version": 1,
            "source_id": snapshot.source_id,
            "snapshot_id": snapshot.snapshot_id,
            "content_hash": snapshot.content_hash,
            "content": snapshot.content,
            "imported_at": snapshot.imported_at.isoformat(),
            "import_path": str(path.resolve()),
            "text_normalization": "ReadFileTool numbered text decoded to LF with final newline",
        }))
        return self.load_snapshot(snapshot.snapshot_id)

    def load_snapshot(self, snapshot_id: str) -> SourceSnapshot:
        try:
            record = json.loads(self.files.read(self._snapshot_path(snapshot_id)))
            snapshot = SourceSnapshot(record["source_id"], record["content"], datetime.fromisoformat(record["imported_at"]))
            if (record["schema_version"] != 1 or record["snapshot_id"] != snapshot_id
                    or snapshot.snapshot_id != snapshot_id or snapshot.content_hash != record["content_hash"]):
                raise ValueError("snapshot identity/digest mismatch")
            return snapshot
        except (KeyError, TypeError, ValueError) as error:
            raise ResearchIOError(f"Invalid stored snapshot: {snapshot_id}") from error

    def query(self, snapshot_id: str, text: str) -> tuple[EvidenceFragment, ...]:
        """Case-insensitive literal line matching in a specific immutable version."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Evidence query must not be empty")
        snapshot = self.load_snapshot(snapshot_id)
        return tuple(EvidenceFragment.from_snapshot(snapshot, line, line)
                     for line, content in enumerate(snapshot.content.splitlines(), 1)
                     if text.casefold() in content.casefold())

    def deliver(self, report: ReportArtifact, evidence: tuple[EvidenceFragment, ...]) -> ReportDelivery:
        snapshots = tuple(self.load_snapshot(snapshot_id) for snapshot_id in report.snapshot_ids)
        validation = validate_report(self.task, report, snapshots, evidence)
        if not validation.ready_for_delivery:
            raise ResearchIOError("Report validation failed: " + ", ".join(validation.errors))
        receipt_path = report.output_path.with_suffix(".json")
        if receipt_path.exists():
            raise ResearchIOError("Report receipt already exists")
        evidence_map = {fragment.evidence_id: fragment for fragment in evidence}
        used = tuple(evidence_map[ref] for ref in report.evidence_ids)
        content = render_report(self.task, report, used)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.files.write_new(report.output_path, content)
        # A receipt attests validation, not completion: its write can take
        # effect before a later Runtime/persistence failure. Only returning
        # ReportDelivery confirms all steps succeeded.
        self.files.write_new(receipt_path, _json({
            "schema_version": 2, "status": "validated", "task_id": self.task.task_id,
            "title": report.title, "question": self.task.question, "constraints": list(self.task.constraints),
            "candidates": list(self.task.candidates),
            "output_path": str(report.output_path), "content_hash": digest,
            "snapshot_ids": list(report.snapshot_ids), "claims": [asdict(c) for c in report.claims],
            "evidence": [asdict(fragment) for fragment in used], "validation_errors": [],
            "references": [{"number": number, "evidence_id": ref}
                           for ref, number in reference_numbers(self.task, report).items()],
            "snapshot_files": {snapshot_id: Path(os.path.relpath(self._snapshot_path(snapshot_id),
                                                                report.output_path.parent)).as_posix()
                               for snapshot_id in report.snapshot_ids},
        }))
        return ReportDelivery(report.output_path, receipt_path, digest, report.snapshot_ids)
