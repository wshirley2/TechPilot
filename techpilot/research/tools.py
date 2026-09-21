"""Read-only Tool wrappers over immutable M1 research snapshots.

The host creates ``ResearchToolService`` with one already-initialized workflow.
These tools never create a provider, import files, or choose writable roots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect
from techpilot.engine.tool_results import ToolResult, ToolStatus
from techpilot.engine.tools.base import Tool

from .contracts import EvidenceFragment, ReportArtifact, ReportClaim
from .runtime_files import ResearchIOError
from .workflow import ResearchWorkflow

MAX_QUERY_PAGE = 20
MAX_CONTEXT_LINES = 100
MAX_CONTEXT_CHARS = 12_000


@dataclass
class ResearchToolService:
    """Session-local evidence ledger for one host-owned research workflow."""

    workflow: ResearchWorkflow
    source_root: Path | None = None
    _evidence: dict[str, EvidenceFragment] = field(default_factory=dict)

    def __post_init__(self) -> None:
        root = self.source_root
        if root is None:
            root = getattr(self.workflow.files, "source_root", None)
        if not isinstance(root, Path):
            raise TypeError("Research tool service requires a host-approved source_root")
        self.source_root = root.resolve()

    def import_source(self, source_id: str, source_path: str) -> str:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be non-empty text")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("source_path must be non-empty text")
        requested = Path(source_path)
        if requested.is_absolute() or ".." in requested.parts:
            raise ValueError("source_path must be a relative path inside the approved source root")
        path = (self.source_root / requested).resolve()
        if not path.is_relative_to(self.source_root) or not path.is_file():
            raise ValueError("source_path is not an approved source file")
        snapshot = self.workflow.import_source(source_id, path)
        return _json({
            "source_id": snapshot.source_id,
            "snapshot_id": snapshot.snapshot_id,
            "content_hash": snapshot.content_hash,
            "total_lines": len(snapshot.content.splitlines()),
        })

    def query(self, snapshot_id: str, text: str, *, offset: int = 1, limit: int = 10) -> str:
        _positive(offset, "offset")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_QUERY_PAGE:
            raise ValueError(f"limit must be between 1 and {MAX_QUERY_PAGE}")
        matches = self.workflow.query(snapshot_id, text)
        page = matches[offset - 1:offset - 1 + limit]
        for fragment in page:
            self._evidence[fragment.evidence_id] = fragment
        next_offset = offset + len(page) if offset + len(page) <= len(matches) else None
        return _json({
            "snapshot_id": snapshot_id,
            "query": text,
            "total_matches": len(matches),
            "offset": offset,
            "next_offset": next_offset,
            "matches": [_preview(fragment) for fragment in page],
        })

    def read(self, snapshot_id: str, *, start_line: int, end_line: int) -> str:
        _positive(start_line, "start_line")
        _positive(end_line, "end_line")
        if end_line < start_line or end_line - start_line + 1 > MAX_CONTEXT_LINES:
            raise ValueError(f"context must contain 1 to {MAX_CONTEXT_LINES} lines")
        snapshot = self.workflow.load_snapshot(snapshot_id)
        fragment = EvidenceFragment.from_snapshot(snapshot, start_line, end_line)
        if len(fragment.quote) > MAX_CONTEXT_CHARS:
            raise ValueError(f"context exceeds {MAX_CONTEXT_CHARS} characters; request a smaller range")
        self._evidence[fragment.evidence_id] = fragment
        return _json({
            "snapshot_id": fragment.snapshot_id,
            "content_hash": fragment.content_hash,
            "evidence_id": fragment.evidence_id,
            "start_line": fragment.start_line,
            "end_line": fragment.end_line,
            "quote": fragment.quote,
        })

    def submit_report(
        self,
        claims: list[object],
        snapshot_ids: list[object],
        output_name: str,
        title: str = "调研报告",
    ) -> str:
        if not isinstance(output_name, str) or Path(output_name).name != output_name or not output_name.endswith(".md"):
            raise ValueError("output_name must be a new Markdown filename without a path")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be non-empty text")
        parsed_claims = tuple(_claim(value) for value in claims)
        parsed_snapshots = _strings(snapshot_ids, "snapshot_ids")
        evidence_ids = tuple(dict.fromkeys(ref for claim in parsed_claims for ref in claim.evidence_ids))
        missing = [ref for ref in evidence_ids if ref not in self._evidence]
        if missing:
            raise ValueError("report references evidence not issued by this research session")
        report = ReportArtifact(
            self.workflow.task.task_id,
            parsed_claims,
            parsed_snapshots,
            self.workflow.task.artifact_directory / output_name,
            title,
        )
        delivery = self.workflow.deliver(report, tuple(self._evidence[ref] for ref in evidence_ids))
        return _json({
            "report_path": str(delivery.report_path),
            "receipt_path": str(delivery.receipt_path),
            "content_hash": delivery.content_hash,
            "snapshot_ids": list(delivery.snapshot_ids),
        })


class ResearchQueryEvidenceTool(Tool):
    name = "research_query_evidence"
    description = "Search one immutable research snapshot and return paged evidence locators."
    parameters = {
        "type": "object",
        "properties": {
            "snapshot_id": {"type": "string", "minLength": 1},
            "query": {"type": "string", "minLength": 1},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_QUERY_PAGE},
        },
        "required": ["snapshot_id", "query"],
    }
    execution_effect = ToolEffect.READ
    execution_concurrency = ToolConcurrency.SAFE

    def __init__(self, service: ResearchToolService):
        self.service = service

    def execution_resources(self, arguments: dict[str, object]) -> tuple[str, ...]:
        snapshot_id = arguments.get("snapshot_id")
        return (f"research-snapshot:{snapshot_id}",) if isinstance(snapshot_id, str) else ()

    def execute(self, snapshot_id: str, query: str, offset: int = 1, limit: int = 10) -> str:
        try:
            return ToolResult(self.service.query(snapshot_id, query, offset=offset, limit=limit))
        except (ResearchIOError, ValueError) as error:
            return ToolResult(f"Error: research query failed: {error}", ToolStatus.ERROR)


class ResearchImportSourceTool(Tool):
    name = "research_import_source"
    description = "Import one host-approved local source into a new immutable research snapshot."
    parameters = {
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "source_path": {"type": "string", "minLength": 1},
        },
        "required": ["source_id", "source_path"],
    }
    execution_effect = ToolEffect.WRITE
    execution_concurrency = ToolConcurrency.EXCLUSIVE

    def __init__(self, service: ResearchToolService):
        self.service = service

    def execute(self, source_id: str, source_path: str) -> str:
        try:
            return ToolResult(self.service.import_source(source_id, source_path))
        except (ResearchIOError, ValueError) as error:
            return ToolResult(
                f"[effect unknown] Error: research import failed: {error}; inspect snapshots before retrying.",
                ToolStatus.EFFECT_UNKNOWN,
            )


class ResearchReadEvidenceTool(Tool):
    name = "research_read_evidence"
    description = "Read a bounded line range from one immutable research snapshot."
    parameters = {
        "type": "object",
        "properties": {
            "snapshot_id": {"type": "string", "minLength": 1},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["snapshot_id", "start_line", "end_line"],
    }
    execution_effect = ToolEffect.READ
    execution_concurrency = ToolConcurrency.SAFE

    def __init__(self, service: ResearchToolService):
        self.service = service

    def execution_resources(self, arguments: dict[str, object]) -> tuple[str, ...]:
        snapshot_id = arguments.get("snapshot_id")
        return (f"research-snapshot:{snapshot_id}",) if isinstance(snapshot_id, str) else ()

    def execute(self, snapshot_id: str, start_line: int, end_line: int) -> str:
        try:
            return ToolResult(self.service.read(snapshot_id, start_line=start_line, end_line=end_line))
        except (ResearchIOError, ValueError) as error:
            return ToolResult(f"Error: research context read failed: {error}", ToolStatus.ERROR)


class ResearchSubmitReportTool(Tool):
    name = "research_submit_report"
    description = "Validate issued evidence and create one new report in the host-owned artifact directory."
    parameters = {
        "type": "object",
        "properties": {
            "claims": {"type": "array"},
            "snapshot_ids": {"type": "array"},
            "output_name": {"type": "string", "minLength": 4},
            "title": {"type": "string", "minLength": 1},
        },
        "required": ["claims", "snapshot_ids", "output_name"],
    }
    execution_effect = ToolEffect.WRITE
    execution_concurrency = ToolConcurrency.EXCLUSIVE

    def __init__(self, service: ResearchToolService):
        self.service = service

    def execute(self, claims: list[object], snapshot_ids: list[object], output_name: str, title: str = "调研报告") -> str:
        try:
            return ToolResult(self.service.submit_report(claims, snapshot_ids, output_name, title))
        except (ResearchIOError, TypeError, ValueError) as error:
            return ToolResult(
                f"[effect unknown] Error: research report submission failed: {error}; inspect artifacts before retrying.",
                ToolStatus.EFFECT_UNKNOWN,
            )


def read_only_research_tools(service: ResearchToolService) -> list[Tool]:
    """Return the immutable-snapshot tools that a host may add to one Runtime."""

    return [ResearchQueryEvidenceTool(service), ResearchReadEvidenceTool(service)]


def research_tools(service: ResearchToolService) -> list[Tool]:
    """Return the complete host-injected minimal research toolset."""

    return [
        ResearchImportSourceTool(service),
        ResearchQueryEvidenceTool(service),
        ResearchReadEvidenceTool(service),
        ResearchSubmitReportTool(service),
    ]


def _positive(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _preview(fragment: EvidenceFragment) -> dict[str, object]:
    quote = fragment.quote
    if len(quote) > 400:
        quote = quote[:397] + "..."
    return {
        "evidence_id": fragment.evidence_id,
        "start_line": fragment.start_line,
        "end_line": fragment.end_line,
        "quote_preview": quote,
    }


def _claim(value: object) -> ReportClaim:
    if not isinstance(value, dict) or set(value) - {"topic", "conclusion", "evidence_ids", "candidate"}:
        raise ValueError("each claim must be an object with topic, conclusion, evidence_ids and optional candidate")
    topic = value.get("topic")
    conclusion = value.get("conclusion")
    candidate = value.get("candidate")
    if not isinstance(topic, str) or not isinstance(conclusion, (str, type(None))) or not isinstance(candidate, (str, type(None))):
        raise TypeError("claim topic/conclusion/candidate fields have invalid types")
    return ReportClaim(topic, conclusion, _strings(value.get("evidence_ids"), "evidence_ids"), candidate)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must be a non-empty list of text")
    values = tuple(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _json(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
