"""Host-configured research file operations for real research Tool Calls.

Unlike ``RuntimeResearchFiles``, this service never creates an Agent or a
provider.  It reuses the native page reader and writer directly after the host
has fixed every readable/writable root for one research task.
"""

from __future__ import annotations

from pathlib import Path

from techpilot.engine.tool_results import ToolStatus
from techpilot.engine.tools import ReadFileTool, WriteFileTool
from techpilot.engine.tools.read import _page_footer
from techpilot.safety.paths import is_sensitive_path

from .runtime_files import MAX_IMPORT_BYTES, MAX_IMPORT_LINES, PAGE_LINE_LIMIT, ResearchIOError


class HostResearchFiles:
    """Read approved sources and write only task-owned research artifacts."""

    def __init__(
        self,
        repository: Path,
        *,
        source_root: Path,
        store_directory: Path,
        artifact_directory: Path,
    ) -> None:
        self.repository = repository.resolve()
        self.source_root = self._checked_root(source_root)
        self.store_directory = self._checked_root(store_directory)
        self.artifact_directory = self._checked_root(artifact_directory)
        self._reader = ReadFileTool()
        self._writer = WriteFileTool()

    def _checked_root(self, value: Path) -> Path:
        resolved = value.resolve()
        if not resolved.is_relative_to(self.repository) or is_sensitive_path(resolved):
            raise ResearchIOError("Research root is outside the repository or sensitive")
        return resolved

    def checked_path(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.repository) or is_sensitive_path(resolved):
            raise ResearchIOError("Research path is outside the repository or sensitive")
        return resolved

    def read(self, path: Path) -> str:
        path = self.checked_path(path)
        if not any(path.is_relative_to(root) for root in (self.source_root, self.store_directory, self.artifact_directory)):
            raise ResearchIOError("Research read path is outside the host-approved roots")
        expected_hash: str | None = None
        expected_total: int | None = None
        offset = 1
        text: list[str] = []
        while True:
            arguments = {"file_path": str(path), "offset": offset, "limit": PAGE_LINE_LIMIT}
            if expected_hash is not None:
                arguments["expected_content_hash"] = expected_hash
            outcome = self._reader.execute(**arguments)
            if outcome.status is not ToolStatus.COMPLETED:
                raise ResearchIOError("Research page read did not complete")
            facts = outcome.facts.to_payload()
            content_hash = _string(facts, "content_hash")
            total_lines = _non_negative_int(facts, "total_lines")
            content_bytes = _non_negative_int(facts, "content_bytes")
            line_start = _positive_or_none(facts, "line_start")
            line_end = _positive_or_none(facts, "line_end")
            next_offset = _positive_or_none(facts, "next_offset")
            if (
                content_hash is None or total_lines is None or content_bytes is None
                or facts.get("encoding") != "utf-8" or facts.get("decoding_replaced") is not False
            ):
                raise ResearchIOError("Research page facts are incomplete")
            if total_lines > MAX_IMPORT_LINES or content_bytes > MAX_IMPORT_BYTES:
                raise ResearchIOError("Research source exceeds the configured budget")
            if expected_hash is None:
                expected_hash, expected_total = content_hash, total_lines
            elif content_hash != expected_hash or total_lines != expected_total:
                raise ResearchIOError("Research source changed while reading pages")
            if total_lines == 0 or line_start != offset or line_end is None or line_end < line_start:
                raise ResearchIOError("Research source page is incomplete")
            text.extend(_decode_page(str(outcome), line_start, line_end, total_lines, next_offset, content_hash))
            if next_offset is None:
                if line_end != total_lines or len(text) != total_lines:
                    raise ResearchIOError("Research source pages did not cover the complete file")
                return "\n".join(text) + "\n"
            if next_offset != line_end + 1:
                raise ResearchIOError("Research source page has an invalid next offset")
            offset = next_offset

    def write_new(self, path: Path, content: str) -> None:
        path = self.checked_path(path)
        if not any(path.is_relative_to(root) for root in (self.store_directory, self.artifact_directory)):
            raise ResearchIOError("Research write path is outside the host-owned roots")
        if path.exists():
            raise ResearchIOError("Refusing to replace an existing research artifact")
        outcome = self._writer.execute(file_path=str(path), content=content)
        if outcome.status is not ToolStatus.COMPLETED or self.read(path) != content:
            raise ResearchIOError("Persisted research artifact differs from the approved content")


def _string(facts: dict[str, object], name: str) -> str | None:
    value = facts.get(name)
    return value if isinstance(value, str) else None


def _non_negative_int(facts: dict[str, object], name: str) -> int | None:
    value = facts.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _positive_or_none(facts: dict[str, object], name: str) -> int | None:
    value = facts.get(name)
    if value is None:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def _decode_page(result: str, start: int, end: int, total: int, next_offset: int | None, digest: str) -> list[str]:
    rows = result.splitlines()
    footer = _page_footer(
        line_start=start,
        line_end=end,
        total_lines=total,
        next_offset=next_offset,
        content_hash=digest,
    )
    if len(rows) != end - start + 2 or rows[-1] != footer:
        raise ResearchIOError("Research source page text does not match page facts")
    content: list[str] = []
    for index, row in enumerate(rows[:-1], start):
        prefix, separator, body = row.partition("\t")
        if not separator or prefix != str(index):
            raise ResearchIOError("Research source page line numbers are incomplete")
        content.append(body)
    return content
