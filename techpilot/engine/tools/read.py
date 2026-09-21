"""Version-bound, line-paginated file reading."""

import hashlib
from pathlib import Path

from techpilot.safety.paths import is_sensitive_path

from ..tool_results import ToolResult, ToolResultFacts, ToolStatus
from ..tool_validation import validated_tool
from .base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file page with line numbers and a source SHA-256. "
        "For later pages, pass the prior expected_content_hash so changed files are not mixed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file",
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "Start line (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Max lines to read. Default 2000.",
            },
            "expected_content_hash": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "SHA-256 returned by an earlier page; rejects changed source content.",
            },
        },
        "required": ["file_path"],
    }

    @validated_tool
    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int = 2000,
        expected_content_hash: str | None = None,
    ) -> str:
        try:
            if is_sensitive_path(Path(file_path).expanduser()):
                return ToolResult("Permission denied read_file: sensitive file", ToolStatus.DENIED)
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            raw = p.read_bytes()
            content_hash = hashlib.sha256(raw).hexdigest()
            try:
                text = raw.decode("utf-8")
                decoding_replaced = False
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
                decoding_replaced = True
            lines = text.splitlines()
            total = len(lines)

            if expected_content_hash is not None and expected_content_hash != content_hash:
                return ToolResult(
                    "Error: source content changed since the previous page; re-read from offset 1.",
                    ToolStatus.ERROR,
                    facts=ToolResultFacts(
                        content_hash=content_hash,
                        content_bytes=len(raw),
                        encoding="utf-8",
                        decoding_replaced=decoding_replaced,
                        total_lines=total,
                    ),
                )

            start = max(0, offset - 1)
            chunk = lines[start : start + limit]
            numbered = [f"{start + i + 1}\t{ln}" for i, ln in enumerate(chunk)]
            line_start = start + 1 if chunk else None
            line_end = start + len(chunk) if chunk else None
            next_offset = line_end + 1 if line_end is not None and line_end < total else None
            result = "\n".join(numbered) or "(empty file)"
            result += "\n" + _page_footer(
                line_start=line_start,
                line_end=line_end,
                total_lines=total,
                next_offset=next_offset,
                content_hash=content_hash,
            )
            return ToolResult(
                result,
                facts=ToolResultFacts(
                    content_hash=content_hash,
                    content_bytes=len(raw),
                    encoding="utf-8",
                    decoding_replaced=decoding_replaced,
                    line_start=line_start,
                    line_end=line_end,
                    total_lines=total,
                    next_offset=next_offset,
                ),
            )
        except Exception as e:
            return f"Error: {e}"


def _page_footer(
    *,
    line_start: int | None,
    line_end: int | None,
    total_lines: int,
    next_offset: int | None,
    content_hash: str,
) -> str:
    """Render the model-visible page locator from the same facts persisted in events."""

    shown = f"{line_start}-{line_end}" if line_start is not None else "no lines"
    next_page = str(next_offset) if next_offset is not None else "none"
    return (
        f"... (page {shown} of {total_lines} lines; next offset: {next_page}; "
        f"source SHA-256: {content_hash})"
    )
