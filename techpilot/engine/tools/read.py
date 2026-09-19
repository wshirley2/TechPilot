"""File reading with line numbers."""

from pathlib import Path

from techpilot.safety.paths import is_sensitive_path

from ..tool_results import ToolResult, ToolStatus
from ..tool_validation import validated_tool
from .base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file's contents with line numbers. "
        "Always read a file before editing it."
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
        },
        "required": ["file_path"],
    }

    @validated_tool
    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            if is_sensitive_path(Path(file_path).expanduser()):
                return ToolResult("Permission denied read_file: sensitive file", ToolStatus.DENIED)
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)

            start = max(0, offset - 1)
            chunk = lines[start : start + limit]
            numbered = [f"{start + i + 1}\t{ln}" for i, ln in enumerate(chunk)]
            result = "\n".join(numbered)

            if total > start + limit:
                result += f"\n... ({total} lines total, showing {start+1}-{start+len(chunk)})"
            return ToolResult(result or "(empty file)")
        except Exception as e:
            return f"Error: {e}"
