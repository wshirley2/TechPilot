"""File creation / overwrite."""

from pathlib import Path

from ..tool_results import ToolResult, ToolStatus
from ..tool_validation import validated_tool
from .base import Tool
from .edit import _changed_files


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    @validated_tool
    def execute(self, file_path: str, content: str) -> str:
        writing = False
        try:
            p = Path(file_path).expanduser().resolve()
            writing = True
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            _changed_files.add(str(p))
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {n_lines} lines to {file_path}"
        except Exception as e:
            if writing:
                return ToolResult(f"[effect unknown] Error: {e}; inspect the file before retrying", ToolStatus.EFFECT_UNKNOWN)
            return f"Error: {e}"
