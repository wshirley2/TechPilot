"""File pattern matching."""

from pathlib import Path

from techpilot.safety.paths import is_safe_search_path, is_sensitive_path

from ..tool_results import ToolResult, ToolStatus
from ..tool_validation import validated_tool
from .base import Tool


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "minLength": 1,
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    @validated_tool
    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            if is_sensitive_path(Path(path).expanduser()):
                return ToolResult("Permission denied glob: sensitive path", ToolStatus.DENIED)
            base = Path(path).expanduser().resolve()
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = [hit for hit in base.glob(pattern) if is_safe_search_path(hit, base)]
            # sort by mtime, newest first
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            # Tool callers operate relative to the requested search root.  Do
            # not expose host-specific absolute workspace paths in the model
            # context: an Agent may otherwise reuse an artifact-parent path
            # in a later call and correctly hit the repository boundary.
            lines = [_display_path(hit, base) for hit in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return ToolResult(result or "No files matched.")
        except Exception as e:
            return f"Error: {e}"


def _display_path(path: Path, root: Path) -> str:
    """Render a search hit relative to its search root for Tool callers."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # A path outside the requested root is unexpected, but preserve a
        # useful diagnostic instead of misrepresenting its location.
        return str(path)
