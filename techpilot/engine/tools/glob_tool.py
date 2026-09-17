"""File pattern matching."""

from pathlib import Path

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
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            base = Path(path).expanduser().resolve()
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = list(base.glob(pattern))
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
            return result or "No files matched."
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
