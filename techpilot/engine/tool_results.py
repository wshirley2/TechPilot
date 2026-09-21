"""Tool outcomes, with a text-compatible bridge for existing tools and providers.

New tools should return ToolResult explicitly. Legacy string classification is
only a compatibility fallback; event consumers should prefer ``tool_status``.
"""

import re
from dataclasses import dataclass
from enum import Enum


class ToolStatus(str, Enum):
    COMPLETED = "completed"
    DENIED = "denied"
    ERROR = "error"
    NOT_EXECUTED = "not executed"
    EFFECT_UNKNOWN = "effect unknown"


@dataclass(frozen=True, slots=True)
class ToolResultFacts:
    """Host-side execution facts kept separate from model-facing result text.

    ``stdout`` and ``stderr`` contain complete streams only.  When a tool has
    to truncate its model-facing output and no full-output artifact exists,
    both are ``None`` so consumers cannot mistake a preview for a record.
    """

    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    cwd: str | None = None
    truncated: bool = False
    output_chars: int | None = None
    output_artifact: str | None = None
    timed_out: bool = False
    timeout_seconds: int | None = None
    content_hash: str | None = None
    content_bytes: int | None = None
    encoding: str | None = None
    decoding_replaced: bool = False
    line_start: int | None = None
    line_end: int | None = None
    total_lines: int | None = None
    next_offset: int | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable event payload with stable field names."""

        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "cwd": self.cwd,
            "truncated": self.truncated,
            "output_chars": self.output_chars,
            "output_artifact": self.output_artifact,
            "timed_out": self.timed_out,
            "timeout_seconds": self.timeout_seconds,
            "content_hash": self.content_hash,
            "content_bytes": self.content_bytes,
            "encoding": self.encoding,
            "decoding_replaced": self.decoding_replaced,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "total_lines": self.total_lines,
            "next_offset": self.next_offset,
        }


class ToolResult(str):
    """Preserve model-facing text plus explicit host-side status and facts."""

    def __new__(
        cls,
        text: str,
        status: ToolStatus = ToolStatus.COMPLETED,
        *,
        facts: ToolResultFacts | None = None,
    ):
        instance = super().__new__(cls, text)
        instance.status = ToolStatus(status)
        instance.facts = facts or ToolResultFacts()
        return instance


def tool_result(value: str) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    text = str(value)
    lowered = text.lstrip().lower()
    if lowered.startswith(("[effect unknown]", "error: timed out after")):
        status = ToolStatus.EFFECT_UNKNOWN
    elif lowered.startswith(("[not executed", "[cancelled before execution]", "[limit reached]")):
        status = ToolStatus.NOT_EXECUTED
    elif lowered.startswith(("[interrupted]", "[cancelled]")):
        status = ToolStatus.EFFECT_UNKNOWN
    elif lowered.startswith(("policy denied", "permission denied", "⚠ blocked")):
        status = ToolStatus.DENIED
    elif lowered.startswith(("error", "invalid regex")) or re.search(r"\[exit code: -?[1-9]\d*\]\s*$", lowered):
        status = ToolStatus.ERROR
    else:
        status = ToolStatus.COMPLETED
    return ToolResult(text, status)


def display_tool_status(text: str, status: object = None) -> str:
    try:
        return ToolStatus(status).value
    except (ValueError, TypeError):
        return tool_result(text).status.value
