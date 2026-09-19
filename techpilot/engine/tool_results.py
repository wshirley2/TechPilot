"""Tool outcomes, with a text-compatible bridge for existing tools and providers.

New tools should return ToolResult explicitly. Legacy string classification is
only a compatibility fallback; event consumers should prefer ``tool_status``.
"""

import re
from enum import Enum


class ToolStatus(str, Enum):
    COMPLETED = "completed"
    DENIED = "denied"
    ERROR = "error"
    NOT_EXECUTED = "not executed"
    EFFECT_UNKNOWN = "effect unknown"


class ToolResult(str):
    """Preserve model-facing text while carrying an explicit host-side status."""

    def __new__(cls, text: str, status: ToolStatus = ToolStatus.COMPLETED):
        instance = super().__new__(cls, text)
        instance.status = ToolStatus(status)
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
