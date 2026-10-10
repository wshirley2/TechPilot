import time

from ..tool_results import ToolResult, ToolStatus
from .base import Tool


class NowTool(Tool):
    name = "now"
    description = "Get the current local date and time. Use this when the user asks about the current time or you need a timestamp."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    def execute(self) -> str:
        return ToolResult(time.strftime("%Y-%m-%d %H:%M:%S"), ToolStatus.COMPLETED)
