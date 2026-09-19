"""Host validation for tool calls before any execution or permission prompt.

Checks the primitive property types used by built-in schemas plus their domain
constraints. This is deliberately not a general JSON Schema implementation.
"""

import inspect
import math
from functools import wraps

from .tool_results import ToolResult, ToolStatus, tool_result


def validate_tool_arguments(tool, arguments) -> None:
    if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
        raise ValueError("arguments must be an object with string keys")
    inspect.signature(tool.execute).bind(**arguments)
    schema = getattr(tool, "parameters", {})
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            raise ValueError(f"missing required argument: {name}")
    types = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: type(v) is int,
        "number": lambda v: type(v) in (int, float) and math.isfinite(v),
        "boolean": lambda v: type(v) is bool,
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
    }
    for name, value in arguments.items():
        rule = properties.get(name, {})
        expected = rule.get("type")
        # Optional grep include=None retains its Python default semantics.
        if tool.name == "grep" and name == "include" and value is None:
            continue
        choices = expected if isinstance(expected, list) else [expected]
        if expected is not None and not any(t in types and types[t](value) for t in choices):
            raise ValueError(f"{name} must be {expected}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{name} must be one of {rule['enum']}")
        if isinstance(value, str) and "minLength" in rule and len(value) < rule["minLength"]:
            raise ValueError(f"{name} must contain at least {rule['minLength']} characters")
        if type(value) in (int, float):
            if "minimum" in rule and value < rule["minimum"]:
                raise ValueError(f"{name} must be >= {rule['minimum']}")
            if "maximum" in rule and value > rule["maximum"]:
                raise ValueError(f"{name} must be <= {rule['maximum']}")
    if tool.name in {"read_file", "edit_file", "write_file", "glob", "grep"}:
        for name in ("file_path", "path"):
            if name in arguments:
                value = arguments[name]
                if not isinstance(value, str) or not value.strip() or "\0" in value:
                    raise ValueError(f"{name} must be a non-empty path without NUL bytes")
    if tool.name == "read_file":
        for name in ("offset", "limit"):
            if name in arguments and (type(arguments[name]) is not int or arguments[name] < 1):
                raise ValueError(f"{name} must be a positive integer")
    if tool.name == "edit_file":
        for name in ("old_string", "new_string"):
            if name in arguments and not isinstance(arguments[name], str):
                raise ValueError(f"{name} must be a string")
        if arguments.get("old_string") == "":
            raise ValueError("old_string must not be empty; provide unique surrounding context")
    if tool.name == "write_file" and "content" in arguments and not isinstance(arguments["content"], str):
        raise ValueError("content must be a string")
    if tool.name == "bash":
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise ValueError("command must be non-empty and contain no NUL bytes")
        if "timeout" in arguments and (type(arguments["timeout"]) is not int or arguments["timeout"] < 1):
            raise ValueError("timeout must be a positive integer")
    if tool.name in {"glob", "grep"}:
        if "pattern" in arguments and not isinstance(arguments["pattern"], str):
            raise ValueError("pattern must be a string")
        if tool.name == "glob" and arguments.get("pattern") == "":
            raise ValueError("pattern must not be empty")


def validated_tool(function):
    """Apply the same checks to direct built-in execute calls and Agent calls."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            bound = inspect.signature(function).bind(self, *args, **kwargs)
            arguments = {key: value for key, value in bound.arguments.items() if key != "self"}
            validate_tool_arguments(self, arguments)
        except (TypeError, ValueError) as error:
            return ToolResult(f"Error: bad arguments for {self.name}: {error}", ToolStatus.ERROR)
        return tool_result(function(self, *args, **kwargs))
    return wrapped
