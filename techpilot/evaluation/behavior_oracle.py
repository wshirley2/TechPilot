"""A deliberately tiny, non-executing oracle for pure Python task fixtures.

It parses source to an AST and interprets only a closed, side-effect-free
subset. There is no ``exec``, import, file access, attribute lookup beyond
approved string methods, or user-defined call dispatch.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BEHAVIOR_ORACLE_VERSION = "2026-09-17.v3"


@dataclass(frozen=True)
class BehaviorCheck:
    id: str
    file_path: str
    function: str
    arguments: tuple[Any, ...]
    expected: Any
    constant_imports: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.id or self.id != self.id.lower():
            raise ValueError("behavior check id must be non-empty lowercase text")
        if not self.file_path or not self.function:
            raise ValueError("behavior check needs a file path and function")
        for module, name in self.constant_imports:
            if not module.isidentifier() or not name.isidentifier():
                raise ValueError("constant imports require identifier module and name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "function": self.function,
            "arguments": list(self.arguments),
            "expected": self.expected,
            "constant_imports": [
                {"module": module, "name": name}
                for module, name in self.constant_imports
            ],
        }


def behavior_oracle_identity() -> dict[str, str]:
    """Return the versioned source identity recorded in model-run manifests."""

    source = Path(__file__).read_bytes()
    return {
        "version": BEHAVIOR_ORACLE_VERSION,
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def evaluate_behavior_check(
    source: str,
    check: BehaviorCheck,
    *,
    workspace_sources: Mapping[str, str] | None = None,
) -> bool:
    """Return false for unsupported or malformed code; never execute it."""

    try:
        module = ast.parse(source, mode="exec")
        if any(not isinstance(node, (ast.FunctionDef, ast.ImportFrom, ast.Assign)) for node in module.body):
            return False
        imported_constants = _resolve_constant_imports(module, check, workspace_sources)
        function = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == check.function
        )
        arguments = function.args.args
        defaults = function.args.defaults
        required = len(arguments) - len(defaults)
        if (
            function.decorator_list
            or function.args.posonlyargs
            or function.args.vararg
            or function.args.kwonlyargs
            or function.args.kwarg
            or not required <= len(check.arguments) <= len(arguments)
        ):
            return False
        environment = _module_scalar_constants(module)
        environment.update(imported_constants)
        environment.update({argument.arg: value for argument, value in zip(arguments, check.arguments, strict=False)})
        for argument, default in zip(arguments[len(check.arguments):], defaults[len(check.arguments) - required:], strict=False):
            environment[argument.arg] = _evaluate(default, environment)
        returned, value = _evaluate_statements(function.body, environment)
        return returned and value == check.expected
    except (KeyError, StopIteration, SyntaxError, TypeError, ValueError):
        return False


def _resolve_constant_imports(
    module: ast.Module,
    check: BehaviorCheck,
    workspace_sources: Mapping[str, str] | None,
) -> dict[str, str | int | float | bool | None]:
    """Resolve only declared scalar constants from sibling source text.

    This is AST interpretation, not Python importing: no module code is
    executed and only a source file consisting exclusively of scalar constant
    assignments is accepted.
    """

    imports: list[tuple[str, str]] = []
    for node in module.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 0 or node.module is None or not node.module.isidentifier():
            raise ValueError("only top-level sibling constant imports are supported")
        for alias in node.names:
            if alias.asname is not None or not alias.name.isidentifier():
                raise ValueError("constant imports cannot use aliases")
            imports.append((node.module, alias.name))
    if tuple(imports) != check.constant_imports:
        raise ValueError("source imports do not match behavior check")
    if not imports:
        return {}
    if workspace_sources is None:
        raise ValueError("constant import checks require workspace sources")

    environment: dict[str, str | int | float | bool | None] = {}
    for module_name, name in imports:
        constants = _parse_constant_module(workspace_sources.get(f"{module_name}.py"))
        environment[name] = constants[name]
    return environment


def _parse_constant_module(source: str | None) -> dict[str, str | int | float | bool | None]:
    if source is None:
        raise ValueError("constant module is missing")
    module = ast.parse(source, mode="exec")
    constants: dict[str, str | int | float | bool | None] = {}
    for node in module.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or not _is_scalar_constant(node.value)
        ):
            raise ValueError("constant module must contain only scalar assignments")
        constants[node.targets[0].id] = node.value.value
    return constants


def _module_scalar_constants(module: ast.Module) -> dict[str, str | int | float | bool | None]:
    """Read only literal module constants; never evaluate module code."""

    constants: dict[str, str | int | float | bool | None] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.type_comment is not None
            or not _is_scalar_constant(node.value)
        ):
            raise ValueError("only scalar module constants are supported")
        constants[node.targets[0].id] = node.value.value
    return constants


def _evaluate_statements(statements: list[ast.stmt], environment: dict[str, Any]) -> tuple[bool, Any]:
    """Interpret only safe assignments, returns, and side-effect-free branches."""

    for statement in statements:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        if isinstance(statement, ast.Return):
            return True, _evaluate(statement.value, environment)
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.type_comment is not None
            ):
                raise ValueError("only single-name assignments are supported")
            value = _evaluate(statement.value, environment)
            if not _is_scalar(value):
                raise ValueError("assignment must evaluate to a scalar")
            environment[statement.targets[0].id] = value
            continue
        if isinstance(statement, ast.If):
            branch = statement.body if _evaluate(statement.test, environment) else statement.orelse
            returned, value = _evaluate_statements(branch, environment)
            if returned:
                return True, value
            continue
        raise ValueError("statement is outside behavior oracle subset")
    return False, None


def _evaluate(node: ast.expr, environment: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise ValueError("dictionary unpacking is outside behavior oracle subset")
        result: dict[str | int | float | bool, str | int | float | bool | None] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            evaluated_key = _evaluate(key, environment)
            evaluated_value = _evaluate(value, environment)
            if type(evaluated_key) not in {str, int, float, bool} or not _is_scalar(evaluated_value):
                raise ValueError("dictionary accepts only scalar keys and values")
            result[evaluated_key] = evaluated_value
        return result
    if isinstance(node, ast.Subscript):
        value = _evaluate(node.value, environment)
        if isinstance(node.slice, ast.Slice):
            if type(value) is not str or node.slice.step is not None:
                raise ValueError("only simple string slices are supported")
            lower = _evaluate(node.slice.lower, environment) if node.slice.lower is not None else None
            upper = _evaluate(node.slice.upper, environment) if node.slice.upper is not None else None
            if lower is not None and type(lower) is not int:
                raise ValueError("string slice bounds must be integers")
            if upper is not None and type(upper) is not int:
                raise ValueError("string slice bounds must be integers")
            return value[lower:upper]
        index = _evaluate(node.slice, environment)
        if type(value) is not dict or type(index) not in {str, int, float, bool}:
            raise ValueError("subscript accepts only an exact dictionary and scalar key")
        return value[index]
    if isinstance(node, ast.IfExp):
        return _evaluate(node.body if _evaluate(node.test, environment) else node.orelse, environment)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, environment)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _evaluate(node.left, environment)
        right = _evaluate(node.right, environment)
        if type(left) not in {int, float} or type(right) not in {int, float}:
            raise ValueError("addition accepts only numeric scalars")
        return left + right
    if isinstance(node, ast.BoolOp):
        result = _evaluate(node.values[0], environment)
        for value in node.values[1:]:
            if isinstance(node.op, ast.And) and not result:
                return result
            if isinstance(node.op, ast.Or) and result:
                return result
            result = _evaluate(value, environment)
        return result
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators):
        values = [_evaluate(node.left, environment)] + [_evaluate(value, environment) for value in node.comparators]
        return all(_compare(left, right, operation) for left, right, operation in zip(values, values[1:], node.ops))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if node.func.id in {"min", "max"}:
            if len(node.args) != 2:
                raise ValueError("min/max requires exactly two arguments")
            values = [_evaluate(argument, environment) for argument in node.args]
            if any(type(value) not in {int, float} for value in values):
                raise ValueError("min/max accepts only two numeric scalars")
            return min(values) if node.func.id == "min" else max(values)
        if node.func.id == "str" and len(node.args) == 1:
            value = _evaluate(node.args[0], environment)
            if not _is_scalar(value):
                raise ValueError("str accepts only a scalar")
            return str(value)
        if node.func.id == "bool" and len(node.args) == 1:
            value = _evaluate(node.args[0], environment)
            if not _is_scalar(value):
                raise ValueError("bool accepts only a scalar")
            return bool(value)
        if node.func.id == "len" and len(node.args) == 1:
            value = _evaluate(node.args[0], environment)
            if type(value) not in {str, dict, list}:
                raise ValueError("len accepts only exact strings, dictionaries, or lists")
            return len(value)
        if (
            node.func.id == "isinstance"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id in {"str", "int", "float", "bool"}
        ):
            value = _evaluate(node.args[0], environment)
            return type(value).__name__ == node.args[1].id
        raise ValueError("function is outside behavior oracle subset")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not node.keywords:
        value = _evaluate(node.func.value, environment)
        arguments = [_evaluate(argument, environment) for argument in node.args]
        allowed = {str: {"strip", "lower", "upper", "casefold", "removeprefix", "replace", "split", "join", "startswith"}, dict: {"get"}}
        if node.func.attr not in allowed.get(type(value), set()):
            raise ValueError("method is not in behavior oracle allowlist")
        if node.func.attr == "split" and arguments:
            raise ValueError("behavior oracle supports only no-argument str.split")
        return getattr(value, node.func.attr)(*arguments)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue) and value.conversion == -1 and value.format_spec is None:
                rendered = _evaluate(value.value, environment)
                if not _is_scalar(rendered):
                    raise TypeError("f-string value must be a scalar")
                parts.append(str(rendered))
            else:
                raise ValueError("f-string is outside behavior oracle subset")
        return "".join(parts)
    raise ValueError("expression is outside behavior oracle subset")


def _compare(left: Any, right: Any, operation: ast.cmpop) -> bool:
    if isinstance(operation, ast.Eq): return left == right
    if isinstance(operation, ast.NotEq): return left != right
    if isinstance(operation, ast.Lt): return left < right
    if isinstance(operation, ast.LtE): return left <= right
    if isinstance(operation, ast.Gt): return left > right
    if isinstance(operation, ast.GtE): return left >= right
    if isinstance(operation, ast.In): return left in right
    if isinstance(operation, ast.Is): return left is right
    if isinstance(operation, ast.IsNot): return left is not right
    raise ValueError("comparison is outside behavior oracle subset")


def _is_scalar(value: Any) -> bool:
    return value is None or type(value) in {str, int, float, bool}


def _is_scalar_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and _is_scalar(node.value)
