"""Effect-aware scheduling facts for one fully-returned Tool Call round.

This module contains no Permission or product policy.  It only decides which
already-authorized candidates may share a read-only execution wave.  Effects
remain enforced by the application executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class ToolEffect(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DELEGATE = "delegate"
    UNKNOWN = "unknown"


class ToolConcurrency(str, Enum):
    SAFE = "safe"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True, slots=True)
class ToolExecutionDescription:
    """Declared facts for one Tool invocation.

    ``resources_known`` intentionally defaults to ``True`` for tools such as
    ``now`` that do not touch a path.  An ambiguous path must set it to False,
    which forces an exclusive barrier even when the nominal effect is READ.
    """

    effect: ToolEffect
    concurrency: ToolConcurrency
    affected_resources: tuple[str, ...] = ()
    cwd: str | None = None
    resources_known: bool = True

    @classmethod
    def unknown(cls) -> ToolExecutionDescription:
        return cls(ToolEffect.UNKNOWN, ToolConcurrency.EXCLUSIVE, resources_known=False)

    @property
    def is_safe_read(self) -> bool:
        return (
            self.effect is ToolEffect.READ
            and self.concurrency is ToolConcurrency.SAFE
            and self.resources_known
        )


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    """One host-prepared tool invocation shared by scheduling and execution.

    ``requested_arguments`` preserve what the model asked for; the executor may
    put repository-normalized values in ``normalized_arguments``.  The opaque
    ``host_context`` is only meaningful to the executor that created it.
    ``create`` makes shallow immutable copies so workers cannot alter the
    scheduling facts of their siblings.
    """

    tool_name: str
    requested_arguments: Mapping[str, Any]
    normalized_arguments: Mapping[str, Any]
    description: ToolExecutionDescription
    host_context: object | None = None

    @classmethod
    def create(
        cls,
        tool_name: str,
        requested_arguments: Mapping[str, Any],
        description: ToolExecutionDescription,
        *,
        normalized_arguments: Mapping[str, Any] | None = None,
        host_context: object | None = None,
    ) -> PreparedToolCall:
        return cls(
            tool_name,
            MappingProxyType(dict(requested_arguments)),
            MappingProxyType(dict(normalized_arguments if normalized_arguments is not None else requested_arguments)),
            description,
            host_context,
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionWave:
    """A contiguous group of provider-order call indexes."""

    indexes: tuple[int, ...]
    concurrent: bool


@dataclass(frozen=True, slots=True)
class ToolExecutionPlan:
    """Partition calls into concurrent read waves and exclusive barriers."""

    descriptions: tuple[ToolExecutionDescription, ...]
    waves: tuple[ToolExecutionWave, ...]

    @classmethod
    def build(cls, descriptions: list[ToolExecutionDescription]) -> ToolExecutionPlan:
        waves: list[ToolExecutionWave] = []
        read_indexes: list[int] = []

        def flush_reads() -> None:
            if read_indexes:
                waves.append(ToolExecutionWave(tuple(read_indexes), concurrent=len(read_indexes) > 1))
                read_indexes.clear()

        for index, description in enumerate(descriptions):
            if description.is_safe_read:
                read_indexes.append(index)
                continue
            flush_reads()
            waves.append(ToolExecutionWave((index,), concurrent=False))
        flush_reads()
        return cls(tuple(descriptions), tuple(waves))


def resources_conflict(left: ToolExecutionDescription, right: ToolExecutionDescription) -> bool:
    """Return the conservative first-version resource conflict decision."""

    if not left.resources_known or not right.resources_known:
        return True
    return not (left.effect is ToolEffect.READ and right.effect is ToolEffect.READ)


_DEFAULT_EFFECTS = {
    "read_file": ToolEffect.READ,
    "glob": ToolEffect.READ,
    "grep": ToolEffect.READ,
    "now": ToolEffect.READ,
    "edit_file": ToolEffect.WRITE,
    "write_file": ToolEffect.WRITE,
    "bash": ToolEffect.EXECUTE,
}
_PATH_ARGUMENTS = {
    "read_file": "file_path",
    "edit_file": "file_path",
    "write_file": "file_path",
    "glob": "path",
    "grep": "path",
}


def declared_tool_description(tool: object, arguments: dict[str, Any]) -> ToolExecutionDescription | None:
    """Read optional per-tool metadata without teaching the scheduler new tools.

    A Tool may implement ``describe_call(arguments)`` or set
    ``execution_effect``, ``execution_concurrency``, ``execution_resources``
    and ``execution_cwd`` attributes.  Bad declarations fail closed.
    """

    describe_call = getattr(tool, "describe_call", None)
    if callable(describe_call):
        try:
            described = describe_call(dict(arguments))
        except Exception:
            return ToolExecutionDescription.unknown()
        if isinstance(described, ToolExecutionDescription):
            return described
        return ToolExecutionDescription.unknown()

    effect = getattr(tool, "execution_effect", None)
    concurrency = getattr(tool, "execution_concurrency", None)
    if effect is None and concurrency is None:
        return None
    try:
        resolved_effect = effect if isinstance(effect, ToolEffect) else ToolEffect(effect)
        resolved_concurrency = (
            concurrency
            if isinstance(concurrency, ToolConcurrency)
            else ToolConcurrency(concurrency)
        )
    except (TypeError, ValueError):
        return ToolExecutionDescription.unknown()
    resources = getattr(tool, "execution_resources", ())
    if callable(resources):
        try:
            resources = resources(dict(arguments))
        except Exception:
            return ToolExecutionDescription.unknown()
    if not isinstance(resources, (tuple, list)) or not all(isinstance(value, str) for value in resources):
        return ToolExecutionDescription.unknown()
    cwd = getattr(tool, "execution_cwd", None)
    if cwd is not None and not isinstance(cwd, str):
        return ToolExecutionDescription.unknown()
    return ToolExecutionDescription(
        resolved_effect,
        resolved_concurrency,
        tuple(resources),
        cwd,
        resources_known=getattr(tool, "execution_resources_known", True) is True,
    )


def default_tool_description(tool_name: str, arguments: dict[str, Any]) -> ToolExecutionDescription:
    """Describe built-ins when no application-specific description exists."""

    effect = _DEFAULT_EFFECTS.get(tool_name)
    if effect is None:
        return ToolExecutionDescription.unknown()
    if effect is not ToolEffect.READ:
        resource = _resource_argument(tool_name, arguments)
        return ToolExecutionDescription(effect, ToolConcurrency.EXCLUSIVE, resource, resources_known=bool(resource))
    if tool_name == "now":
        return ToolExecutionDescription(ToolEffect.READ, ToolConcurrency.SAFE)
    resource = _resource_argument(tool_name, arguments, default=".")
    return ToolExecutionDescription(
        ToolEffect.READ,
        ToolConcurrency.SAFE if resource else ToolConcurrency.EXCLUSIVE,
        resource,
        resources_known=bool(resource),
    )


def _resource_argument(tool_name: str, arguments: dict[str, Any], default: str | None = None) -> tuple[str, ...]:
    argument_name = _PATH_ARGUMENTS.get(tool_name)
    if argument_name is None:
        return ()
    value = arguments.get(argument_name, default)
    if not isinstance(value, str) or not value:
        return ()
    return (value,)
