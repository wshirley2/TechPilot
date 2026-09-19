"""Repository-scoped, permission-gated Tool execution for Chat."""

from __future__ import annotations

import re
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import uuid4

from techpilot.engine.agent import ToolExecutionContext
from techpilot.engine.events import RuntimeEventType
from techpilot.engine.permissions import (
    DenyPermissionPrompt,
    PermissionAction,
    PermissionEffect,
    PermissionManager,
    PermissionRequest,
)
from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect, ToolExecutionDescription
from techpilot.engine.tool_results import ToolResult, ToolStatus, tool_result
from techpilot.engine.tool_validation import validate_tool_arguments
from techpilot.engine.tools.base import Tool
from techpilot.engine.tools.bash import BashTool
from techpilot.engine.tools.edit import _changed_files
from techpilot.engine.trusted_diff import FileWriteProposal, SourceSnapshotChanged, TrustedDiffError
from techpilot.safety.paths import is_sensitive_path

from ..execution import (
    CommandKind,
    ExecutionControlAssessment,
    ExecutionControlPolicy,
    ExternalEffect,
    FileCategory,
    ImpactScope,
    NormalizedCommand,
    NormalizedToolRequest,
    OperationKind,
    PathBoundary,
    RequiredControl,
)
from .permissions import ChatPermissionPolicy, command_effect, command_prefix, command_tokens

_PATH_ARGUMENTS = {
    "read_file": "file_path",
    "edit_file": "file_path",
    "write_file": "file_path",
    "glob": "path",
    "grep": "path",
}
_TOOL_EFFECTS = {
    "read_file": PermissionEffect.READ,
    "glob": PermissionEffect.READ,
    "grep": PermissionEffect.READ,
    "now": PermissionEffect.READ,
    "edit_file": PermissionEffect.WRITE,
    "write_file": PermissionEffect.WRITE,
    "bash": PermissionEffect.EXECUTE,
    "fetch_url": PermissionEffect.NETWORK,
    "agent": PermissionEffect.DELEGATE,
}
class RepositoryToolExecutor:
    """Bind tools to a repository and enforce C3 permissions before effects."""

    def __init__(
        self,
        repository_root: str | Path,
        permission_manager: PermissionManager | None = None,
        *,
        task_id: str | None = None,
    ):
        self.repository_root = Path(repository_root).resolve()
        self.task_id = task_id
        self.permission_manager = permission_manager or PermissionManager(
            ChatPermissionPolicy(),
            DenyPermissionPrompt(),
        )
        self._side_effect_lock = threading.Lock()
        self._turn_stop_lock = threading.Lock()
        self._turn_stop_message: str | None = None
        self._execution_control_policy = ExecutionControlPolicy()

    def begin_turn(self) -> None:
        """Clear the optional stop request left by the previous Chat turn."""

        with self._turn_stop_lock:
            self._turn_stop_message = None

    def set_task_id(self, task_id: str | None) -> None:
        """Refresh the optional Chat task correlation after Session resume."""

        self.task_id = task_id

    def consume_turn_stop_message(self) -> str | None:
        """Return and clear a user-denial request to end the current turn."""

        with self._turn_stop_lock:
            message = self._turn_stop_message
            self._turn_stop_message = None
            return message

    def execute(self, tool: Tool, arguments: dict[str, Any]) -> str:
        return self.execute_call(tool, arguments, tool_call_id=uuid4().hex)

    def describe_call(self, tool: Tool, arguments: dict[str, Any]) -> ToolExecutionDescription | None:
        """Expose conservative scheduling facts without changing C3 enforcement.

        The Agent uses this before executing a fully returned Tool Call round.
        Unknown tools deliberately return ``None`` so the generic scheduler
        falls back to UNKNOWN + EXCLUSIVE.
        """

        effect = {
            "read_file": ToolEffect.READ,
            "glob": ToolEffect.READ,
            "grep": ToolEffect.READ,
            "now": ToolEffect.READ,
            "edit_file": ToolEffect.WRITE,
            "write_file": ToolEffect.WRITE,
            "bash": ToolEffect.EXECUTE,
            "fetch_url": ToolEffect.NETWORK,
            "agent": ToolEffect.DELEGATE,
        }.get(tool.name)
        if effect is None:
            return None
        cwd = str(self.repository_root)
        if tool.name == "now":
            return ToolExecutionDescription(ToolEffect.READ, ToolConcurrency.SAFE, cwd=cwd)
        path_argument = _PATH_ARGUMENTS.get(tool.name)
        if path_argument is not None:
            boundary, resources, resolved = self._normalized_path_fact(arguments.get(path_argument, "."))
            if boundary in {PathBoundary.REPOSITORY, PathBoundary.APPROVED_ARTIFACT} and resolved is not None:
                return ToolExecutionDescription(
                    effect,
                    ToolConcurrency.SAFE if effect is ToolEffect.READ else ToolConcurrency.EXCLUSIVE,
                    (str(resolved),),
                    cwd,
                )
            return ToolExecutionDescription(effect, ToolConcurrency.EXCLUSIVE, resources, cwd, resources_known=False)
        return ToolExecutionDescription(effect, ToolConcurrency.EXCLUSIVE, cwd=cwd, resources_known=False)

    def execute_call(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        *,
        tool_call_id: str,
        execution_context: ToolExecutionContext | None = None,
    ) -> str:
        try:
            validate_tool_arguments(tool, arguments)
        except (TypeError, ValueError) as error:
            return ToolResult(f"Error: bad arguments for {tool.name}: {error}", ToolStatus.ERROR)
        return tool_result(self._execute_call(
            tool, arguments, tool_call_id=tool_call_id, execution_context=execution_context,
        ))

    def _execute_call(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        *,
        tool_call_id: str,
        execution_context: ToolExecutionContext | None = None,
    ) -> str:
        """Execute the exact Runtime-held call after path and permission checks."""

        normalized = dict(arguments)
        path_boundary = PathBoundary.REPOSITORY
        affected_paths: tuple[str, ...] = ()
        path_argument = _PATH_ARGUMENTS.get(tool.name)
        if path_argument:
            value = normalized.get(path_argument, ".")
            path_boundary, affected_paths, resolved_path = self._normalized_path_fact(value)
            if resolved_path is not None:
                normalized[path_argument] = str(resolved_path)

        control_request = self._normalized_control_request(
            tool.name,
            normalized,
            path_boundary=path_boundary,
            affected_paths=affected_paths,
        )
        assessment = self._execution_control_policy.assess(control_request)
        self._emit_execution_control_assessment(
            execution_context,
            tool_call_id,
            control_request,
            assessment,
        )
        if assessment.required_control is RequiredControl.BLOCK:
            message = _control_message("该操作已被阻断，未执行。", assessment)
            # A rejected read/search has performed no side effect.  Keep the
            # boundary intact but let the Agent use the denial as feedback and
            # retry inside the repository.  Writes, shell commands and other
            # effectful operations still end the turn after a block.
            if tool.name not in {"read_file", "glob", "grep"}:
                self._request_turn_stop(message)
            return f"Policy denied {tool.name}: {message}"
        if tool.name == "glob":
            pattern_error = _validate_relative_pattern(normalized.get("pattern"), "glob pattern")
            if pattern_error:
                return f"Policy denied glob: {pattern_error}"
        if tool.name == "grep" and normalized.get("include") is not None:
            pattern_error = _validate_relative_pattern(normalized.get("include"), "grep include")
            if pattern_error:
                return f"Policy denied grep: {pattern_error}"
        if path_argument and path_boundary not in {
            PathBoundary.REPOSITORY,
            PathBoundary.APPROVED_ARTIFACT,
        }:
            return f"Policy denied {tool.name}: path could not be normalized within the repository"

        if tool.name in {"read_file", "grep", "glob"}:
            raw = Path(arguments.get(path_argument, ".")).expanduser()
            requested = raw if raw.is_absolute() else self.repository_root / raw
            if is_sensitive_path(requested):
                return ToolResult(f"Permission denied {tool.name}: sensitive file", ToolStatus.DENIED)

        if tool.name in {"edit_file", "write_file"}:
            with self._side_effect_lock:
                return self._execute_write(tool, normalized, tool_call_id, assessment)
        if tool.name == "bash":
            if not isinstance(tool, BashTool):
                return "Error: bash tool does not support a repository working directory"
            with self._side_effect_lock:
                return self._execute_command(tool, normalized, tool_call_id, assessment)

        request = PermissionRequest(
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            effect=_TOOL_EFFECTS.get(tool.name, PermissionEffect.UNKNOWN),
            normalized_arguments=normalized,
            reason="Repository-scoped read request",
            scope=str(self.repository_root),
        )
        decision = self.permission_manager.authorize(request)
        if decision.action is not PermissionAction.ALLOW:
            return _denied(tool.name, decision.reason)
        return tool.execute(**normalized)

    def _execute_command(
        self,
        tool: BashTool,
        normalized: dict[str, Any],
        tool_call_id: str,
        assessment: ExecutionControlAssessment,
    ) -> str:
        command = normalized.get("command")
        if not isinstance(command, str):
            return "Policy denied bash: command must be a string"
        tokens = command_tokens(command)
        request = PermissionRequest(
            tool_call_id=tool_call_id,
            tool_name="bash",
            effect=command_effect(command),
            normalized_arguments=normalized,
            reason=_confirmation_reason(
                "Shell command may have repository or environment side effects",
                assessment,
            ),
            scope=command,
            command_tokens=tokens,
            command_prefix=command_prefix(tokens),
        )
        decision = self.permission_manager.authorize(request)
        if decision.action is not PermissionAction.ALLOW:
            self._stop_after_interactive_denial("bash", decision)
            return _denied("bash", decision.reason)
        return tool.execute_in(
            command,
            cwd=str(self.repository_root),
            timeout=normalized.get("timeout", 120),
        )

    def _execute_write(
        self,
        tool: Tool,
        normalized: dict[str, Any],
        tool_call_id: str,
        assessment: ExecutionControlAssessment,
    ) -> str:
        force_prompt = False
        for _attempt in range(4):
            try:
                proposal = self._build_proposal(tool.name, normalized)
            except (OSError, TrustedDiffError) as error:
                return f"Error: {error}"
            if not proposal.has_changes:
                return f"No changes needed for {proposal.display_path}"

            request = PermissionRequest(
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                effect=PermissionEffect.WRITE,
                normalized_arguments=normalized,
                reason=(
                    _confirmation_reason(
                        "Source changed after an earlier approval; review the regenerated diff",
                        assessment,
                    )
                    if force_prompt
                    else _confirmation_reason(
                        "File content will change only after approval",
                        assessment,
                    )
                ),
                scope=proposal.display_path,
                trusted_preview=proposal.trusted_diff,
                source_snapshot=proposal.source_snapshot,
            )
            decision = self.permission_manager.authorize(request, force_prompt=force_prompt)
            if decision.action is not PermissionAction.ALLOW:
                self._stop_after_interactive_denial(tool.name, decision)
                return _denied(tool.name, decision.reason)
            try:
                proposal.apply()
            except SourceSnapshotChanged:
                force_prompt = True
                continue
            except OSError as error:
                return ToolResult(
                    f"[effect unknown] Error: {error}; inspect the file before retrying",
                    ToolStatus.EFFECT_UNKNOWN,
                )

            _changed_files.add(str(proposal.path))
            if tool.name == "edit_file":
                return f"Edited {proposal.display_path}\n{proposal.trusted_diff}"
            line_count = proposal.candidate_content.count("\n") + (
                1
                if proposal.candidate_content and not proposal.candidate_content.endswith("\n")
                else 0
            )
            return f"Wrote {line_count} lines to {proposal.display_path}\n{proposal.trusted_diff}"
        return "Error: source kept changing during approval; no file was written"

    def _build_proposal(self, tool_name: str, normalized: dict[str, Any]) -> FileWriteProposal:
        path = Path(normalized["file_path"])
        display_path = path.relative_to(self.repository_root).as_posix()
        if tool_name == "edit_file":
            old_string = normalized.get("old_string")
            new_string = normalized.get("new_string")
            if not isinstance(old_string, str) or not isinstance(new_string, str):
                raise TrustedDiffError("edit_file requires string old_string and new_string")
            return FileWriteProposal.for_edit(
                path,
                display_path=display_path,
                old_string=old_string,
                new_string=new_string,
            )
        content = normalized.get("content")
        if not isinstance(content, str):
            raise TrustedDiffError("write_file requires string content")
        return FileWriteProposal.for_write(path, display_path=display_path, content=content)

    def _resolve_path(self, value: object) -> Path:
        boundary, _, candidate = self._normalized_path_fact(value)
        if boundary is PathBoundary.REPOSITORY and candidate is not None:
            return candidate
        if boundary is PathBoundary.OUTSIDE_REPOSITORY:
            raise ValueError(f"path is outside repository: {value}")
        raise ValueError("path must be a non-empty string")

    def _normalized_path_fact(self, value: object) -> tuple[PathBoundary, tuple[str, ...], Path | None]:
        if not isinstance(value, (str, Path)) or not str(value):
            return PathBoundary.UNRESOLVED, ("<invalid path>",), None
        raw = Path(str(value)).expanduser()
        candidate = raw.resolve() if raw.is_absolute() else (self.repository_root / raw).resolve()
        if _is_dangerous_system_path(candidate):
            return PathBoundary.DANGEROUS_SYSTEM, (str(candidate),), candidate
        try:
            relative = candidate.relative_to(self.repository_root).as_posix()
        except ValueError:
            return PathBoundary.OUTSIDE_REPOSITORY, (str(candidate),), candidate
        return PathBoundary.REPOSITORY, (relative,), candidate

    def _normalized_control_request(
        self,
        tool_name: str,
        normalized: dict[str, Any],
        *,
        path_boundary: PathBoundary,
        affected_paths: tuple[str, ...],
    ) -> NormalizedToolRequest:
        command = _normalized_command(normalized.get("command")) if tool_name == "bash" else None
        operation = _operation_kind(tool_name, command)
        return NormalizedToolRequest(
            tool_name=tool_name,
            operation=operation,
            path_boundary=path_boundary,
            affected_paths=affected_paths,
            file_categories=_file_categories(affected_paths, command),
            impact_scope=_impact_scope(operation, command, affected_paths),
            command=command,
            external_effect=_external_effect(tool_name, command),
        )

    def _emit_execution_control_assessment(
        self,
        execution_context: ToolExecutionContext | None,
        tool_call_id: str,
        request: NormalizedToolRequest,
        assessment: ExecutionControlAssessment,
    ) -> None:
        if execution_context is None:
            return
        execution_context.emit(
            RuntimeEventType.EXECUTION_CONTROL_ASSESSED,
            tool_call_id=tool_call_id,
            payload={
                "task_id": self.task_id,
                "tool_name": request.tool_name,
                "normalized_summary": _control_summary(request),
                "required_control": assessment.required_control.value,
                "reasons": _serialized_reasons(assessment),
            },
        )

    def _request_turn_stop(self, message: str) -> None:
        with self._turn_stop_lock:
            self._turn_stop_message = message

    def _stop_after_interactive_denial(self, tool_name: str, decision) -> None:
        """Stop retry loops only when a user explicitly rejected an ASK request."""

        if decision.action is not PermissionAction.DENY or not decision.prompted:
            return
        self._request_turn_stop(
            f"已按你的拒绝停止本轮后续操作；{tool_name} 没有执行。你可以继续说明下一步需求。"
        )


def _normalized_command(value: object) -> NormalizedCommand:
    if not isinstance(value, str) or not value.strip():
        return NormalizedCommand(is_parseable=False)
    tokens = command_tokens(value)
    lowered = tuple(token.lower() for token in tokens)
    has_fix = any(token in {"--fix", "--write", "--update-snapshots", "-w"} for token in lowered)
    has_complex_shell = bool(re.search(r"(?:\|\|?|&&?|;|[<>]{1,2}|\$\()", value))
    return NormalizedCommand(
        tokens=tokens,
        kind=_command_kind(lowered, has_fix),
        is_parseable=bool(tokens),
        has_pipeline=has_complex_shell,
        has_redirection=bool(re.search(r"[<>]{1,2}", value)),
        has_command_substitution="$(" in value,
        has_fix=has_fix,
    )


def _command_kind(tokens: tuple[str, ...], has_fix: bool) -> CommandKind:
    names = tuple(Path(token).name.lower() for token in tokens)
    if names[:1] in {("dir",), ("ls",)}:
        return CommandKind.READ_ONLY_SHELL
    git_subcommand = _git_subcommand(names)
    if git_subcommand == "push":
        return CommandKind.PUSH
    if git_subcommand in {"apply", "am"}:
        return CommandKind.PATCH_APPLY
    if "publish" in names:
        return CommandKind.PUBLISH
    if any(name in {"protoc", "openapi-generator", "datamodel-codegen"} for name in names):
        return CommandKind.CODE_GENERATION
    if has_fix or any(name in {"black", "autopep8", "isort", "prettier"} for name in names):
        return CommandKind.FORMAT
    if len(names) >= 2 and names[0] == "git" and names[1] in {
        "blame", "branch", "cat-file", "describe", "diff", "grep", "log", "ls-files", "show", "status",
    }:
        return CommandKind.READ_ONLY_GIT
    if any(name in {"pytest", "ruff", "flake8", "mypy", "eslint", "pylint", "vitest", "jest"} for name in names):
        return CommandKind.TEST if any(name in {"pytest", "vitest", "jest"} for name in names) else CommandKind.LINT
    return CommandKind.GENERAL


def _git_subcommand(names: tuple[str, ...]) -> str | None:
    """Return a Git subcommand while tolerating its global path options."""

    if names[:1] != ("git",):
        return None
    option_with_value = {"-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"}
    skip_value = False
    for name in names[1:]:
        if skip_value:
            skip_value = False
            continue
        if name in option_with_value:
            skip_value = True
            continue
        if name.startswith("-"):
            continue
        return name
    return None


def _operation_kind(tool_name: str, command: NormalizedCommand | None) -> OperationKind:
    fixed = {
        "read_file": OperationKind.READ,
        "glob": OperationKind.SEARCH,
        "grep": OperationKind.SEARCH,
        "edit_file": OperationKind.WRITE,
        "write_file": OperationKind.WRITE,
        "fetch_url": OperationKind.NETWORK,
        "now": OperationKind.READ,
    }.get(tool_name)
    if fixed is not None:
        return fixed
    tokens = tuple(token.lower() for token in command.tokens) if command is not None else ()
    names = tuple(Path(token).name for token in tokens)
    if names[:2] == ("git", "rm") or names[:1] in {"rm", "del", "erase", "rmdir", "rd", "remove-item", "unlink"}:
        return OperationKind.DELETE
    if names[:2] == ("git", "mv") or names[:1] in {"mv", "move", "move-item"}:
        return OperationKind.MOVE
    if names[:1] in {"ren", "rename", "rename-item"}:
        return OperationKind.RENAME
    if command is not None and command.kind is CommandKind.PUBLISH:
        return OperationKind.PUBLISH
    return OperationKind.COMMAND


def _file_categories(paths: tuple[str, ...], command: NormalizedCommand | None) -> frozenset[FileCategory]:
    categories: set[FileCategory] = set()
    for path in paths:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        if name.endswith(".lock") or name in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"}:
            categories.add(FileCategory.LOCK_FILE)
        elif name in {"pyproject.toml", "package.json", "requirements.txt", "pipfile", "cargo.toml", "go.mod"}:
            categories.add(FileCategory.DEPENDENCY_MANIFEST)
        elif "/migrations/" in f"/{normalized}" or "migration" in name:
            categories.add(FileCategory.DATABASE_MIGRATION)
        elif normalized.startswith(".github/workflows/") or name in {".gitlab-ci.yml", "azure-pipelines.yml"}:
            categories.add(FileCategory.CI_CONFIG)
        elif any(part in normalized for part in ("deploy/", "deployment/", "dockerfile", "docker-compose", "helm/")):
            categories.add(FileCategory.DEPLOYMENT_CONFIG)
        else:
            categories.add(FileCategory.SOURCE)
    if command and command.kind is CommandKind.CODE_GENERATION:
        categories.add(FileCategory.SOURCE)
    return frozenset(categories or {FileCategory.OTHER})


def _impact_scope(
    operation: OperationKind,
    command: NormalizedCommand | None,
    paths: tuple[str, ...],
) -> ImpactScope:
    if len(paths) > 1:
        return ImpactScope.MULTI_FILE
    if paths and paths[0].endswith("/"):
        return ImpactScope.DIRECTORY
    if operation in {OperationKind.DELETE, OperationKind.MOVE, OperationKind.RENAME}:
        return ImpactScope.UNKNOWN
    if command and command.kind in {CommandKind.FORMAT, CommandKind.CODE_GENERATION}:
        return ImpactScope.UNKNOWN
    return ImpactScope.SINGLE_FILE


def _external_effect(tool_name: str, command: NormalizedCommand | None) -> ExternalEffect:
    if tool_name == "fetch_url":
        return ExternalEffect.NETWORK
    if command is None:
        return ExternalEffect.NONE
    if command.kind is CommandKind.PUSH:
        return ExternalEffect.PUSH
    if command.kind is CommandKind.PUBLISH:
        return ExternalEffect.PUBLISH
    names = {Path(token).name.lower() for token in command.tokens}
    if names & {"curl", "wget", "invoke-webrequest", "invoke-restmethod", "iwr", "irm"}:
        return ExternalEffect.NETWORK
    return ExternalEffect.NONE


def _control_summary(request: NormalizedToolRequest) -> dict[str, object]:
    summary: dict[str, object] = {
        "operation": request.operation.value,
        "path_boundary": request.path_boundary.value,
        "affected_paths": list(request.affected_paths),
        "file_categories": sorted(category.value for category in request.file_categories),
        "impact_scope": request.impact_scope.value,
        "external_effect": request.external_effect.value,
    }
    if request.command is not None:
        summary["command"] = {
            "tokens": list(request.command.tokens),
            "kind": request.command.kind.value,
            "is_parseable": request.command.is_parseable,
            "has_pipeline": request.command.has_pipeline,
            "has_redirection": request.command.has_redirection,
            "has_command_substitution": request.command.has_command_substitution,
            "has_fix": request.command.has_fix,
        }
    return summary


def _serialized_reasons(assessment: ExecutionControlAssessment) -> list[dict[str, object]]:
    return [
        {
            "code": reason.code.value,
            "required_control": reason.required_control.value,
            "message": reason.message,
            "evidence": list(reason.evidence),
        }
        for reason in assessment.reasons
    ]


def _control_message(prefix: str, assessment: ExecutionControlAssessment) -> str:
    details = "；".join(
        f"{reason.message}（{'，'.join(reason.evidence)}）"
        for reason in assessment.reasons
    )
    return f"{prefix}\n原因与证据：{details}"


def _confirmation_reason(default: str, assessment: ExecutionControlAssessment) -> str:
    """Attach deterministic control facts to the existing C3 confirmation prompt."""

    reasons = [reason for reason in assessment.reasons if reason.required_control is RequiredControl.CONFIRM]
    if not reasons:
        return default
    details = "\n".join(
        f"- {reason.message}：{'；'.join(reason.evidence)}"
        for reason in reasons
    )
    return f"{default}\n执行控制原因与证据：\n{details}"


def _is_dangerous_system_path(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return normalized.startswith(("c:/windows/", "c:/program files/", "/etc/", "/usr/", "/bin/", "/sbin/"))


def _denied(tool_name: str, reason: str) -> str:
    return f"Permission denied {tool_name}: {reason}"


def _validate_relative_pattern(value: object, label: str) -> str | None:
    if not isinstance(value, str) or not value:
        return f"{label} must be a non-empty string"
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return f"{label} must be relative to the repository search path"
    if ".." in value.replace("\\", "/").split("/"):
        return f"{label} cannot contain parent-directory traversal"
    return None
