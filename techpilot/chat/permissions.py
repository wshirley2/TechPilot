"""TechPilot Chat permission policy and terminal approval prompt."""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import PureWindowsPath

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from techpilot.engine.permissions import (
    PermissionDecision,
    PermissionEffect,
    PermissionGrantScope,
    PermissionRequest,
)
from techpilot.engine.tools.bash import _check_dangerous

InputFunction = Callable[[str], str]

_DELETE_COMMANDS = {"del", "erase", "rd", "rmdir", "rm", "remove-item", "unlink"}
_NETWORK_COMMANDS = {
    "curl",
    "invoke-restmethod",
    "invoke-webrequest",
    "irm",
    "iwr",
    "wget",
}
_INSTALL_COMMANDS = {
    "apt",
    "apt-get",
    "brew",
    "choco",
    "dnf",
    "pacman",
    "winget",
    "yum",
}
_READ_ONLY_GIT = {
    "blame",
    "branch",
    "cat-file",
    "describe",
    "diff",
    "for-each-ref",
    "grep",
    "log",
    "ls-files",
    "name-rev",
    "rev-parse",
    "shortlog",
    "show",
    "status",
}
_READ_ONLY_SHELL = {"dir", "ls"}
_SHELL_OPERATORS = {"&", "&&", "|", "||", ";", ">", ">>", "<", "<<"}

# This is deliberately much narrower than the permission allow-list.  A command
# that is permitted to run (for example a test command) may still write caches,
# execute project code, or change the environment, so it must not enter a SAFE
# scheduler wave merely because permission policy allows it.
_SAFE_GIT_STATUS_FLAGS = {
    "--short",
    "-s",
    "--porcelain",
    "--branch",
    "-b",
    "--untracked-files=no",
    "--untracked-files=normal",
    "--untracked-files=all",
    "--ignored=no",
    "--ignored=traditional",
    "--ignored=matching",
}
_SAFE_GIT_DIFF_FLAGS = {
    "--stat",
    "--name-only",
    "--name-status",
    "--summary",
    "--compact-summary",
    "--cached",
    "--staged",
    "--check",
    "--no-ext-diff",
    "--quiet",
    "--exit-code",
}


class ChatPermissionPolicy:
    """Deterministic default policy for one repository-scoped Chat runtime."""

    def __init__(self, validation_commands: Sequence[Sequence[str]] = ()):
        self._validation_commands = {
            subprocess.list2cmdline(list(command)) for command in validation_commands
        }

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        if request.effect is PermissionEffect.READ:
            return PermissionDecision.allow("Repository-scoped reads are allowed")
        if request.effect is PermissionEffect.WRITE:
            return PermissionDecision.ask("File writes require review of a trusted diff")
        if request.tool_name != "bash":
            return PermissionDecision.deny(f"Tool {request.tool_name!r} is not enabled in Chat")
        return self._command_decision(request)

    def _command_decision(self, request: PermissionRequest) -> PermissionDecision:
        command = request.normalized_arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return PermissionDecision.deny("bash requires a non-empty command")

        dangerous = dangerous_command_reason(command)
        if dangerous:
            return PermissionDecision.deny(f"Dangerous command blocked: {dangerous}")
        if command in self._validation_commands:
            return PermissionDecision.allow("Command matches a repository validation command")

        tokens = request.command_tokens
        if not tokens:
            return PermissionDecision.deny("Command could not be parsed safely")
        if _has_shell_operator(tokens, command):
            return PermissionDecision.ask("Compound shell commands require approval")
        if _is_read_only_git(tokens):
            return PermissionDecision.allow("Read-only Git command is allowed")
        if _is_read_only_shell(tokens):
            return PermissionDecision.allow("Read-only directory listing command is allowed")
        if _is_test_or_lint(tokens):
            return PermissionDecision.allow("Test or lint command is allowed")
        if _is_install(tokens):
            return PermissionDecision.ask("Dependency installation requires approval")
        if _is_network(tokens):
            return PermissionDecision.ask("Network access requires approval")
        if _is_delete(tokens):
            return PermissionDecision.ask("File deletion requires approval")
        return PermissionDecision.ask("This command is not covered by an automatic allow rule")


class TerminalPermissionPrompt:
    """Rich terminal prompt kept separate from PermissionManager for testing."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        input_fn: InputFunction = input,
    ) -> None:
        self.console = console or Console()
        self.input_fn = input_fn

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        body = "\n".join([
            f"工具：{request.tool_name}",
            f"操作类型：{_effect_label(request.effect)}",
            f"目标范围：{request.scope}",
            f"说明：{_approval_reason(request)}",
        ])
        self.console.print(Panel(Text(body), title="需要你的确认", border_style="yellow"))
        if request.trusted_preview:
            self.console.print("[bold]变更预览（真实 Diff）[/bold]")
            self.console.print(Syntax(request.trusted_preview, "diff", word_wrap=True))

        choices = [
            "请选择：",
            "1. 仅允许这一次",
            "2. 本会话内允许此范围",
        ]
        if request.command_prefix:
            choices.append(f"3. 本会话内允许命令前缀：{' '.join(request.command_prefix)}")
        choices.append("0. 拒绝（默认；直接回车也拒绝）")
        self.console.print("\n".join(choices), markup=False)

        while True:
            try:
                available = "1 / 2 / 3 / 0" if request.command_prefix else "1 / 2 / 0"
                answer = self.input_fn(f"请输入 {available} 后回车 > ").strip().lower()
            except EOFError:
                return PermissionDecision.deny("权限输入已结束")
            if answer in {"", "0", "n", "no", "拒绝"}:
                return PermissionDecision.deny("用户拒绝了本次操作")
            if answer in {"1", "y", "yes", "允许", "允许一次"}:
                return PermissionDecision.allow("用户仅允许本次操作")
            if answer in {"2", "s", "session", "会话"}:
                return PermissionDecision.allow(
                    "用户允许本会话内的同一范围操作",
                    PermissionGrantScope.SESSION,
                )
            if answer in {"3", "p", "prefix", "前缀"} and request.command_prefix:
                return PermissionDecision.allow(
                    "用户允许本会话内的命令前缀",
                    PermissionGrantScope.PREFIX,
                )
            self.console.print(f"[yellow]请输入 {available}；直接回车表示拒绝。[/yellow]")


def command_tokens(command: str) -> tuple[str, ...]:
    """Parse enough shell structure for conservative permission rules."""

    try:
        values = shlex.split(command, posix=False)
    except ValueError:
        return ()
    return tuple(_strip_quotes(value) for value in values if value)


def command_prefix(tokens: Sequence[str]) -> tuple[str, ...]:
    """Return a visible, token-bound prefix suitable for a session grant."""

    if not tokens or any(_contains_shell_syntax(token) for token in tokens):
        return ()
    names = [_command_name(token) for token in tokens]
    first = names[0]
    if _is_python(first) and len(tokens) >= 3 and tokens[1].lower() == "-m":
        length = 4 if names[2] == "pip" and len(tokens) >= 4 else 3
        return tuple(tokens[:length])
    if first in {"git", "npm", "pnpm", "yarn", "cargo", "go", "dotnet"} and len(tokens) >= 2:
        length = 3 if first in {"npm", "pnpm", "yarn"} and names[1] == "run" and len(tokens) >= 3 else 2
        return tuple(tokens[:length])
    return (tokens[0],)


def command_effect(command: str) -> PermissionEffect:
    tokens = command_tokens(command)
    if _is_network(tokens) or _is_install(tokens):
        return PermissionEffect.NETWORK
    return PermissionEffect.EXECUTE


def is_concurrency_safe_read_command(command: object) -> bool:
    """Return True only for a small, cross-platform-safe read-only command set.

    This answers scheduling, not permission.  The shell used by ``BashTool`` is
    platform dependent (``cmd.exe`` on this project's Windows target), so this
    function intentionally accepts only literal Git invocations without shell
    syntax, path operands, revisions, or user-controlled expansion.  Native
    ``read_file``/``grep`` remain the preferred way to inspect repository text.
    """

    if not isinstance(command, str) or not command.strip():
        return False
    # Detect syntax in the original command, not only shlex tokens: cmd.exe and
    # POSIX shells can interpret metacharacters differently.
    if any(marker in command for marker in ("&", "|", ";", ">", "<", "`", "$", "%", "!", "\n", "\r", "(", ")")):
        return False
    tokens = command_tokens(command)
    if not tokens or _has_shell_operator(tokens, command) or any(_contains_shell_syntax(token) for token in tokens):
        return False
    names = [_command_name(token) for token in tokens]
    if len(names) < 2 or names[0] != "git":
        return False
    subcommand = names[1]
    arguments = tuple(token.lower() for token in tokens[2:])
    if subcommand == "status":
        return all(argument in _SAFE_GIT_STATUS_FLAGS for argument in arguments)
    if subcommand == "diff":
        return all(argument in _SAFE_GIT_DIFF_FLAGS for argument in arguments)
    return (subcommand, arguments) in {
        ("rev-parse", ("--show-toplevel",)),
        ("branch", ("--show-current",)),
    }


def dangerous_command_reason(command: str) -> str | None:
    """Catch high-risk POSIX and PowerShell/cmd destructive commands."""

    existing = _check_dangerous(command)
    if existing:
        return existing
    lowered = command.lower()
    patterns = (
        (r"\bremove-item\b(?=.*\s-recurse\b)(?=.*\s-force\b)", "forced recursive PowerShell delete"),
        (r"\b(?:rd|rmdir)\b(?=.*(?:/s|\s-s\b))(?=.*(?:/q|\s-force\b))", "forced recursive directory delete"),
        (r"\b(?:del|erase)\b(?=.*(?:/s|\s-s\b))(?=.*(?:/q|\s-force\b))", "forced recursive file delete"),
        (r"\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f)", "destructive Git cleanup"),
        (r"\b(?:format|diskpart|shutdown|stop-computer)\b", "system-destructive command"),
    )
    for pattern, reason in patterns:
        if re.search(pattern, lowered):
            return reason
    return None


def _has_shell_operator(tokens: Sequence[str], command: str) -> bool:
    return "\n" in command or "\r" in command or any(token in _SHELL_OPERATORS for token in tokens)


def _is_read_only_git(tokens: Sequence[str]) -> bool:
    names = [_command_name(token) for token in tokens]
    if len(names) < 2 or names[0] != "git" or names[1] not in _READ_ONLY_GIT:
        return False
    lowered = [token.lower() for token in tokens[2:]]
    if any(token == "-o" or token.startswith("--output") for token in lowered):
        return False
    return not (
        names[1] == "branch"
        and len(names) > 2
        and not all(value.startswith("-") for value in tokens[2:])
    )


def _is_read_only_shell(tokens: Sequence[str]) -> bool:
    names = [_command_name(token) for token in tokens]
    return bool(names) and names[0] in _READ_ONLY_SHELL


def _is_test_or_lint(tokens: Sequence[str]) -> bool:
    names = [_command_name(token) for token in tokens]
    lowered = [token.lower() for token in tokens]
    if any(flag in lowered for flag in ("--fix", "--write", "--update-snapshots", "-w")):
        return False
    if not names:
        return False
    if _is_python(names[0]) and len(names) >= 3 and lowered[1] == "-m":
        return names[2] in {"pytest", "mypy", "pyright"} or (
            names[2] == "ruff" and len(names) >= 4 and names[3] == "check"
        )
    if names[0] in {"pytest", "mypy", "pyright"}:
        return True
    if names[0] in {"ruff", "eslint"}:
        return len(names) >= 2 and names[1] == "check" if names[0] == "ruff" else True
    if names[0] in {"cargo", "go", "dotnet"}:
        return len(names) >= 2 and names[1] == "test"
    if names[0] in {"npm", "pnpm", "yarn"}:
        return any(name in {"test", "lint"} for name in names[1:3])
    return False


def _is_install(tokens: Sequence[str]) -> bool:
    names = [_command_name(token) for token in tokens]
    if not names:
        return False
    if names[0] in _INSTALL_COMMANDS:
        return True
    if _is_python(names[0]) and len(names) >= 4:
        return names[1:4] == ["-m", "pip", "install"]
    if names[0] == "pip" and len(names) >= 2:
        return names[1] == "install"
    if names[0] in {"npm", "pnpm", "yarn"} and len(names) >= 2:
        return names[1] in {"add", "ci", "install"}
    if names[0] in {"poetry", "uv"} and len(names) >= 2:
        return names[1] in {"add", "install", "sync"}
    return False


def _is_network(tokens: Sequence[str]) -> bool:
    names = [_command_name(token) for token in tokens]
    if not names:
        return False
    if names[0] in _NETWORK_COMMANDS:
        return True
    return len(names) >= 2 and names[0] == "git" and names[1] in {"clone", "fetch", "pull", "push"}


def _is_delete(tokens: Sequence[str]) -> bool:
    names = [_command_name(token) for token in tokens]
    return bool(names) and (
        names[0] in _DELETE_COMMANDS
        or (len(names) >= 2 and names[0] == "git" and names[1] in {"clean", "restore"})
    )


def _command_name(value: str) -> str:
    return PureWindowsPath(_strip_quotes(value)).name.lower().removesuffix(".exe")


def _is_python(value: str) -> bool:
    return value == "py" or value.startswith("python")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _contains_shell_syntax(value: str) -> bool:
    return any(marker in value for marker in ("&", "|", ";", ">", "<", "`", "$(", "\n", "\r"))


def _effect_label(effect: PermissionEffect) -> str:
    return {
        PermissionEffect.WRITE: "写入文件",
        PermissionEffect.EXECUTE: "执行命令",
        PermissionEffect.NETWORK: "访问网络或安装依赖",
    }.get(effect, effect.value)


def _approval_reason(request: PermissionRequest) -> str:
    if request.source_snapshot:
        if request.reason.startswith("Source changed"):
            return "文件在等待确认期间发生变化，请重新核对这份 Diff。"
        return "文件将在你确认后才会写入。"
    if request.tool_name == "bash":
        return "该命令可能改变环境、文件或网络状态，需要你的确认。"
    return "该操作需要你的确认。"
