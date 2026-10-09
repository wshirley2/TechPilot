"""Command-line entry point for TechPilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from techpilot.engine.runtime_control import RuntimeLimits

from . import __version__
from .chat.permissions import TerminalPermissionPrompt
from .chat.session import ChatSession, TerminalEventSink
from .chat.tui import TechPilotTui, tui_supported
from .config.user import load_user_config, run_setup_wizard
from .runtime import RuntimeBootstrap, RuntimeBootstrapInput
from .runtime.sessions import SessionStore
from .safety.trust import confirm_workspace_access

# Keep retired spellings reserved so they cannot become repository paths.
_TOP_LEVEL_COMMANDS = {
    "chat", "run", "status", "profile", "plan", "plans", "workspace", "sessions", "session", "init", "reconfigure",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="techpilot",
        description="A repository-aware Chat and Goal agent.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Show the current TechPilot capabilities.")
    subparsers.add_parser("init", help="Create user-level provider configuration with an interactive wizard.")
    subparsers.add_parser("reconfigure", help="Repair or replace user-level provider configuration.")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive coding-agent chat in a repository.")
    chat_parser.add_argument("repository", nargs="?", type=Path, default=Path("."), help="Repository directory.")
    chat_parser.add_argument("-m", "--model", help="Override the configured model.")
    chat_parser.add_argument("--base-url", help="Override the OpenAI-compatible API base URL.")
    chat_parser.add_argument("--api-key", help="Override the configured API key.")
    chat_parser.add_argument("--resume", help="Resume an event Session for this repository.")
    chat_parser.add_argument("--sessions-dir", type=Path, help="Directory containing event Session JSONL files.")
    chat_parser.add_argument("--tui", action="store_true", help="Use the optional interactive terminal interface.")
    _add_runtime_limits(chat_parser)

    sessions_parser = subparsers.add_parser("sessions", aliases=["session"], help="Inspect recoverable Chat Sessions.")
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_command", required=True)
    sessions_list_parser = sessions_subparsers.add_parser("list", help="List event Sessions for a repository.")
    sessions_list_parser.add_argument("repository", nargs="?", type=Path, default=Path("."))
    sessions_list_parser.add_argument("--sessions-dir", type=Path)
    sessions_show_parser = sessions_subparsers.add_parser("show", help="Show one event Session and recovery warnings.")
    sessions_show_parser.add_argument("session_id")
    sessions_show_parser.add_argument("repository", nargs="?", type=Path, default=Path("."))
    sessions_show_parser.add_argument("--sessions-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_values = list(sys.argv[1:] if argv is None else argv)
    if raw_values[:1] == ["run"]:
        parser.error("Managed Run execution was removed; use `techpilot chat` and `/goal` instead")
    if raw_values[:1] and raw_values[0] in {"plan", "plans", "workspace"}:
        parser.error("Plan and Workspace commands were removed; use `techpilot chat` and `/goal` instead")
    if raw_values[:1] == ["profile"]:
        parser.error("The standalone profile command was removed; repository analysis remains available in Chat")
    args = parser.parse_args(_normalize_command(raw_values))

    if args.command in {"init", "reconfigure"}:
        run_setup_wizard(repairing=args.command == "reconfigure")
        return 0

    # Only the bare command changes its presentation default.  Explicit
    # `techpilot chat` and `techpilot chat --tui` remain compatible.
    if not raw_values:
        state = load_user_config()
        if not state.is_complete:
            if state.problem:
                print(state.problem, file=sys.stderr)
            run_setup_wizard(repairing=state.problem != "配置文件不存在")
        args.tui = True

    if args.command == "status":
        print("TechPilot supports repository-aware Chat and durable Goals.")
        return 0
    if args.command == "chat":
        return _run_chat(parser, args)
    if args.command in {"sessions", "session"}:
        return _run_sessions(parser, args)
    parser.print_help()
    return 0


def _add_runtime_limits(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime limits")
    group.add_argument("--max-provider-calls", type=int, help="Stop a turn after this many provider calls.")
    group.add_argument("--max-tool-rounds", type=int, help="Stop a turn before another tool-call round.")
    group.add_argument("--max-turn-seconds", type=float, help="Stop at the next safe boundary after this duration.")
    group.add_argument("--max-input-tokens", type=int, help="Maximum projected input context tokens.")
    group.add_argument("--max-total-tokens", type=int, help="Maximum provider-reported tokens for one turn.")
    group.add_argument("--max-cost-usd", type=float, help="Maximum estimated provider cost for one turn.")


def _normalize_command(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        return ["chat", "."]
    if values[0] not in _TOP_LEVEL_COMMANDS and not values[0].startswith("-"):
        return ["chat", *values]
    return values


def _run_chat(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    tui = TechPilotTui() if args.tui and tui_supported() else None
    if args.tui and tui is None:
        print("--tui requires an interactive TTY; falling back to the standard Chat CLI.", file=sys.stderr)
    console = Console()
    # Keep the Claude-like workspace gate before RuntimeBootstrap.  The
    # isatty check keeps test/non-interactive invocations from blocking and
    # leaves the ordinary CLI's existing behavior unchanged.
    if tui is not None and sys.stdin.isatty() and not confirm_workspace_access(args.repository, console=console):
        return 0
    sink = tui.event_sink if tui is not None else TerminalEventSink(console)
    bootstrap = RuntimeBootstrap()
    try:
        runtime = bootstrap.build(RuntimeBootstrapInput(
            repository=args.repository,
            event_sink=sink,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            permission_prompt=tui.permission_prompt if tui is not None else TerminalPermissionPrompt(console),
            session_directory=args.sessions_dir,
            resume_session_id=args.resume,
            limits=_runtime_limits_for(args),
        ))
    except ValueError as error:
        parser.error(str(error))
    if tui is not None:
        tui.bind_runtime(runtime)
        return tui.run()
    return ChatSession(runtime, console=console).run()


def _runtime_limits_for(args: argparse.Namespace) -> RuntimeLimits:
    return RuntimeLimits(
        max_provider_calls=getattr(args, "max_provider_calls", None),
        max_tool_rounds=getattr(args, "max_tool_rounds", None),
        max_turn_seconds=getattr(args, "max_turn_seconds", None),
        max_input_tokens=getattr(args, "max_input_tokens", None),
        max_total_tokens=getattr(args, "max_total_tokens", None),
        max_cost_usd=getattr(args, "max_cost_usd", None),
    )


def _run_sessions(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.repository.is_dir():
        parser.error(f"Repository directory does not exist: {args.repository}")
    store = SessionStore.for_repository(args.repository.resolve(), args.sessions_dir)
    if args.sessions_command == "list":
        records = store.list()
        if not records:
            print("No saved Sessions.")
            return 0
        for item in records:
            print(
                f"{item.session_id}\t{item.status}\t{item.model or '?'}\t"
                f"events={len(item.events)}\twarnings={len(item.warnings)}"
            )
        return 0
    try:
        item = store.replay(args.session_id)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    payload = {
        "session_id": item.session_id,
        "repository_root": str(item.repository_root) if item.repository_root else None,
        "model": item.model,
        "status": item.status,
        "last_result": item.last_result.to_dict() if item.last_result else None,
        "event_count": len(item.events),
        "message_count": len(item.messages),
        "model_message_count": len(item.model_messages),
        "prompt_tokens": item.prompt_tokens,
        "completion_tokens": item.completion_tokens,
        "warnings": item.warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
