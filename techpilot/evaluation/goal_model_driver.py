"""Line-oriented Chat driver for the isolated Goal model pilot.

The driver uses the product ChatSession and Runtime. It substitutes a narrow
operator permission prompt so a non-interactive pipe can drive the same Goal
commands without granting arbitrary shell or out-of-scope file effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rich.console import Console

from techpilot.chat.session import ChatSession, TerminalEventSink
from techpilot.config.user import resolve_runtime_config
from techpilot.engine.permissions import PermissionDecision, PermissionEffect, PermissionRequest
from techpilot.engine.runtime_control import RuntimeLimits
from techpilot.evaluation.goal_model_v0 import SUITE, load_deck
from techpilot.runtime.bootstrap import RuntimeBootstrap, RuntimeBootstrapInput
from techpilot.runtime.long_tasks import LongTaskStore


class PilotPermissionPrompt:
    """Approve one trusted diff within the frozen allow-list; deny other asks."""

    def __init__(self, workspace: Path, allowed_paths: set[str]) -> None:
        self.workspace = workspace.resolve()
        self.allowed_paths = allowed_paths

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        if request.effect is not PermissionEffect.WRITE or request.tool_name not in {"edit_file", "write_file"}:
            return PermissionDecision.deny("Pilot operator protocol denies this prompted effect")
        raw_path = request.normalized_arguments.get("file_path")
        if not isinstance(raw_path, str) or not request.trusted_preview:
            return PermissionDecision.deny("Pilot write lacks a path or trusted diff")
        path = Path(raw_path).resolve()
        if not path.is_relative_to(self.workspace):
            return PermissionDecision.deny("Pilot write escapes the case workspace")
        if path.relative_to(self.workspace).as_posix() not in self.allowed_paths:
            return PermissionDecision.deny("Pilot write is outside the frozen allowed paths")
        print(f"[pilot permission] one trusted write: {path.relative_to(self.workspace).as_posix()}", flush=True)
        return PermissionDecision.allow("Frozen pilot contract permits this trusted file diff")


def _line_input(prompt: str) -> str:
    print(prompt, end="", flush=True)
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    print(line.rstrip("\r\n"), flush=True)
    return line


def run_driver(prepared_root: Path, *, case_id: str, deck_path: Path | None = None) -> int:
    deck = load_deck(deck_path) if deck_path else load_deck()
    if case_id not in {case.case_id for case in deck.cases}:
        raise ValueError("Goal pilot case is not part of the frozen deck")
    root = prepared_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("suite") != SUITE
        or manifest.get("record_type") != "preparation"
        or manifest.get("case_set_digest") != deck.digest
        or not isinstance(manifest.get("model"), str)
        or not isinstance(manifest.get("cases"), list)
        or case_id not in manifest["cases"]
    ):
        raise ValueError("Goal pilot manifest does not match this case")
    config = resolve_runtime_config(model=manifest["model"])
    runtime_config = {
        "provider": config.provider,
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens,
        "max_context_tokens": config.max_context_tokens,
        "custom_base_url": bool(config.base_url),
        "base_url_sha256": (
            hashlib.sha256(config.base_url.encode("utf-8")).hexdigest()
            if config.base_url else None
        ),
    }
    if manifest.get("runtime_config") != runtime_config:
        raise ValueError("Goal pilot runtime configuration changed after preparation")
    case_root = (root / case_id).resolve()
    workspace = (case_root / "workspace").resolve()
    sessions = (case_root / "sessions").resolve()
    if not case_root.is_relative_to(root) or not workspace.is_relative_to(case_root) or not workspace.is_dir():
        raise ValueError("Goal pilot workspace is missing or outside the prepared output")
    task = LongTaskStore.for_repository(workspace, session_directory=sessions).replay(f"goal-pilot-{case_id}")
    if task.session_id is None or task.repository_root != workspace:
        raise ValueError("Goal pilot Task is not bound to this case workspace and Session")
    console = Console(force_terminal=False)
    limits = deck.raw["turn_limits"]
    runtime = RuntimeBootstrap().build(RuntimeBootstrapInput(
        repository=workspace,
        event_sink=TerminalEventSink(console),
        model=manifest["model"],
        session_directory=sessions,
        resume_session_id=task.session_id,
        permission_prompt=PilotPermissionPrompt(workspace, set(deck.raw["allowed_paths"])),
        limits=RuntimeLimits(**limits),
    ))
    return ChatSession(runtime, console=console, input_fn=_line_input).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drive one prepared Goal model pilot case through product Chat.")
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--deck", type=Path)
    args = parser.parse_args(argv)
    return run_driver(args.prepared_root, case_id=args.case_id, deck_path=args.deck)


if __name__ == "__main__":
    raise SystemExit(main())
