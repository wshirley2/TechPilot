"""Freeze an auditable, billable-model Goal pilot before any model call.

This module prepares isolated workspaces and a protocol manifest. It never
starts a provider or executes candidate code. Goal execution and external
scoring are separate later phases so a prepared deck cannot imply success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from techpilot.config.user import resolve_runtime_config
from techpilot.engine.events import NullEventSink, RuntimeEventType
from techpilot.runtime.bootstrap import RuntimeBootstrap, RuntimeBootstrapInput
from techpilot.runtime.effect_journal import EffectJournalStore
from techpilot.runtime.goal_plans import GoalPlan
from techpilot.runtime.goal_setup import GoalSetupTool
from techpilot.runtime.long_task_workflow import LongTaskWorkflow
from techpilot.runtime.long_tasks import CompletionCriterionKind, LongTaskStore
from techpilot.runtime.sessions import SessionStore

SUITE = "goal-model-pilot-v0"
SCHEMA_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DECK_PATH = _PROJECT_ROOT / "benchmarks" / "goal_model_v0" / "deck.json"
_KINDS = {"normal", "restart_after_stage", "constraint_after_stage"}
_CASE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")


def _relative_file(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("fixture path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"fixture path escapes its source directory: {value}")
    return path


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class GoalPilotCase:
    case_id: str
    kind: str
    goal: str
    intervention: dict[str, str] | None


@dataclass(frozen=True)
class GoalPilotDeck:
    raw: dict[str, Any]
    cases: tuple[GoalPilotCase, ...]
    fixture_root: Path
    fixture_paths: tuple[PurePosixPath, ...]
    fixture_hashes: dict[str, str]
    digest: str


def load_deck(path: Path = _DECK_PATH) -> GoalPilotDeck:
    """Validate the public pilot contract and hash the exact fixture bytes."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {
        "suite", "schema_version", "fixture", "plan", "completion_criteria",
        "allowed_paths", "external_acceptance", "turn_limits", "protocol_limits", "cases",
    }
    if not isinstance(payload, dict) or frozenset(payload) not in {frozenset(required_fields), frozenset(required_fields | {"model_constraints"})}:
        raise ValueError("Goal pilot deck has unexpected top-level fields")
    if payload["suite"] != SUITE or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Goal pilot suite or schema version is unsupported")
    fixture = payload["fixture"]
    if not isinstance(fixture, dict) or set(fixture) != {"directory", "files"}:
        raise ValueError("Goal pilot fixture requires directory and files")
    fixture_directory = _relative_file(fixture["directory"])
    fixture_root = (_PROJECT_ROOT / fixture_directory).resolve()
    if not fixture_root.is_dir() or not fixture_root.is_relative_to(_PROJECT_ROOT):
        raise ValueError("Goal pilot fixture directory is missing or outside the project")
    raw_files = fixture["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Goal pilot fixture must list source files")
    paths = tuple(_relative_file(item) for item in raw_files)
    if len(set(paths)) != len(paths):
        raise ValueError("Goal pilot fixture has duplicate paths")
    hashes: dict[str, str] = {}
    for item in paths:
        source = (fixture_root / item).resolve()
        if not source.is_relative_to(fixture_root) or not source.is_file():
            raise ValueError(f"Goal pilot fixture file is missing or escapes the fixture: {item}")
        hashes[item.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()

    if not isinstance(payload["plan"], dict) or set(payload["plan"]) != {"reason", "steps"}:
        raise ValueError("Goal pilot plan must contain reason and steps only")
    plan = GoalPlan.from_payload({"version": 1, **payload["plan"]})
    if len(plan.steps) < 2:
        raise ValueError("Goal pilot requires a multi-stage structured plan")
    if any(isinstance(step.check, str) for step in plan.steps):
        raise ValueError("Goal pilot stages require structured checks")
    criteria = payload["completion_criteria"]
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("Goal pilot requires final completion criteria")
    for criterion in criteria:
        if not isinstance(criterion, dict) or set(criterion) != {"kind", "path", "expected_text"}:
            raise ValueError("Goal pilot completion criteria must use kind, path and expected_text")
        if CompletionCriterionKind(criterion["kind"]) is not CompletionCriterionKind.FILE_CONTAINS:
            raise ValueError("Goal pilot v0 supports only file_contains completion criteria")
        _relative_file(criterion["path"])
        if not isinstance(criterion["expected_text"], str) or not criterion["expected_text"].strip():
            raise ValueError("Goal pilot completion text cannot be empty")
    allowed = payload["allowed_paths"]
    if not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("Goal pilot allowed_paths must be a non-empty unique list")
    for item in allowed:
        _relative_file(item)
    constraints = payload.get("model_constraints", [])
    if not isinstance(constraints, list) or len(constraints) > 12 or any(
        not isinstance(item, str) or not item.strip() or len(item) > 2_000 for item in constraints
    ):
        raise ValueError("Goal pilot model constraints must be non-empty strings")
    external = payload["external_acceptance"]
    if not isinstance(external, dict) or set(external) != {
        "default_text_stdout", "explicit_text_stdout", "json_items", "invalid_format",
        "unicode_item", "readme_example", "unchanged_paths",
    }:
        raise ValueError("Goal pilot external acceptance has unexpected fields")
    if any(not isinstance(external[key], str) or not external[key] for key in (
        "default_text_stdout", "explicit_text_stdout", "invalid_format", "unicode_item", "readme_example",
    )):
        raise ValueError("Goal pilot external acceptance requires non-empty expected values")
    if not isinstance(external["json_items"], list) or not external["json_items"] or any(
        not isinstance(item, str) or not item for item in external["json_items"]
    ):
        raise ValueError("Goal pilot JSON acceptance requires non-empty item strings")
    unchanged = external["unchanged_paths"]
    if not isinstance(unchanged, list) or not unchanged or len(set(unchanged)) != len(unchanged):
        raise ValueError("Goal pilot unchanged_paths must be a non-empty unique list")
    for item in unchanged:
        _relative_file(item)
        if item not in hashes:
            raise ValueError("Goal pilot unchanged path must be part of the frozen fixture")
    turn_limits = payload["turn_limits"]
    if not isinstance(turn_limits, dict) or set(turn_limits) != {
        "max_provider_calls", "max_tool_rounds", "max_turn_seconds",
    } or any(type(turn_limits[key]) is not int or turn_limits[key] <= 0 for key in (
        "max_provider_calls", "max_tool_rounds",
    )) or not isinstance(turn_limits["max_turn_seconds"], (int, float)) or (
        isinstance(turn_limits["max_turn_seconds"], bool)
        or not math.isfinite(turn_limits["max_turn_seconds"])
        or turn_limits["max_turn_seconds"] <= 0
    ):
        raise ValueError("Goal pilot turn limits must be positive numbers")
    protocol_limits = payload["protocol_limits"]
    if not isinstance(protocol_limits, dict) or set(protocol_limits) != {
        "max_continuous_batches", "max_work_turns",
    } or any(type(value) is not int or value <= 0 for value in protocol_limits.values()):
        raise ValueError("Goal pilot protocol limits must be positive integers")
    if protocol_limits["max_work_turns"] > 4 * protocol_limits["max_continuous_batches"]:
        raise ValueError("Goal pilot work-turn limit exceeds its continuous-batch capacity")

    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != 3:
        raise ValueError("Goal pilot v0 requires exactly three scenario cases")
    cases: list[GoalPilotCase] = []
    for raw in raw_cases:
        if not isinstance(raw, dict) or set(raw) != {"id", "kind", "goal", "intervention"}:
            raise ValueError("Goal pilot case has unexpected fields")
        case_id, kind, goal, intervention = (
            raw["id"], raw["kind"], raw["goal"], raw["intervention"],
        )
        if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
            raise ValueError("Goal pilot case ID must be a safe lowercase directory name")
        if not isinstance(kind, str) or kind not in _KINDS or not isinstance(goal, str) or not goal.strip():
            raise ValueError("Goal pilot case needs a supported kind and non-empty goal")
        if kind == "normal" and intervention is not None:
            raise ValueError("Normal Goal pilot case cannot have an intervention")
        if kind != "normal":
            if not isinstance(intervention, dict) or set(intervention) != ({"after_stage"} if kind == "restart_after_stage" else {"after_stage", "constraint"}):
                raise ValueError("Goal pilot intervention fields do not match its kind")
            if intervention["after_stage"] not in {step.step_id for step in plan.steps[:-1]}:
                raise ValueError("Goal pilot intervention must follow a non-final plan stage")
            if kind == "constraint_after_stage" and (
                not isinstance(intervention["constraint"], str) or not intervention["constraint"].strip()
            ):
                raise ValueError("Goal pilot injected constraint cannot be empty")
        cases.append(GoalPilotCase(case_id, kind, goal.strip(), intervention))
    if {case.kind for case in cases} != _KINDS or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Goal pilot must contain one unique case of each scenario kind")
    digest = hashlib.sha256(_canonical({"deck": payload, "fixture_hashes": hashes})).hexdigest()
    return GoalPilotDeck(payload, tuple(cases), fixture_root, paths, hashes, digest)


def prepare_pilot(
    output: Path, *, model: str, max_cost_usd: float | None, max_total_tokens: int | None,
    deck: GoalPilotDeck | None = None,
) -> Path:
    """Create a frozen protocol and isolated workspaces; never call a model."""

    if not model.strip():
        raise ValueError("Goal pilot preparation requires an explicit model ID")
    if max_cost_usd is None and max_total_tokens is None:
        raise ValueError("Goal pilot preparation requires a cost or token budget")
    if max_cost_usd is not None and (not math.isfinite(max_cost_usd) or max_cost_usd <= 0):
        raise ValueError("Goal pilot max_cost_usd must be finite and positive")
    if max_total_tokens is not None and (type(max_total_tokens) is not int or max_total_tokens <= 0):
        raise ValueError("Goal pilot max_total_tokens must be a positive integer")
    selected = deck or load_deck()
    runtime_config = resolve_runtime_config(model=model.strip())
    destination = output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "track": "model",
        "record_type": "preparation",
        "case_set_digest": selected.digest,
        "fixture_hashes": selected.fixture_hashes,
        "model": model.strip(),
        "runtime_config": {
            "provider": runtime_config.provider,
            "temperature": runtime_config.temperature,
            "max_output_tokens": runtime_config.max_tokens,
            "max_context_tokens": runtime_config.max_context_tokens,
            "custom_base_url": bool(runtime_config.base_url),
            "base_url_sha256": (
                hashlib.sha256(runtime_config.base_url.encode("utf-8")).hexdigest()
                if runtime_config.base_url else None
            ),
        },
        "max_cost_usd": max_cost_usd,
        "max_total_tokens": max_total_tokens,
        "budget_semantics": "administrative suite ceiling; Runtime limits apply per turn, not as a hard Goal-wide cap",
        "cases": [case.case_id for case in selected.cases],
    }
    for case in selected.cases:
        case_root = destination / case.case_id
        workspace = case_root / "workspace"
        workspace.mkdir(parents=True)
        for item in selected.fixture_paths:
            target = workspace / item
            target.parent.mkdir(parents=True, exist_ok=True)
            content = (selected.fixture_root / item).read_bytes()
            if hashlib.sha256(content).hexdigest() != selected.fixture_hashes[item.as_posix()]:
                raise ValueError(f"Goal pilot fixture changed after the deck was loaded: {item}")
            target.write_bytes(content)
        contract = {
            "case_id": case.case_id,
            "kind": case.kind,
            "goal": case.goal,
            "intervention": case.intervention,
            "plan": selected.raw["plan"],
            "completion_criteria": selected.raw["completion_criteria"],
            "allowed_paths": selected.raw["allowed_paths"],
            "external_acceptance": selected.raw["external_acceptance"],
            "turn_limits": selected.raw["turn_limits"],
            "protocol_limits": selected.raw["protocol_limits"],
            "fixture_hashes": selected.fixture_hashes,
        }
        if "model_constraints" in selected.raw:
            contract["model_constraints"] = selected.raw["model_constraints"]
        (case_root / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def setup_pilot_case(
    prepared_root: Path, *, case_id: str, deck: GoalPilotDeck | None = None,
) -> dict[str, str]:
    """Persist the frozen human-authored Goal contract without a model call."""

    selected = deck or load_deck()
    root = prepared_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    case = next((item for item in selected.cases if item.case_id == case_id), None)
    if (
        case is None or not isinstance(manifest, dict)
        or manifest.get("record_type") != "preparation"
        or manifest.get("suite") != SUITE
        or manifest.get("case_set_digest") != selected.digest
        or not isinstance(manifest.get("model"), str)
        or not isinstance(manifest.get("cases"), list)
        or case_id not in manifest["cases"]
    ):
        raise ValueError("Goal pilot case does not match the prepared frozen deck")
    config = resolve_runtime_config(model=manifest["model"])
    current_runtime_config = {
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
    if manifest.get("runtime_config") != current_runtime_config:
        raise ValueError("Goal pilot runtime configuration changed after preparation")
    case_root = (root / case_id).resolve()
    workspace = (case_root / "workspace").resolve()
    sessions = (case_root / "sessions").resolve()
    if not case_root.is_relative_to(root) or not workspace.is_relative_to(case_root) or not workspace.is_dir():
        raise ValueError("Goal pilot workspace is missing or outside the prepared output")
    if sessions.exists() or (workspace / ".techpilot" / "tasks").exists():
        raise FileExistsError("Goal pilot case already has Session or Task state")
    runtime = RuntimeBootstrap().build(RuntimeBootstrapInput(
        repository=workspace,
        event_sink=NullEventSink(),
        model=manifest["model"],
        session_directory=sessions,
    ))
    task_id = f"goal-pilot-{case_id}"
    proposal = GoalSetupTool._validate(
        case.goal, selected.raw.get("model_constraints", []), selected.raw["completion_criteria"], selected.raw["plan"],
    )
    workflow = LongTaskWorkflow(runtime, task_id=task_id, goal=case.goal)
    workflow.create_for_setup()
    task = workflow.accept_setup(proposal)
    runtime.ensure_persisted()
    return {"case_id": case_id, "task_id": task.task_id, "session_id": task.session_id or ""}


def collect_pilot_snapshot(
    prepared_root: Path, *, case_id: str, task_id: str, label: str,
    deck: GoalPilotDeck | None = None,
) -> Path:
    """Read durable Goal facts into a new observation, without scoring code."""

    if not isinstance(label, str) or not _CASE_ID.fullmatch(label):
        raise ValueError("Goal pilot observation label must be a safe lowercase name")
    selected = deck or load_deck()
    root = prepared_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("record_type") != "preparation"
        or manifest.get("suite") != SUITE
        or manifest.get("case_set_digest") != selected.digest
        or not isinstance(manifest.get("cases"), list)
    ):
        raise ValueError("Goal pilot manifest does not match this frozen deck")
    if case_id not in {case.case_id for case in selected.cases} or case_id not in manifest["cases"]:
        raise ValueError("Goal pilot case is not part of the prepared manifest")
    case_root = root / case_id
    workspace = (case_root / "workspace").resolve()
    session_directory = (case_root / "sessions").resolve()
    if not workspace.is_dir() or not workspace.is_relative_to(root):
        raise ValueError("Goal pilot workspace is missing or outside the prepared output")
    task = LongTaskStore.for_repository(workspace, session_directory=session_directory).replay(task_id)
    if task.repository_root != workspace or task.session_id is None:
        raise ValueError("Goal pilot Task belongs to another workspace or has no Session")
    session = SessionStore(session_directory).replay(task.session_id)
    if session.repository_root != workspace or session.model != manifest["model"]:
        raise ValueError("Goal pilot Session identity does not match the prepared protocol")
    journal = EffectJournalStore.for_session_directory(session_directory).replay(task.session_id)
    task_events = task.events[:task.applied_event_count]
    stop_events = [event.payload for event in task_events if event.event_type == "goal_run_stopped"]
    stop_turns = [payload.get("turns") for payload in stop_events]
    stop_tokens = [payload.get("tokens") for payload in stop_events]
    stop_costs = [payload.get("cost_usd") for payload in stop_events]
    tool_requests = [event.tool_call_id for event in session.events if event.event_type == RuntimeEventType.TOOL_REQUESTED.value]
    tool_completions = [event.tool_call_id for event in session.events if event.event_type == RuntimeEventType.TOOL_COMPLETED.value]
    repeated_ids = sorted(call_id for call_id, count in Counter(tool_requests).items() if call_id and count > 1)
    try:
        first_event_at = datetime.fromisoformat(session.events[0].created_at) if session.events else None
        last_event_at = datetime.fromisoformat(session.events[-1].created_at) if session.events else None
        elapsed_wall_seconds = (
            max(0.0, (last_event_at - first_event_at).total_seconds())
            if first_event_at is not None and last_event_at is not None else None
        )
    except (TypeError, ValueError):
        elapsed_wall_seconds = None
    plan = task.plan
    observed = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "track": "model",
        "case_id": case_id,
        "case_set_digest": selected.digest,
        "label": label,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task.task_id,
        "session_id": task.session_id,
        "task_status": task.status.value,
        "plan_version": plan.version if plan else None,
        "planned_stages": len(plan.steps) if plan else 0,
        "verified_stages": list(task.verified_step_ids),
        "selected_stage": task.selected_step_id,
        "ready_stages": list(task.ready_step_ids),
        "stage_assessments": [
            {"step_id": item.step_id, "status": item.status.value, "checkpoint_id": item.checkpoint_id}
            for item in task.stage_assessments
        ],
        "checkpoints": len(task.checkpoints),
        "task_event_count": task.applied_event_count,
        "session_event_count": len(session.events),
        "completion_claim_present": task.active_completion_claim is not None,
        "completion_verification_passed": (
            task.last_completion_verification.passed if task.last_completion_verification else None
        ),
        "stop_reasons": [payload.get("reason") for payload in stop_events],
        "observed_continuous_batches": len(stop_events),
        "observed_continuous_work_turns": (
            sum(stop_turns) if all(type(value) is int for value in stop_turns) else None
        ),
        "last_continuous_stop": task.last_continuous_stop,
        "provider_calls": session.provider_calls,
        "tool_requests": len(tool_requests),
        "tool_completions": len(tool_completions),
        "repeated_tool_call_ids": repeated_ids,
        "context_compressions": sum(event.event_type == RuntimeEventType.CONTEXT_COMPRESSED.value for event in session.events),
        "wall_elapsed_seconds_including_idle": elapsed_wall_seconds,
        "session_resumes": sum(event.event_type == "session_resumed" for event in session.events),
        "task_resumes": sum(event.event_type == "task_resumed" for event in task_events),
        "confirmed_constraints": len(task.active_constraints),
        "active_constraint_texts": [item.text for item in task.active_constraints],
        "observed_continuous_tokens": sum(stop_tokens) if stop_tokens and all(type(value) is int for value in stop_tokens) else None,
        "observed_continuous_cost_usd": (
            sum(stop_costs) if stop_costs and all(type(value) in {int, float} for value in stop_costs) else None
        ),
        "completed_turn_tokens_lower_bound": session.total_tokens,
        "ambiguous_effect_call_ids": [entry.tool_call_id for entry in journal.ambiguous_entries],
        "task_recovery_blockers": list(task.recovery_blockers),
        "task_warnings": list(task.warnings),
        "session_warnings": list(session.warnings),
        "effect_journal_warnings": list(journal.warnings),
        "independent_acceptance": None,
        "failure_attribution": None,
    }
    observations = case_root / "observations"
    observations.mkdir(exist_ok=True)
    path = observations / f"{label}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(observed, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect, prepare, set up or collect the Goal model pilot without calling a provider.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true", help="Print the frozen case IDs and digest only.")
    mode.add_argument("--prepare", type=Path, help="Create this new, isolated output directory.")
    mode.add_argument("--setup-case", type=Path, help="Persist one frozen Goal setup without a model call.")
    mode.add_argument("--collect", type=Path, help="Read a prepared output and collect one durable Goal snapshot.")
    parser.add_argument("--model", help="Explicit model ID frozen in a prepared protocol.")
    parser.add_argument("--max-cost-usd", type=float, help="Administrative suite cost ceiling in USD.")
    parser.add_argument("--max-total-tokens", type=int, help="Administrative suite token ceiling.")
    parser.add_argument("--case-id", help="Prepared case ID used with --collect.")
    parser.add_argument("--task-id", help="Durable Goal Task ID used with --collect.")
    parser.add_argument("--label", help="New observation label used with --collect.")
    parser.add_argument("--deck", type=Path, help="Explicit versioned deck path; defaults to frozen v0.")
    args = parser.parse_args(argv)
    deck = load_deck(args.deck) if args.deck else load_deck()
    if args.inspect:
        print(json.dumps({"suite": SUITE, "case_set_digest": deck.digest, "cases": [
            {"id": case.case_id, "kind": case.kind} for case in deck.cases
        ]}, ensure_ascii=False, indent=2))
        return 0
    if args.collect is not None:
        if not args.case_id or not args.task_id or not args.label:
            parser.error("--collect requires --case-id, --task-id and --label")
        print(collect_pilot_snapshot(
            args.collect, case_id=args.case_id, task_id=args.task_id, label=args.label, deck=deck,
        ))
        return 0
    if args.setup_case is not None:
        if not args.case_id:
            parser.error("--setup-case requires --case-id")
        print(json.dumps(setup_pilot_case(args.setup_case, case_id=args.case_id, deck=deck), ensure_ascii=False, indent=2))
        return 0
    if not args.model:
        parser.error("--prepare requires --model")
    path = prepare_pilot(
        args.prepare, model=args.model,
        max_cost_usd=args.max_cost_usd, max_total_tokens=args.max_total_tokens, deck=deck,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
