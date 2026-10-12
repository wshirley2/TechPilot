"""Append-only operator protocol for the Goal model pilot.

Records refer to separately collected durable observations and independent
assessments. This module never starts a provider or authorizes a model batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from techpilot.evaluation.goal_model_oracle import candidate_digest_for_workspace
from techpilot.evaluation.goal_model_v0 import SCHEMA_VERSION, SUITE, GoalPilotDeck, load_deck

_LABEL = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_PHASES = {
    "start", "after_batch", "before_restart", "after_restart",
    "before_constraint", "after_constraint", "final",
}
_FAILURES = {"evidence", "planning", "tool", "permission", "recovery", "budget", "model_execution", "undetermined"}
_RESTART_ACCEPTANCE_MODES = {"day_later", "cross_process"}


def _digest(value: object) -> str:
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _prepared(root: Path, deck: GoalPilotDeck) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("suite") != SUITE
        or manifest.get("record_type") != "preparation"
        or manifest.get("case_set_digest") != deck.digest
        or manifest.get("cases") != [case.case_id for case in deck.cases]
        or not isinstance(manifest.get("runtime_config"), dict)
    ):
        raise ValueError("Goal pilot manifest does not match the frozen case set")
    return manifest


def _evidence(root: Path, case_id: str, folder: str, label: str, deck: GoalPilotDeck) -> dict[str, Any]:
    if not _LABEL.fullmatch(label):
        raise ValueError("Goal pilot evidence label must be a safe lowercase name")
    path = root / case_id / folder / f"{label}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("suite") != SUITE
        or payload.get("track") != "model"
        or payload.get("case_id") != case_id
        or payload.get("case_set_digest") != deck.digest
        or payload.get("label") != label
    ):
        raise ValueError("Goal pilot evidence does not match this case and deck")
    return payload


def _case_workspace(root: Path, case_id: str) -> Path:
    case_root = (root / case_id).resolve()
    workspace = (case_root / "workspace").resolve()
    if not case_root.is_relative_to(root) or not workspace.is_relative_to(case_root) or not workspace.is_dir():
        raise ValueError("Goal pilot workspace is missing or outside the prepared output")
    return workspace


def _records(root: Path) -> list[dict[str, Any]]:
    directory = (root / "protocol").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Goal pilot protocol directory escapes prepared output")
    if not directory.exists():
        return []
    paths = sorted(directory.glob("*.json"))
    records = []
    previous_hash = None
    for index, path in enumerate(paths, start=1):
        if path.name != f"{index:04d}.json":
            raise ValueError("Goal pilot protocol record sequence is not contiguous")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("sequence") != index:
            raise ValueError("Goal pilot protocol record has an invalid sequence")
        recorded_hash = record.get("record_hash")
        unsigned = {key: value for key, value in record.items() if key != "record_hash"}
        if record.get("previous_hash") != previous_hash or recorded_hash != _digest(unsigned):
            raise ValueError("Goal pilot protocol record hash chain is invalid")
        records.append(record)
        previous_hash = recorded_hash
    return records


def _verified_records(root: Path, deck: GoalPilotDeck) -> list[dict[str, Any]]:
    records = _records(root)
    known_cases = {case.case_id for case in deck.cases}
    for record in records:
        case_id = record.get("case_id")
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("suite") != SUITE
            or record.get("record_type") != "operator_boundary"
            or record.get("case_set_digest") != deck.digest
            or case_id not in known_cases
        ):
            raise ValueError("Goal pilot protocol record identity is invalid")
        observation = _evidence(root, case_id, "observations", record["observation_label"], deck)
        if _digest(observation) != record["observation_digest"]:
            raise ValueError("Goal pilot protocol observation changed after recording")
        if record["assessment_label"] is not None:
            assessment = _evidence(root, case_id, "assessments", record["assessment_label"], deck)
            if _digest(assessment) != record["assessment_digest"]:
                raise ValueError("Goal pilot protocol assessment changed after recording")
            workspace = _case_workspace(root, case_id)
            if assessment.get("candidate_digest") != candidate_digest_for_workspace(workspace, deck=deck):
                raise ValueError("Goal pilot candidate changed after independent acceptance")
    return records


def _nonnegative_cost(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("Reported suite cost must be a finite non-negative number")
    return value


def _nonnegative_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("Reported suite tokens must be a non-negative integer")
    return value


def _latest_meter(records: list[dict[str, Any]], field: str) -> float | int | None:
    return next((record[field] for record in reversed(records) if record[field] is not None), None)


def _require_observation_shape(observation: dict[str, Any]) -> None:
    for name in ("task_id", "session_id"):
        if not isinstance(observation.get(name), str) or not observation[name]:
            raise ValueError(f"Goal pilot observation lacks {name}")
    if not isinstance(observation.get("task_status"), str):
        raise TypeError("Goal pilot observation lacks task_status")
    for name in (
        "task_event_count", "session_event_count", "observed_continuous_batches",
        "observed_continuous_work_turns", "session_resumes", "confirmed_constraints", "checkpoints",
    ):
        if type(observation.get(name)) is not int or observation[name] < 0:
            raise ValueError(f"Goal pilot observation lacks a valid {name}")
    if not isinstance(observation.get("verified_stages"), list) or not isinstance(
        observation.get("active_constraint_texts"), list
    ):
        raise TypeError("Goal pilot observation lacks stage or constraint facts")


def record_boundary(
    prepared_root: Path, *, case_id: str, phase: str, observation_label: str,
    process_instance: str | None = None, assessment_label: str | None = None,
    reported_suite_cost_usd: float | None = None, reported_suite_tokens: int | None = None,
    failure_attribution: str | None = None, operator_note: str = "",
    unplanned_human_interventions: int = 0, status_card_next_action: str | None = None,
    next_step_accurate: bool | None = None,
    restart_acceptance_mode: str | None = None,
    deck: GoalPilotDeck | None = None,
) -> Path:
    """Bind one operator action to existing durable evidence in a new record."""

    selected = deck or load_deck()
    root = prepared_root.expanduser().resolve()
    manifest = _prepared(root, selected)
    case = next((item for item in selected.cases if item.case_id == case_id), None)
    if case is None or phase not in _PHASES:
        raise ValueError("Goal pilot case or phase is not supported")
    observation = _evidence(root, case_id, "observations", observation_label, selected)
    _require_observation_shape(observation)
    if not isinstance(operator_note, str) or len(operator_note) > 2000:
        raise ValueError("Goal pilot operator note must be at most 2000 characters")
    if type(unplanned_human_interventions) is not int or unplanned_human_interventions < 0:
        raise ValueError("Goal pilot unplanned human interventions must be a non-negative integer")
    if status_card_next_action is not None and (
        not isinstance(status_card_next_action, str) or not status_card_next_action.strip()
        or len(status_card_next_action) > 500
    ):
        raise ValueError("Goal pilot status-card next action must be 1–500 characters")
    if next_step_accurate is not None and type(next_step_accurate) is not bool:
        raise ValueError("Goal pilot next-step accuracy must be true or false")
    cost = _nonnegative_cost(reported_suite_cost_usd)
    tokens = _nonnegative_tokens(reported_suite_tokens)
    records = _verified_records(root, selected)
    case_records = [record for record in records if record["case_id"] == case_id]
    prior = case_records[-1] if case_records else None
    prior_observation = (
        _evidence(root, case_id, "observations", prior["observation_label"], selected)
        if prior else None
    )
    start_observation = (
        _evidence(root, case_id, "observations", case_records[0]["observation_label"], selected)
        if case_records else observation
    )
    if phase == "start":
        if prior is not None:
            raise ValueError("Goal pilot case already has a start record")
        if observation["task_status"] != "running":
            raise ValueError("Goal pilot start requires a running Task after setup")
    elif prior is None or prior["phase"] == "final":
        raise ValueError("Goal pilot case must start first and cannot continue after final")
    if prior is not None:
        if observation["task_id"] != prior["task_id"] or observation["session_id"] != prior["session_id"]:
            raise ValueError("Goal pilot Task/Session identity changed between boundaries")
        if observation_label == prior["observation_label"] or any(
            record["observation_label"] == observation_label for record in case_records
        ):
            raise ValueError("Goal pilot boundary requires a new observation")
        for field in (
            "task_event_count", "session_event_count", "observed_continuous_batches",
            "observed_continuous_work_turns", "session_resumes", "confirmed_constraints", "checkpoints",
        ):
            if observation[field] < prior_observation[field]:
                raise ValueError(f"Goal pilot durable {field} moved backwards")
    limits = selected.raw["protocol_limits"]
    work_turns_since_start = observation["checkpoints"] - start_observation["checkpoints"]
    if work_turns_since_start < 0:
        raise ValueError("Goal pilot checkpoint count is earlier than case start")
    if phase == "after_batch" and (
        observation["observed_continuous_batches"] <= prior_observation["observed_continuous_batches"]
    ):
        raise ValueError("Goal pilot after_batch needs a new durable /goal run stop")
    if phase in {"after_batch", "final"}:
        if manifest.get("max_cost_usd") is not None and cost is None:
            raise ValueError("Goal pilot batch/final needs a reported suite cost review")
        if manifest.get("max_total_tokens") is not None and tokens is None:
            raise ValueError("Goal pilot batch/final needs a reported suite token review")
    if phase in {"before_restart", "after_restart"} and case.kind != "restart_after_stage":
        raise ValueError("Goal pilot restart markers belong only to the restart case")
    if phase in {"before_constraint", "after_constraint"} and case.kind != "constraint_after_stage":
        raise ValueError("Goal pilot constraint markers belong only to the constraint case")
    if phase in {"before_restart", "before_constraint"}:
        if "inspect" not in observation["verified_stages"] or "implement" in observation["verified_stages"]:
            raise ValueError("Goal pilot intervention must follow verified inspect and precede implement")
        if any(record["phase"] == phase for record in case_records):
            raise ValueError("Goal pilot intervention boundary was already recorded")
    if phase in {"before_restart", "after_restart"}:
        if not isinstance(process_instance, str) or not process_instance.strip():
            raise ValueError("Goal pilot restart requires a process instance label")
    elif process_instance is not None:
        raise ValueError("Goal pilot process instance is only used for restart evidence")
    if phase == "before_restart":
        if restart_acceptance_mode not in _RESTART_ACCEPTANCE_MODES:
            raise ValueError("Goal pilot before_restart requires an explicit restart acceptance mode")
    elif restart_acceptance_mode is not None:
        raise ValueError("Goal pilot restart acceptance mode belongs to before_restart only")
    if phase == "after_restart":
        if prior["phase"] != "before_restart" or process_instance == prior["process_instance"]:
            raise ValueError("Goal pilot restart needs a different process instance after before_restart")
        if observation["session_resumes"] <= prior_observation["session_resumes"]:
            raise ValueError("Goal pilot restart lacks a durable Session resume event")
        if "inspect" not in observation["verified_stages"]:
            raise ValueError("Goal pilot restart lost its verified inspect stage")
        if "implement" not in observation.get("ready_stages", []) and observation.get("selected_stage") != "implement":
            raise ValueError("Goal pilot restart does not point to the next implementation stage")
        if not status_card_next_action or next_step_accurate is None:
            raise ValueError("Goal pilot restart needs the observed status-card action and accuracy judgment")
    elif next_step_accurate is not None:
        raise ValueError("Goal pilot next-step accuracy belongs to after_restart only")
    if phase == "after_constraint":
        if prior["phase"] != "before_constraint":
            raise ValueError("Goal pilot constraint acceptance needs before_constraint first")
        expected = case.intervention["constraint"]
        if expected in prior_observation["active_constraint_texts"] or expected not in observation["active_constraint_texts"]:
            raise ValueError("Goal pilot new constraint is not verified in durable Task state")
    assessment = None
    if phase == "final":
        if not assessment_label:
            raise ValueError("Goal pilot final boundary requires external acceptance evidence")
        assessment = _evidence(root, case_id, "assessments", assessment_label, selected)
        if assessment.get("record_type") != "external_acceptance" or type(assessment.get("passed")) is not bool:
            raise ValueError("Goal pilot final evidence is not an independent assessment")
        workspace = _case_workspace(root, case_id)
        if assessment.get("candidate_digest") != candidate_digest_for_workspace(workspace, deck=selected):
            raise ValueError("Goal pilot candidate changed after its independent assessment")
        if not assessment["passed"] and failure_attribution not in _FAILURES:
            raise ValueError("Goal pilot failed case requires a failure attribution")
    elif assessment_label is not None or failure_attribution is not None:
        raise ValueError("Goal pilot assessment and failure attribution belong to final only")
    for field, value in (("reported_suite_cost_usd", cost), ("reported_suite_tokens", tokens)):
        previous = _latest_meter(records, field)
        if value is not None and previous is not None and value < previous:
            raise ValueError(f"Goal pilot {field} cannot decrease across the suite")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "record_type": "operator_boundary",
        "case_set_digest": selected.digest,
        "sequence": len(records) + 1,
        "case_id": case_id,
        "phase": phase,
        "observation_label": observation_label,
        "observation_digest": _digest(observation),
        "task_id": observation["task_id"],
        "session_id": observation["session_id"],
        "process_instance": process_instance,
        "restart_acceptance_mode": restart_acceptance_mode,
        "assessment_label": assessment_label,
        "assessment_digest": _digest(assessment) if assessment is not None else None,
        "reported_suite_cost_usd": cost,
        "reported_suite_tokens": tokens,
        "failure_attribution": failure_attribution,
        "operator_note": operator_note,
        "work_turns_since_start": work_turns_since_start,
        "protocol_limits_exceeded": (
            observation["observed_continuous_batches"] > limits["max_continuous_batches"]
            or work_turns_since_start > limits["max_work_turns"]
        ),
        "unplanned_human_interventions": unplanned_human_interventions,
        "status_card_next_action": status_card_next_action,
        "next_step_accurate": next_step_accurate,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "previous_hash": records[-1]["record_hash"] if records else None,
    }
    record = {**unsigned, "record_hash": _digest(unsigned)}
    directory = root / "protocol"
    directory.mkdir(exist_ok=True)
    path = directory / f"{record['sequence']:04d}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def build_report(prepared_root: Path, *, deck: GoalPilotDeck | None = None) -> dict[str, Any]:
    """Summarize only recorded evidence; leave absent results explicitly pending."""

    selected = deck or load_deck()
    root = prepared_root.expanduser().resolve()
    manifest = _prepared(root, selected)
    records = _verified_records(root, selected)
    cases = []
    for case in selected.cases:
        case_records = [record for record in records if record["case_id"] == case.case_id]
        latest = case_records[-1] if case_records else None
        observation = _evidence(root, case.case_id, "observations", latest["observation_label"], selected) if latest else None
        final = next((record for record in reversed(case_records) if record["phase"] == "final"), None)
        assessment = (
            _evidence(root, case.case_id, "assessments", final["assessment_label"], selected)
            if final else None
        )
        before = next((record for record in case_records if record["phase"] == "before_restart"), None)
        after = next((record for record in case_records if record["phase"] == "after_restart"), None)
        after_observation = (
            _evidence(root, case.case_id, "observations", after["observation_label"], selected)
            if after else None
        )
        restart_acceptance_mode = before.get("restart_acceptance_mode", "day_later") if before else None
        restart_delay_hours = None
        if before and after:
            restart_delay_hours = (
                datetime.fromisoformat(after["recorded_at"]) - datetime.fromisoformat(before["recorded_at"])
            ).total_seconds() / 3600
        missing = []
        if not case_records:
            missing.append("start_boundary")
        if case.kind == "restart_after_stage" and (not before or not after):
            missing.append("restart_evidence")
        if case.kind == "restart_after_stage" and restart_acceptance_mode != "cross_process" and (
            restart_delay_hours is None or restart_delay_hours < 24
        ):
            missing.append("day_later_resume_evidence")
        if case.kind == "restart_after_stage" and (
            not after or after["next_step_accurate"] is not True
        ):
            missing.append("accurate_next_step_after_restart")
        if case.kind == "constraint_after_stage" and not any(
            record["phase"] == "after_constraint" for record in case_records
        ):
            missing.append("confirmed_constraint_evidence")
        if final is None:
            missing.append("final_boundary_and_assessment")
        has_continuous_batch = any(record["phase"] == "after_batch" for record in case_records)
        has_post_restart_work = (
            case.kind == "restart_after_stage"
            and final is not None
            and after_observation is not None
            and observation["checkpoints"] > after_observation["checkpoints"]
        )
        execution_evidence_mode = (
            "continuous_batch" if has_continuous_batch else
            "post_restart_work_turn" if has_post_restart_work else None
        )
        if final and execution_evidence_mode is None:
            missing.append("continuous_execution_evidence")
        if any(record["protocol_limits_exceeded"] for record in case_records):
            missing.append("frozen_protocol_limits_exceeded")
        if final and final["failure_attribution"] == "undetermined":
            missing.append("resolved_failure_attribution")
        cases.append({
            "case_id": case.case_id,
            "kind": case.kind,
            "recorded_boundaries": len(case_records),
            "task_id": latest["task_id"] if latest else None,
            "session_id": latest["session_id"] if latest else None,
            "task_status": observation.get("task_status") if observation else None,
            "verified_stages": observation.get("verified_stages") if observation else None,
            "planned_stages": observation.get("planned_stages") if observation else None,
            "completion_verification_passed": observation.get("completion_verification_passed") if observation else None,
            "completion_claim_present": observation.get("completion_claim_present") if observation else None,
            "external_acceptance_passed": assessment["passed"] if assessment else None,
            "false_completion_claim": (
                (
                    observation.get("completion_claim_present") is True
                    or observation.get("completion_verification_passed") is True
                ) and assessment["passed"] is False
                if observation and assessment else None
            ),
            "repeated_tool_call_ids": observation.get("repeated_tool_call_ids") if observation else None,
            "provider_calls": observation.get("provider_calls") if observation else None,
            "context_compressions": observation.get("context_compressions") if observation else None,
            "observed_continuous_batches": observation.get("observed_continuous_batches") if observation else None,
            "execution_evidence_mode": execution_evidence_mode,
            "observed_continuous_work_turns": observation.get("observed_continuous_work_turns") if observation else None,
            "work_turns_since_start": latest["work_turns_since_start"] if latest else None,
            "observed_continuous_tokens": observation.get("observed_continuous_tokens") if observation else None,
            "observed_continuous_cost_usd": observation.get("observed_continuous_cost_usd") if observation else None,
            "elapsed_wall_seconds_including_idle": observation.get("wall_elapsed_seconds_including_idle") if observation else None,
            "restart_delay_hours": restart_delay_hours,
            "restart_acceptance_mode": restart_acceptance_mode,
            "restart_next_step_accurate": after["next_step_accurate"] if after else None,
            "restart_status_card_next_action": after["status_card_next_action"] if after else None,
            "planned_interventions": sum(record["phase"] in {
                "before_restart", "before_constraint",
            } for record in case_records),
            "unplanned_human_interventions": sum(
                record["unplanned_human_interventions"] for record in case_records
            ),
            "failure_attribution": final["failure_attribution"] if final else None,
            "missing_evidence": missing,
        })
    cost = _latest_meter(records, "reported_suite_cost_usd")
    tokens = _latest_meter(records, "reported_suite_tokens")
    max_cost = manifest.get("max_cost_usd")
    max_tokens = manifest.get("max_total_tokens")
    budget_evidence_complete = (max_cost is None or cost is not None) and (max_tokens is None or tokens is not None)
    budget_within_ceiling = budget_evidence_complete and (
        max_cost is None or cost <= max_cost
    ) and (max_tokens is None or tokens <= max_tokens)
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "record_type": "pilot_report",
        "case_set_digest": selected.digest,
        "model": manifest["model"],
        "prepared_runtime_config": manifest["runtime_config"],
        "reported_suite_cost_usd": cost,
        "reported_suite_tokens": tokens,
        "max_cost_usd": max_cost,
        "max_total_tokens": max_tokens,
        "usage_accounting": "Known provider-reported usage only; failed requests may lack usage data",
        "budget_evidence_complete": budget_evidence_complete,
        "budget_within_ceiling": budget_within_ceiling,
        "cases": cases,
        "protocol_evidence_complete": budget_within_ceiling and all(not case["missing_evidence"] for case in cases),
        "interpretation": "Protocol evidence only; three cases cannot estimate general model success rate",
    }


def review_next_batch(
    prepared_root: Path, *, case_id: str, deck: GoalPilotDeck | None = None,
) -> dict[str, Any]:
    """Return a conservative manual budget review, never permission to call a model."""

    selected = deck or load_deck()
    root = prepared_root.expanduser().resolve()
    manifest = _prepared(root, selected)
    if case_id not in {case.case_id for case in selected.cases}:
        raise ValueError("Goal pilot case is not part of the frozen deck")
    records = _verified_records(root, selected)
    case_records = [record for record in records if record["case_id"] == case_id]
    latest = case_records[-1] if case_records else None
    observation = (
        _evidence(root, case_id, "observations", latest["observation_label"], selected)
        if latest else None
    )
    cost = _latest_meter(records, "reported_suite_cost_usd")
    tokens = _latest_meter(records, "reported_suite_tokens")
    reasons = []
    if latest is None:
        reasons.append("case_has_no_start_boundary")
    elif latest["phase"] in {"before_restart", "before_constraint", "final"}:
        reasons.append(f"case_at_{latest['phase']}_boundary")
    if latest and latest["phase"] == "after_restart" and latest["next_step_accurate"] is not True:
        reasons.append("next_step_after_restart_is_inaccurate")
    if observation is not None:
        limits = selected.raw["protocol_limits"]
        if observation["task_status"] != "running":
            reasons.append(f"task_not_running_{observation['task_status']}")
        selected_stage = observation.get("selected_stage")
        selected_assessment = next((
            item for item in reversed(observation.get("stage_assessments", []))
            if isinstance(item, dict) and item.get("step_id") == selected_stage
        ), None)
        if selected_assessment is not None and selected_assessment.get("status") == "blocked":
            reasons.append("selected_stage_blocked_for_continuous_run")
        if observation["observed_continuous_batches"] >= limits["max_continuous_batches"]:
            reasons.append("continuous_batch_limit_reached")
        if latest["work_turns_since_start"] >= limits["max_work_turns"]:
            reasons.append("work_turn_limit_reached")
        if observation.get("task_recovery_blockers") or observation.get("ambiguous_effect_call_ids"):
            reasons.append("unresolved_recovery_or_effect_evidence")
    max_cost = manifest.get("max_cost_usd")
    max_tokens = manifest.get("max_total_tokens")
    if max_cost is not None and cost is None:
        reasons.append("suite_cost_not_reported")
    elif max_cost is not None and cost >= max_cost:
        reasons.append("suite_cost_ceiling_reached")
    if max_tokens is not None and tokens is None:
        reasons.append("suite_tokens_not_reported")
    elif max_tokens is not None and tokens >= max_tokens:
        reasons.append("suite_token_ceiling_reached")
    return {
        "suite": SUITE,
        "case_id": case_id,
        "case_set_digest": selected.digest,
        "decision": "stop" if reasons else "manual_review_required",
        "reasons": reasons,
        "reported_suite_cost_usd": cost,
        "remaining_reported_cost_usd": max_cost - cost if max_cost is not None and cost is not None else None,
        "reported_suite_tokens": tokens,
        "remaining_reported_tokens": max_tokens - tokens if max_tokens is not None and tokens is not None else None,
        "warning": "Advisory only; Runtime does not enforce a hard Goal-wide cost ceiling",
        "usage_accounting": "Known provider-reported usage only; failed requests may lack usage data",
    }


def write_report(prepared_root: Path, *, label: str, deck: GoalPilotDeck | None = None) -> Path:
    if not _LABEL.fullmatch(label):
        raise ValueError("Goal pilot report label must be a safe lowercase name")
    root = prepared_root.expanduser().resolve()
    report = build_report(root, deck=deck)
    directory = root / "reports"
    directory.mkdir(exist_ok=True)
    path = directory / f"{label}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record and summarize a Goal pilot without running a model.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", type=Path, metavar="PREPARED_ROOT")
    mode.add_argument("--report", type=Path, metavar="PREPARED_ROOT")
    mode.add_argument("--budget-review", type=Path, metavar="PREPARED_ROOT")
    parser.add_argument("--case-id")
    parser.add_argument("--phase", choices=sorted(_PHASES))
    parser.add_argument("--observation-label")
    parser.add_argument("--process-instance")
    parser.add_argument("--restart-acceptance-mode", choices=sorted(_RESTART_ACCEPTANCE_MODES))
    parser.add_argument("--assessment-label")
    parser.add_argument("--reported-suite-cost-usd", type=float)
    parser.add_argument("--reported-suite-tokens", type=int)
    parser.add_argument("--failure-attribution", choices=sorted(_FAILURES))
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--unplanned-human-interventions", type=int, default=0)
    parser.add_argument("--status-card-next-action")
    parser.add_argument("--next-step-accurate", choices=["true", "false"])
    parser.add_argument("--label", help="New report label used with --report.")
    parser.add_argument("--deck", type=Path, help="Explicit versioned deck path; defaults to frozen v0.")
    args = parser.parse_args(argv)
    deck = load_deck(args.deck) if args.deck else load_deck()
    if args.report is not None:
        if not args.label:
            parser.error("--report requires --label")
        print(write_report(args.report, label=args.label, deck=deck))
        return 0
    if args.budget_review is not None:
        if not args.case_id:
            parser.error("--budget-review requires --case-id")
        print(json.dumps(review_next_batch(args.budget_review, case_id=args.case_id, deck=deck), ensure_ascii=False, indent=2))
        return 0
    if not args.case_id or not args.phase or not args.observation_label:
        parser.error("--record requires --case-id, --phase and --observation-label")
    print(record_boundary(
        args.record, case_id=args.case_id, phase=args.phase,
        observation_label=args.observation_label, process_instance=args.process_instance,
        restart_acceptance_mode=args.restart_acceptance_mode,
        assessment_label=args.assessment_label,
        reported_suite_cost_usd=args.reported_suite_cost_usd,
        reported_suite_tokens=args.reported_suite_tokens,
        failure_attribution=args.failure_attribution, operator_note=args.operator_note,
        unplanned_human_interventions=args.unplanned_human_interventions,
        status_card_next_action=args.status_card_next_action,
        next_step_accurate=(args.next_step_accurate == "true") if args.next_step_accurate else None,
        deck=deck,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
