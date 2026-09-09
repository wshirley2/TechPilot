"""Private executable long-task holdout loading without publishing case detail."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ReplayCase, ReplayCaseOrigin, ReplayCategory, ReplayTrack
from .holdout import HoldoutFormatError, HoldoutSummary
from .runner import ReplayRunner

LONG_TASK_HOLDOUT_SUITE = "long-task-holdout-v0"
LONG_TASK_HOLDOUT_FORMAT = "long-task-executable-v0"
LONG_TASK_HOLDOUT_SCHEMA_VERSION = 1
LONG_TASK_HOLDOUT_FIELDS = (
    "assertions",
    "category",
    "expected_support",
    "id",
    "initial_state",
    "interruption_point",
    "provider_script",
    "recovery_action",
    "runtime_requirement",
    "title",
    "why_independent",
)


@dataclass(frozen=True)
class LongTaskHoldoutManifest:
    suite: str
    format: str
    schema_version: int
    case_count: int
    case_set_digest: str

    @classmethod
    def from_dict(cls, payload: object) -> LongTaskHoldoutManifest:
        if not isinstance(payload, dict):
            raise HoldoutFormatError("long-task holdout manifest must be a JSON object")
        required = ("suite", "format", "schema_version", "case_count", "case_set_digest")
        missing = [field for field in required if field not in payload]
        if missing:
            raise HoldoutFormatError(
                "long-task holdout manifest is missing required fields: " + ", ".join(missing)
            )
        try:
            manifest = cls(
                suite=str(payload["suite"]),
                format=str(payload["format"]),
                schema_version=int(payload["schema_version"]),
                case_count=int(payload["case_count"]),
                case_set_digest=str(payload["case_set_digest"]),
            )
        except (TypeError, ValueError) as error:
            raise HoldoutFormatError("long-task holdout manifest has invalid field types") from error
        if manifest.suite != LONG_TASK_HOLDOUT_SUITE:
            raise HoldoutFormatError(f"long-task holdout manifest suite must be {LONG_TASK_HOLDOUT_SUITE}")
        if manifest.format != LONG_TASK_HOLDOUT_FORMAT:
            raise HoldoutFormatError(f"long-task holdout manifest format must be {LONG_TASK_HOLDOUT_FORMAT}")
        if manifest.schema_version != LONG_TASK_HOLDOUT_SCHEMA_VERSION:
            raise HoldoutFormatError("unsupported long-task holdout manifest schema_version")
        if manifest.case_count < 1:
            raise HoldoutFormatError("long-task holdout manifest case_count must be positive")
        if not _is_sha256(manifest.case_set_digest):
            raise HoldoutFormatError("long-task holdout manifest case_set_digest must be a sha256 hex value")
        return manifest


@dataclass(frozen=True)
class LongTaskHoldoutCase:
    """A private case that can be converted to a generic Runtime replay input."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: object) -> LongTaskHoldoutCase:
        if not isinstance(payload, dict) or tuple(sorted(payload)) != LONG_TASK_HOLDOUT_FIELDS:
            raise HoldoutFormatError("long-task holdout case has unexpected fields")
        case_id = payload["id"]
        if not isinstance(case_id, str) or not case_id or case_id != case_id.lower():
            raise HoldoutFormatError("long-task holdout case id must be non-empty lowercase text")
        try:
            ReplayCategory(str(payload["category"]))
        except ValueError as error:
            raise HoldoutFormatError("long-task holdout case category is unsupported") from error
        if payload["expected_support"] != "supported":
            raise HoldoutFormatError("long-task executable cases must declare expected_support as supported")
        if payload["runtime_requirement"] != "fixed-provider-v0":
            raise HoldoutFormatError("long-task executable cases require fixed-provider-v0")
        for field in ("title", "why_independent"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise HoldoutFormatError(f"long-task holdout case requires non-empty {field}")
        for field in ("initial_state", "provider_script", "interruption_point", "recovery_action", "assertions"):
            if not isinstance(payload[field], dict):
                raise HoldoutFormatError(f"long-task holdout case {field} must be a JSON object")
        return cls(payload=dict(payload))

    @property
    def id(self) -> str:
        return str(self.payload["id"])

    def to_replay_case(self) -> ReplayCase:
        return ReplayCase(
            id=self.id,
            suite=LONG_TASK_HOLDOUT_SUITE,
            category=ReplayCategory(str(self.payload["category"])),
            scenario="long-task-holdout",
            input={
                "initial_state": dict(self.payload["initial_state"]),
                "provider_script": dict(self.payload["provider_script"]),
                "interruption_point": dict(self.payload["interruption_point"]),
                "recovery_action": dict(self.payload["recovery_action"]),
            },
            expected={"assertions": dict(self.payload["assertions"])},
            track=ReplayTrack.RUNTIME,
            origin=ReplayCaseOrigin.HOLDOUT,
            description=str(self.payload["title"]),
        )


@dataclass(frozen=True)
class LongTaskHoldoutSuite:
    manifest: LongTaskHoldoutManifest
    cases: tuple[LongTaskHoldoutCase, ...]

    @property
    def replay_cases(self) -> tuple[ReplayCase, ...]:
        return tuple(case.to_replay_case() for case in self.cases)


def load_long_task_holdout_suite(root: Path) -> LongTaskHoldoutSuite:
    """Load a private executable deck, retaining its full content only in memory."""

    root = root.expanduser().resolve()
    manifest = _load_manifest(root / "manifest.json")
    payloads = _load_payloads(root / "cases.jsonl")
    cases = tuple(_case_at_line(payload, index) for index, payload in enumerate(payloads, start=1))
    if len(cases) != manifest.case_count:
        raise HoldoutFormatError("long-task holdout case_count does not match cases.jsonl")
    if len({case.id for case in cases}) != len(cases):
        raise HoldoutFormatError("long-task holdout case ids must be unique")
    if _payload_digest(payloads) != manifest.case_set_digest:
        raise HoldoutFormatError("long-task holdout case_set_digest does not match cases.jsonl")
    return LongTaskHoldoutSuite(manifest=manifest, cases=cases)


def long_task_holdout_case_set_metadata(root: Path) -> tuple[int, str]:
    """Return only executable-deck count and digest, never a case payload."""

    root = root.expanduser().resolve()
    payloads = _load_payloads(root / "cases.jsonl")
    cases = tuple(_case_at_line(payload, index) for index, payload in enumerate(payloads, start=1))
    if len({case.id for case in cases}) != len(cases):
        raise HoldoutFormatError("long-task holdout case ids must be unique")
    return len(cases), _payload_digest(payloads)


def run_long_task_holdout(root: Path, runner: ReplayRunner | None = None) -> HoldoutSummary:
    suite = load_long_task_holdout_suite(root)
    report = (runner or ReplayRunner()).run(suite.replay_cases)
    failure_kind_by_case = {
        outcome.case_id: _safe_failure_kind(outcome.failure)
        for outcome in report.outcomes
        if not outcome.passed
    }
    failure_kinds: dict[str, int] = {}
    for kind in failure_kind_by_case.values():
        failure_kinds[kind] = failure_kinds.get(kind, 0) + 1
    observed_by_case = {
        outcome.case_id: dict(outcome.observed)
        for outcome in report.outcomes
        if not outcome.passed and outcome.observed
    }
    return replace(
        HoldoutSummary.from_report(report),
        case_set_digest=suite.manifest.case_set_digest,
        replay_case_set_digest=report.case_set_digest,
        failure_kinds=failure_kinds,
        failure_kind_by_case=failure_kind_by_case,
        observed_by_case=observed_by_case,
    )


def default_long_task_holdout_report_path(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root.expanduser().resolve() / "reports" / f"{LONG_TASK_HOLDOUT_SUITE}-{stamp}.json"


def _load_manifest(path: Path) -> LongTaskHoldoutManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HoldoutFormatError("long-task holdout manifest.json is missing") from error
    except OSError as error:
        raise HoldoutFormatError("long-task holdout manifest.json could not be read") from error
    except json.JSONDecodeError as error:
        raise HoldoutFormatError("long-task holdout manifest.json is not valid JSON") from error
    return LongTaskHoldoutManifest.from_dict(payload)


def _load_payloads(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HoldoutFormatError("could not load long-task holdout cases.jsonl") from error
    payloads: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise HoldoutFormatError(f"invalid JSON in long-task holdout case at line {index}") from error
        if not isinstance(payload, dict):
            raise HoldoutFormatError(f"long-task holdout case at line {index} is not a JSON object")
        payloads.append(payload)
    if not payloads:
        raise HoldoutFormatError("long-task holdout cases.jsonl must contain at least one case")
    return tuple(payloads)


def _case_at_line(payload: dict[str, Any], index: int) -> LongTaskHoldoutCase:
    try:
        return LongTaskHoldoutCase.from_dict(payload)
    except HoldoutFormatError as error:
        raise HoldoutFormatError(f"invalid long-task holdout case at line {index}") from error


def _payload_digest(payloads: tuple[dict[str, Any], ...]) -> str:
    encoded = json.dumps(payloads, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _safe_failure_kind(failure: str | None) -> str:
    """Classify without writing a private assertion, path, or Provider response."""

    if failure is None:
        return "unknown"
    if failure.startswith("long-task-holdout:"):
        return failure.removeprefix("long-task-holdout:")
    if failure.startswith("private long-task") and "mismatch" in failure:
        return "assertion_mismatch"
    if failure == "private long-task recovery did not stop for an ambiguous effect":
        return "recovery_safety_mismatch"
    if failure == "StopIteration":
        return "provider_script_exhausted"
    if failure.startswith(("private long-task", "replay case requires")):
        return "execution_contract_mismatch"
    if failure.startswith("LongTask"):
        return "task_state_rejected"
    return "runtime_execution_error"
