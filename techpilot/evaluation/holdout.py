"""Private holdout loading with structural validation and redacted reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ReplayCase, ReplayCaseOrigin, ReplayCategory, ReplayReport, ReplayTrack, case_set_digest
from .runner import ReplayRunner

HOLDOUT_SCHEMA_VERSION = 1
HOLDOUT_SUITE = "holdout-v0"
LONG_TASK_HOLDOUT_DESIGN_FORMAT = "long-task-holdout-design-v0"
LONG_TASK_HOLDOUT_DESIGN_FIELDS = (
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


class HoldoutFormatError(ValueError):
    """A private suite cannot be run safely using the public Replay contract."""


@dataclass(frozen=True)
class HoldoutManifest:
    """The non-sensitive integrity metadata required beside a private case deck."""

    suite: str
    schema_version: int
    case_count: int
    case_set_digest: str

    @classmethod
    def from_dict(cls, payload: object) -> HoldoutManifest:
        if not isinstance(payload, dict):
            raise HoldoutFormatError("holdout manifest must be a JSON object")
        required = ("suite", "schema_version", "case_count", "case_set_digest")
        missing = [field for field in required if field not in payload]
        if missing:
            raise HoldoutFormatError(f"holdout manifest is missing required fields: {', '.join(missing)}")
        try:
            manifest = cls(
                suite=str(payload["suite"]),
                schema_version=int(payload["schema_version"]),
                case_count=int(payload["case_count"]),
                case_set_digest=str(payload["case_set_digest"]),
            )
        except (TypeError, ValueError) as error:
            raise HoldoutFormatError("holdout manifest has invalid field types") from error
        if manifest.suite != HOLDOUT_SUITE:
            raise HoldoutFormatError(f"holdout manifest suite must be {HOLDOUT_SUITE}")
        if manifest.schema_version != HOLDOUT_SCHEMA_VERSION:
            raise HoldoutFormatError("unsupported holdout manifest schema_version")
        if manifest.case_count < 1:
            raise HoldoutFormatError("holdout manifest case_count must be positive")
        if len(manifest.case_set_digest) != 64 or any(char not in "0123456789abcdef" for char in manifest.case_set_digest):
            raise HoldoutFormatError("holdout manifest case_set_digest must be a sha256 hex value")
        return manifest


@dataclass(frozen=True)
class HoldoutSuite:
    """A validated private deck, retained only in process memory while running."""

    manifest: HoldoutManifest
    cases: tuple[ReplayCase, ...]


@dataclass(frozen=True)
class HoldoutSummary:
    """The only persisted or printed holdout result; it intentionally omits case detail."""

    suite: str
    git_commit: str
    git_dirty: bool
    case_set_digest: str
    passed: int
    total: int
    categories: dict[str, dict[str, int]]
    failed_case_ids: tuple[str, ...]
    replay_case_set_digest: str | None = None
    failure_kinds: dict[str, int] = field(default_factory=dict)
    failure_kind_by_case: dict[str, str] = field(default_factory=dict)
    observed_by_case: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: ReplayReport) -> HoldoutSummary:
        return cls(
            suite=report.suite,
            git_commit=report.git_commit,
            git_dirty=report.git_dirty,
            case_set_digest=report.case_set_digest,
            passed=report.passed,
            total=report.total,
            categories=report.category_results,
            failed_case_ids=tuple(outcome.case_id for outcome in report.outcomes if not outcome.passed),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": HOLDOUT_SCHEMA_VERSION,
            "kind": "holdout-summary-v0",
            "suite": self.suite,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "case_set_digest": self.case_set_digest,
            "passed": self.passed,
            "total": self.total,
            "categories": self.categories,
            "failed_case_ids": list(self.failed_case_ids),
        }
        if self.replay_case_set_digest is not None:
            payload["replay_case_set_digest"] = self.replay_case_set_digest
        if self.failure_kinds:
            payload["failure_kinds"] = self.failure_kinds
        if self.failure_kind_by_case:
            payload["failure_kind_by_case"] = self.failure_kind_by_case
        if self.observed_by_case:
            payload["observed_by_case"] = self.observed_by_case
        return payload


@dataclass(frozen=True)
class HoldoutCaseSchema:
    """Field names and count only; no private case values leave the local process."""

    case_count: int
    fields: tuple[str, ...]


@dataclass(frozen=True)
class LongTaskHoldoutDesignMetadata:
    """Non-sensitive identity facts for an independently-authored design deck.

    This is deliberately not a Replay report.  A design deck describes intended
    scenarios, but becomes executable only after it adopts the separate,
    versioned long-task execution contract.
    """

    case_count: int
    case_set_digest: str
    fields: tuple[str, ...]


def load_holdout_suite(root: Path) -> HoldoutSuite:
    """Load one external holdout deck without copying it into the repository."""

    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    cases_path = root / "cases.jsonl"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise HoldoutFormatError("holdout manifest.json is missing") from error
    except OSError as error:
        raise HoldoutFormatError("holdout manifest.json could not be read") from error
    try:
        manifest_payload = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise HoldoutFormatError("holdout manifest.json is not valid JSON") from error
    manifest = HoldoutManifest.from_dict(manifest_payload)
    cases = _load_cases(cases_path)
    if len(cases) != manifest.case_count:
        raise HoldoutFormatError("holdout case_count does not match cases.jsonl")
    digest = case_set_digest(cases)
    if digest != manifest.case_set_digest:
        raise HoldoutFormatError("holdout case_set_digest does not match cases.jsonl")
    return HoldoutSuite(manifest=manifest, cases=cases)


def run_holdout(root: Path, runner: ReplayRunner | None = None) -> HoldoutSummary:
    """Run a validated holdout deck and return a result without private details."""

    suite = load_holdout_suite(root)
    report = (runner or ReplayRunner()).run(suite.cases)
    return HoldoutSummary.from_report(report)


def holdout_case_set_metadata(root: Path) -> tuple[int, str]:
    """Return private deck count and digest without printing or persisting case detail."""

    root = root.expanduser().resolve()
    cases = _load_cases(root / "cases.jsonl")
    if any(case.suite != HOLDOUT_SUITE or case.origin is not ReplayCaseOrigin.HOLDOUT for case in cases):
        raise HoldoutFormatError("holdout cases.jsonl must use holdout-v0 with origin holdout")
    return len(cases), case_set_digest(cases)


def inspect_holdout_case_schema(root: Path) -> HoldoutCaseSchema:
    """Read only JSON object keys so an external deck can be adapted without revealing cases."""

    root = root.expanduser().resolve()
    try:
        lines = (root / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HoldoutFormatError("could not load holdout cases.jsonl") from error
    fields: set[str] = set()
    count = 0
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise HoldoutFormatError(f"invalid JSON in holdout case at line {index}") from error
        if not isinstance(payload, dict):
            raise HoldoutFormatError(f"holdout case at line {index} is not a JSON object")
        fields.update(str(key) for key in payload)
        count += 1
    if count == 0:
        raise HoldoutFormatError("holdout cases.jsonl must contain at least one case")
    return HoldoutCaseSchema(case_count=count, fields=tuple(sorted(fields)))


def inspect_long_task_holdout_design(root: Path) -> LongTaskHoldoutDesignMetadata:
    """Validate a private long-task design deck without exposing case values.

    Unlike ``holdout-v0``, this format is not routed to the public Replay
    runner.  Keeping that distinction explicit prevents a narrative design
    card from being silently treated as a passing Runtime test.
    """

    payloads = _load_jsonl_payloads(root / "cases.jsonl")
    ids: set[str] = set()
    for index, payload in enumerate(payloads, start=1):
        fields = tuple(sorted(payload))
        if fields != LONG_TASK_HOLDOUT_DESIGN_FIELDS:
            raise HoldoutFormatError(f"invalid long-task holdout design case at line {index}")
        case_id = payload["id"]
        if not isinstance(case_id, str) or not case_id or case_id != case_id.lower() or case_id in ids:
            raise HoldoutFormatError(f"invalid long-task holdout design case at line {index}")
        ids.add(case_id)
        if any(not _has_value(payload[field]) for field in LONG_TASK_HOLDOUT_DESIGN_FIELDS if field != "id"):
            raise HoldoutFormatError(f"invalid long-task holdout design case at line {index}")
    return LongTaskHoldoutDesignMetadata(
        case_count=len(payloads),
        case_set_digest=_payload_set_digest(payloads),
        fields=LONG_TASK_HOLDOUT_DESIGN_FIELDS,
    )


def default_holdout_report_path(root: Path) -> Path:
    """Create a non-overwriting summary filename under the private report directory."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root.expanduser().resolve() / "reports" / f"{HOLDOUT_SUITE}-{stamp}.json"


def write_holdout_summary(summary: HoldoutSummary, output: Path) -> Path:
    """Persist only redacted summary fields, never case inputs or observed outputs."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def _load_cases(path: Path) -> tuple[ReplayCase, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HoldoutFormatError("could not load holdout cases.jsonl") from error
    cases: list[ReplayCase] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            case = _case_from_dict(payload)
        except (json.JSONDecodeError, HoldoutFormatError, TypeError, ValueError) as error:
            raise HoldoutFormatError(f"invalid holdout case at line {index}") from error
        cases.append(case)
    if not cases:
        raise HoldoutFormatError("holdout cases.jsonl must contain at least one case")
    if len({case.id for case in cases}) != len(cases):
        raise HoldoutFormatError("holdout case ids must be unique")
    return tuple(cases)


def _load_jsonl_payloads(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HoldoutFormatError("could not load holdout cases.jsonl") from error
    payloads: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise HoldoutFormatError(f"invalid JSON in holdout case at line {index}") from error
        if not isinstance(payload, dict):
            raise HoldoutFormatError(f"holdout case at line {index} is not a JSON object")
        payloads.append(payload)
    if not payloads:
        raise HoldoutFormatError("holdout cases.jsonl must contain at least one case")
    return tuple(payloads)


def _case_from_dict(payload: object) -> ReplayCase:
    if not isinstance(payload, dict):
        raise HoldoutFormatError("holdout case must be a JSON object")
    try:
        case = ReplayCase(
            id=str(payload["id"]),
            suite=str(payload["suite"]),
            category=ReplayCategory(str(payload["category"])),
            scenario=str(payload["scenario"]),
            input=_mapping(payload["input"]),
            expected=_mapping(payload["expected"]),
            track=ReplayTrack(str(payload.get("track", ReplayTrack.RUNTIME.value))),
            origin=ReplayCaseOrigin(str(payload.get("origin", ReplayCaseOrigin.HOLDOUT.value))),
            description=str(payload.get("description", "")),
        )
    except (KeyError, ValueError) as error:
        raise HoldoutFormatError("holdout case does not match the Runtime Replay contract") from error
    if case.suite != HOLDOUT_SUITE:
        raise HoldoutFormatError(f"holdout case suite must be {HOLDOUT_SUITE}")
    if case.origin is not ReplayCaseOrigin.HOLDOUT:
        raise HoldoutFormatError("holdout case origin must be holdout")
    return case


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HoldoutFormatError("holdout case input and expected values must be JSON objects")
    return value


def _has_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def _payload_set_digest(payloads: tuple[dict[str, Any], ...]) -> str:
    import hashlib

    encoded = json.dumps(payloads, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
