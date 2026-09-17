"""Auditable, small-sample evaluation of real Coding Agent task attempts.

The deterministic Replay deck validates Runtime contracts.  This module keeps a
separate model track: a task card describes the user-visible task and its
independent acceptance criteria, while the Runner records what a real
``TaskRuntime`` did in an isolated workspace.  It deliberately has no default
provider, so importing it cannot start a billable model call.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import socket
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from techpilot.engine.events import CallbackEventSink, RuntimeEvent, RuntimeEventType
from techpilot.engine.permissions import PermissionDecision, PermissionEffect, PermissionRequest
from techpilot.engine.runtime_control import RuntimeLimits
from techpilot.engine.tools import EditFileTool, GlobTool, GrepTool, ReadFileTool, WriteFileTool
from techpilot.runtime import RuntimeBootstrap, RuntimeBootstrapInput
from techpilot.runtime.contracts import RuntimeResultStatus

from .behavior_oracle import BehaviorCheck, evaluate_behavior_check
from .contracts import ModelEvaluationManifest

MODEL_CODING_DEV_V0_SUITE = "model-coding-dev-v0"
MODEL_CODING_TAX_REPAIR_V1_SUITE = "model-coding-tax-repair-v1"
MODEL_CODING_V1_SUITE = "model-coding-v1"
MODEL_CODING_V1_SEED_SUITE = "model-coding-v1-seed"
MODEL_CODING_V1_SEED_QA_V1_SUITE = "model-coding-v1-seed-qa-v1"
MODEL_CODING_V1_FORMAL_CANDIDATE_IDS = (
    "coding-v1-repair-clamp-006", "coding-v1-repair-role-case-008", "coding-v1-repair-normalize-011",
    "coding-v1-cross-turn-tax-018", "coding-v1-regression-normalize-019",
    "coding-v1b1-repair-none-default-025", "coding-v1b1-repair-port-range-027", "coding-v1b1-repair-tag-normalize-028", "coding-v1b1-repair-closed-range-030", "coding-v1b1-cross-turn-delimiter-032", "coding-v1b1-cross-turn-mode-033", "coding-v1b1-context-product-name-034", "coding-v1b1-holdout-cross-turn-prefix-036", "coding-v1b1-regression-serializer-039",
    "coding-v1b2-repair-prefix-045", "coding-v1b2-repair-bool-046", "coding-v1b2-repair-latency-047", "coding-v1b2-repair-membership-048", "coding-v1b2-repair-display-default-049", "coding-v1b2-repair-record-role-051", "coding-v1b2-repair-state-case-052", "coding-v1b2-cross-turn-unit-054", "coding-v1b2-cross-turn-limit-055", "coding-v1b2-context-settings-source-056", "coding-v1b2-dev-scope-057", "coding-v1b2-holdout-cross-turn-title-060", "coding-v1b2-holdout-normalize-062", "coding-v1b2-regression-none-066", "coding-v1b2-regression-fields-067", "coding-v1b2-regression-boundary-069", "coding-v1b2-regression-cross-turn-070",
    "coding-v1b3-repair-priority-076", "coding-v1b3-repair-retries-077", "coding-v1b3-repair-quota-078", "coding-v1b3-repair-names-079", "coding-v1b3-repair-records-080", "coding-v1b3-repair-slug-081", "coding-v1b3-repair-ports-082", "coding-v1b3-repair-messages-083", "coding-v1b3-cross-summaries-084", "coding-v1b3-cross-formats-085", "coding-v1b3-cross-states-086", "coding-v1b3-cross-rows-087", "coding-v1b3-repair-holdout_label-090", "coding-v1b3-repair-holdout_flag-091", "coding-v1b3-repair-holdout_range-092", "coding-v1b3-repair-holdout_status-093", "coding-v1b3-cross-holdout_render-094", "coding-v1b3-cross-holdout_join-095", "coding-v1b3-repair-reg_none-098", "coding-v1b3-repair-reg_boundary-099", "coding-v1b3-cross-reg_cross-100",
)


class ModelTaskKind(str, Enum):
    READ_ONLY = "read-only"
    PATCH = "patch"


class ModelTaskFamily(str, Enum):
    """Behavior under test; this is not a claim about task uniqueness."""

    CODE_READING = "code-reading"
    SCOPED_REPAIR = "scoped-repair"
    MULTI_TURN_CONSTRAINT = "multi-turn-constraint"
    PERMISSION_BOUNDARY = "permission-boundary"
    FAILURE_RECOVERY = "failure-recovery"
    CONTEXT_CONTINUITY = "context-continuity"


class ModelTaskSource(str, Enum):
    """Auditable origin of the task design, never a model-written answer key."""

    MANUAL_CONTRACT = "manual-contract"
    OBSERVED_REGRESSION = "observed-regression"
    FAULT_INJECTION = "fault-injection"
    PUBLIC_PATTERN_ADAPTED = "public-pattern-adapted"
    HUMAN_REVIEWED_LLM_CANDIDATE = "human-reviewed-llm-candidate"


class ModelTaskSplit(str, Enum):
    """Whether a card may inform iteration, must stay frozen, or protects regressions."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"
    REGRESSION = "regression"


class ModelTaskReviewState(str, Enum):
    """Human review state of evaluator-only task-quality evidence."""

    REVIEWED = "reviewed"
    FROZEN = "frozen"


class ModelTaskAcceptanceLevel(str, Enum):
    """Whether a card has only structural checks or a safe behavioral oracle."""

    STATIC_REGRESSION = "static-regression"
    BEHAVIORAL = "behavioral"


class ModelTaskStaticDisposition(str, Enum):
    """The first actionable reason a static-regression card is not behavioral.

    A disposition is a governance label, not an acceptance result. It makes
    the remaining work explicit and prevents static content checks from being
    presented as proof of executable behavior.
    """

    SCOPE_OR_READ_ONLY = "scope-or-read-only"
    EXTEND_BEHAVIOR_ORACLE = "extend-behavior-oracle"
    REWRITE_REQUIRED = "rewrite-required"
    SOURCE_EVIDENCE_REQUIRED = "source-evidence-required"


@dataclass(frozen=True)
class ModelTaskNegativeExample:
    """A deliberately plausible but unacceptable outcome for one task card."""

    id: str
    files: Mapping[str, str] = field(default_factory=dict)
    response: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.id or self.id != self.id.lower():
            raise ValueError("model task negative example id must be non-empty lowercase text")
        if not self.rationale.strip():
            raise ValueError("model task negative example requires a rationale")
        for path in self.files:
            _relative_path(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "files": dict(self.files),
            "response": self.response,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ModelTaskQualityEvidence:
    """Private evaluator evidence that a task's oracle separates right from wrong.

    The Runner never puts this evidence in the task workspace or prompt.  It is
    used only by the local deck-quality gate before a card becomes runnable.
    """

    reference_files: Mapping[str, str] = field(default_factory=dict)
    reference_response: str = ""
    negative_examples: tuple[ModelTaskNegativeExample, ...] = ()
    reviewed_by: str = ""
    review_note: str = ""
    review_state: ModelTaskReviewState = ModelTaskReviewState.REVIEWED

    def __post_init__(self) -> None:
        if not self.reviewed_by.strip() or not self.review_note.strip():
            raise ValueError("model task quality evidence requires reviewer and review note")
        for path in self.reference_files:
            _relative_path(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_files": dict(self.reference_files),
            "reference_response": self.reference_response,
            "negative_examples": [example.to_dict() for example in self.negative_examples],
            "reviewed_by": self.reviewed_by,
            "review_note": self.review_note,
            "review_state": self.review_state.value,
        }


@dataclass(frozen=True)
class ModelTaskQualityIssue:
    card_id: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"card_id": self.card_id, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ModelTaskQualityReport:
    """Auditable result of validating a task deck without calling a model."""

    cards: tuple[ModelTaskCard, ...]
    issues: tuple[ModelTaskQualityIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        def counts(values: Sequence[str]) -> dict[str, int]:
            return {value: values.count(value) for value in sorted(set(values))}

        return {
            "schema_version": 1,
            "case_count": len(self.cards),
            "passed": self.passed,
            "case_set_digest": model_task_set_digest(self.cards),
            "split_counts": counts([card.split.value for card in self.cards]),
            "family_counts": counts([card.family.value for card in self.cards]),
            "source_counts": counts([card.source.value for card in self.cards]),
            "template_counts": counts([card.template_id for card in self.cards]),
            "acceptance_level_counts": counts([card.acceptance_level.value for card in self.cards]),
            "static_disposition_counts": counts([
                card.static_disposition.value if card.static_disposition is not None else "untriaged"
                for card in self.cards
                if card.acceptance_level is ModelTaskAcceptanceLevel.STATIC_REGRESSION
            ]),
            "issues": [issue.to_dict() for issue in self.issues],
        }


class ModelAttemptStatus(str, Enum):
    PASSED = "passed"
    TASK_FAILED = "task_failed"
    RUNTIME_FAILED = "runtime_failed"
    LIMIT_REACHED = "limit_reached"


class ModelEvaluationOutputLockedError(RuntimeError):
    """Raised before a Provider call when another Runner owns an output directory."""


@dataclass(frozen=True)
class ModelTaskCard:
    """One visible Coding task plus rules evaluated outside the Agent workspace."""

    id: str
    suite: str
    kind: ModelTaskKind
    prompts: tuple[str, ...]
    initial_files: Mapping[str, str]
    allowed_paths: tuple[str, ...] = ()
    required_file_contents: Mapping[str, str] = field(default_factory=dict)
    required_response_facts: tuple[str, ...] = ()
    limits: RuntimeLimits = field(default_factory=lambda: RuntimeLimits(max_provider_calls=8, max_tool_rounds=8))
    description: str = ""
    family: ModelTaskFamily = ModelTaskFamily.CODE_READING
    source: ModelTaskSource = ModelTaskSource.MANUAL_CONTRACT
    source_reference: str = "legacy-card"
    template_id: str = "legacy"
    split: ModelTaskSplit = ModelTaskSplit.DEVELOPMENT
    quality: ModelTaskQualityEvidence | None = None
    acceptance_level: ModelTaskAcceptanceLevel = ModelTaskAcceptanceLevel.STATIC_REGRESSION
    behavior_checks: tuple[BehaviorCheck, ...] = ()
    static_disposition: ModelTaskStaticDisposition | None = None

    def __post_init__(self) -> None:
        if not self.id or self.id != self.id.lower():
            raise ValueError("model task id must be non-empty lowercase text")
        if not self.suite:
            raise ValueError("model task suite must not be empty")
        if not self.prompts or any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("model task requires at least one non-empty prompt")
        if not self.initial_files:
            raise ValueError("model task requires initial files")
        if not self.template_id or self.template_id != self.template_id.lower():
            raise ValueError("model task template_id must be non-empty lowercase text")
        if not self.source_reference.strip():
            raise ValueError("model task source_reference must not be blank")
        for path in (*self.initial_files, *self.required_file_contents, *self.allowed_paths):
            _relative_path(path)
        if self.kind is ModelTaskKind.READ_ONLY:
            if self.allowed_paths or self.required_file_contents:
                raise ValueError("read-only model task cannot allow or require file changes")
        elif not self.allowed_paths:
            raise ValueError("patch model task requires allowed_paths")
        if self.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL and not self.behavior_checks:
            raise ValueError("behavioral model task requires at least one behavior check")
        if self.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL and self.required_file_contents:
            raise ValueError("behavioral model task cannot require exact source text")
        if self.acceptance_level is ModelTaskAcceptanceLevel.STATIC_REGRESSION and self.behavior_checks:
            raise ValueError("static regression task cannot contain behavior checks")
        if self.acceptance_level is ModelTaskAcceptanceLevel.BEHAVIORAL and self.static_disposition is not None:
            raise ValueError("behavioral model task cannot contain a static disposition")
        for check in self.behavior_checks:
            _relative_path(check.file_path)

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "kind": self.kind.value,
            "prompts": list(self.prompts),
            "initial_files": dict(self.initial_files),
            "allowed_paths": list(self.allowed_paths),
            "required_file_contents": dict(self.required_file_contents),
            "required_response_facts": list(self.required_response_facts),
            "limits": asdict(self.limits),
            "description": self.description,
            "family": self.family.value,
            "source": self.source.value,
            "source_reference": self.source_reference,
            "template_id": self.template_id,
            "split": self.split.value,
            "acceptance_level": self.acceptance_level.value,
            "behavior_checks": [check.to_dict() for check in self.behavior_checks],
            **({"static_disposition": self.static_disposition.value} if self.static_disposition is not None else {}),
            **({"quality": self.quality.to_dict()} if self.quality is not None else {}),
        }


@dataclass(frozen=True)
class ModelAttemptOutcome:
    task_id: str
    attempt: int
    status: ModelAttemptStatus
    response: str
    checks: Mapping[str, bool]
    failure_categories: tuple[str, ...]
    provider_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost: float | None
    elapsed_seconds: float
    artifact_directory: str
    runtime_status: str | None = None

    @property
    def passed(self) -> bool:
        return self.status is ModelAttemptStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempt": self.attempt,
            "status": self.status.value,
            "passed": self.passed,
            "response": self.response,
            "checks": dict(self.checks),
            "failure_categories": list(self.failure_categories),
            "provider_calls": self.provider_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": self.estimated_cost,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "artifact_directory": self.artifact_directory,
            "runtime_status": self.runtime_status,
        }


@dataclass(frozen=True)
class ModelEvaluationReport:
    manifest: ModelEvaluationManifest
    cards: tuple[ModelTaskCard, ...]
    outcomes: tuple[ModelAttemptOutcome, ...]

    def __post_init__(self) -> None:
        card_ids = {card.id for card in self.cards}
        if not self.outcomes or any(outcome.task_id not in card_ids for outcome in self.outcomes):
            raise ValueError("model evaluation report outcomes must belong to its task cards")
        counts = {card.id: sum(outcome.task_id == card.id for outcome in self.outcomes) for card in self.cards}
        if len(set(counts.values())) != 1:
            raise ValueError("model evaluation report requires the same attempt count for every task")
        for card in self.cards:
            attempts = {outcome.attempt for outcome in self.outcomes if outcome.task_id == card.id}
            if attempts != set(range(1, counts[card.id] + 1)):
                raise ValueError("model evaluation report task attempts must be contiguous from one")

    @property
    def started(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(outcome.passed for outcome in self.outcomes)

    @property
    def attempts_per_task(self) -> int:
        return self.started // len(self.cards)

    def _task_outcomes(self, card: ModelTaskCard) -> tuple[ModelAttemptOutcome, ...]:
        return tuple(sorted((outcome for outcome in self.outcomes if outcome.task_id == card.id), key=lambda outcome: outcome.attempt))

    def _pass_at(self, attempts: int) -> float | None:
        if self.attempts_per_task < attempts:
            return None
        passed_tasks = sum(any(outcome.passed for outcome in self._task_outcomes(card)[:attempts]) for card in self.cards)
        return passed_tasks / len(self.cards)

    def _family_metrics(self) -> dict[str, dict[str, int | float | None]]:
        metrics: dict[str, dict[str, int | float | None]] = {}
        for family in ModelTaskFamily:
            cards = tuple(card for card in self.cards if card.family is family)
            if not cards:
                continue
            outcomes = tuple(outcome for card in cards for outcome in self._task_outcomes(card))
            metrics[family.value] = {
                "case_count": len(cards),
                "attempt_count": len(outcomes),
                "passed_attempt_count": sum(outcome.passed for outcome in outcomes),
                "attempt_pass_rate": sum(outcome.passed for outcome in outcomes) / len(outcomes),
                "pass_at_1": sum(self._task_outcomes(card)[0].passed for card in cards) / len(cards),
                "pass_at_3": (
                    sum(any(outcome.passed for outcome in self._task_outcomes(card)[:3]) for card in cards) / len(cards)
                    if self.attempts_per_task >= 3
                    else None
                ),
            }
        return metrics

    def _distribution(self, values: Sequence[int | float | None]) -> dict[str, float | int | None]:
        observed = sorted(float(value) for value in values if value is not None)
        if not observed:
            return {"observed_count": 0, "p50": None, "p95": None}
        return {
            "observed_count": len(observed),
            "p50": _nearest_rank(observed, 0.50),
            "p95": _nearest_rank(observed, 0.95),
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "attempt_pass_rate": self.passed / self.started,
            "task_pass_at_1": self._pass_at(1),
            "task_pass_at_3": self._pass_at(3),
            "by_family": self._family_metrics(),
            "elapsed_seconds": self._distribution([outcome.elapsed_seconds for outcome in self.outcomes]),
            "provider_calls": self._distribution([outcome.provider_calls for outcome in self.outcomes]),
            "prompt_tokens": self._distribution([outcome.prompt_tokens for outcome in self.outcomes]),
            "completion_tokens": self._distribution([outcome.completion_tokens for outcome in self.outcomes]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "manifest": self.manifest.to_dict(),
            "case_count": len(self.cards),
            "attempt_count": self.started,
            "attempts_per_task": self.attempts_per_task,
            "passed": self.passed,
            "metrics": self.metrics(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


class ModelEvaluationRunner:
    """Run model task cards through the normal Runtime in retained workspaces."""

    def __init__(self, runtime_bootstrap: RuntimeBootstrap) -> None:
        self.runtime_bootstrap = runtime_bootstrap

    def run(
        self,
        cards: Sequence[ModelTaskCard],
        *,
        manifest: ModelEvaluationManifest,
        output_directory: Path,
        attempts_per_task: int = 1,
        limits_override: RuntimeLimits | None = None,
    ) -> ModelEvaluationReport:
        selected = tuple(cards)
        if not selected:
            raise ValueError("model evaluation requires at least one task card")
        if attempts_per_task <= 0:
            raise ValueError("attempts_per_task must be positive")
        if {card.suite for card in selected} != {manifest.suite}:
            raise ValueError("model evaluation cards must match the manifest suite")
        if model_task_set_digest(selected) != manifest.case_set_digest:
            raise ValueError("model evaluation cards do not match the manifest digest")

        output_root = output_directory.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        with _EvaluationOutputLease(output_root):
            outcomes = _restore_or_initialize_progress(
                output_root,
                manifest=manifest,
                cards=selected,
                attempts_per_task=attempts_per_task,
            )
            completed = {(outcome.task_id, outcome.attempt) for outcome in outcomes}
            for card in selected:
                for attempt in range(1, attempts_per_task + 1):
                    if (card.id, attempt) in completed:
                        continue
                    limits = _merged_limits(card.limits, limits_override)
                    outcomes.append(self._run_attempt(card, attempt, output_root, model=manifest.model, limits=limits))
                    completed.add((card.id, attempt))
                    _write_progress(output_root, cards=selected, outcomes=outcomes, attempts_per_task=attempts_per_task)
            report = ModelEvaluationReport(manifest=manifest, cards=selected, outcomes=tuple(outcomes))
            _write_json(output_root / "report.json", report.to_dict())
            return report

    def _run_attempt(
        self,
        card: ModelTaskCard,
        attempt: int,
        output_root: Path,
        *,
        model: str,
        limits: RuntimeLimits,
    ) -> ModelAttemptOutcome:
        attempt_root = output_root / card.id / f"attempt-{attempt}"
        workspace = attempt_root / "workspace"
        session_directory = attempt_root / "sessions"
        if attempt_root.exists():
            saved_outcome = _load_completed_attempt(attempt_root, task_id=card.id, attempt=attempt)
            if saved_outcome is not None:
                return saved_outcome
            _archive_incomplete_attempt(attempt_root)
        workspace.mkdir(parents=True)
        _write_initial_files(workspace, card.initial_files)
        before = _read_files(workspace)
        events: list[RuntimeEvent] = []
        started = time.monotonic()
        response = ""
        runtime_error: Exception | None = None
        runtime = None
        terminal_status: str | None = None
        terminal_reason: str | None = None
        try:
            runtime = self.runtime_bootstrap.build(RuntimeBootstrapInput(
                repository=workspace,
                event_sink=CallbackEventSink(events.append),
                model=model,
                tools=[ReadFileTool(), GlobTool(), GrepTool(), EditFileTool(), WriteFileTool()],
                permission_prompt=_CardPermissionPrompt(card.allowed_paths),
                session_directory=session_directory,
                limits=limits,
                task_id=f"{card.id}-{attempt}",
            ))
            responses: list[str] = []
            for prompt in card.prompts:
                responses.append(runtime.run_turn(prompt))
                result = runtime.last_result
                if result is not None and result.status is not RuntimeResultStatus.SUCCEEDED:
                    terminal_status = result.status.value
                    terminal_reason = result.reason
                    break
            response = responses[-1]
            prompt_tokens = _optional_int(getattr(runtime.agent.llm, "total_prompt_tokens", None))
            completion_tokens = _optional_int(getattr(runtime.agent.llm, "total_completion_tokens", None))
            estimated_cost = _optional_float(getattr(runtime.agent.llm, "estimated_cost", None))
        except Exception as error:  # preserve the failed attempt as an artifact
            runtime_error = error
            prompt_tokens = _optional_int(getattr(runtime.agent.llm, "total_prompt_tokens", None)) if runtime else None
            completion_tokens = _optional_int(getattr(runtime.agent.llm, "total_completion_tokens", None)) if runtime else None
            estimated_cost = _optional_float(getattr(runtime.agent.llm, "estimated_cost", None)) if runtime else None
        elapsed = time.monotonic() - started
        _write_events(attempt_root / "events.jsonl", events)
        after = _read_files(workspace)
        changes = _write_patch(attempt_root / "changes.patch", before, after)
        checks = _score(card, before, after, response)
        provider_calls = sum(event.event_type is RuntimeEventType.PROVIDER_STARTED for event in events)
        failures = [name for name, passed in checks.items() if not passed]
        if runtime_error is not None:
            status = ModelAttemptStatus.RUNTIME_FAILED
            failures.insert(0, f"runtime:{type(runtime_error).__name__}")
        elif terminal_status == RuntimeResultStatus.LIMIT_REACHED.value:
            status = ModelAttemptStatus.LIMIT_REACHED
            failures.insert(0, f"runtime:limit_reached:{terminal_reason or 'unknown'}")
        elif terminal_status is not None:
            status = ModelAttemptStatus.RUNTIME_FAILED
            failures.insert(0, f"runtime:{terminal_status}:{terminal_reason or 'unknown'}")
        elif failures:
            status = ModelAttemptStatus.TASK_FAILED
        else:
            status = ModelAttemptStatus.PASSED
        outcome = ModelAttemptOutcome(
            task_id=card.id,
            attempt=attempt,
            status=status,
            response=response,
            checks=checks,
            failure_categories=tuple(failures),
            provider_calls=provider_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=estimated_cost,
            elapsed_seconds=elapsed,
            artifact_directory=str(attempt_root),
            runtime_status=terminal_status,
        )
        _write_json(attempt_root / "attempt.json", outcome.to_dict() | {"patch_bytes": changes})
        return outcome


class _CardPermissionPrompt:
    """Approve only task-card writes; commands are absent from the task tool set."""

    def __init__(self, allowed_paths: tuple[str, ...]) -> None:
        self.allowed_paths = tuple(_relative_path(path).as_posix() for path in allowed_paths)

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        if request.effect is not PermissionEffect.WRITE:
            return PermissionDecision.deny("model evaluation only pre-approves declared file writes")
        try:
            target = _relative_path(request.scope).as_posix()
        except ValueError:
            return PermissionDecision.deny("model evaluation write target is outside its task card")
        if any(target == allowed or target.startswith(f"{allowed}/") for allowed in self.allowed_paths):
            return PermissionDecision.allow("write path is declared by the model evaluation task card")
        return PermissionDecision.deny("write path is outside the model evaluation task card")


def build_model_coding_dev_v0_cards() -> tuple[ModelTaskCard, ...]:
    """Three development cards that exercise the first end-to-end model harness."""

    return (
        ModelTaskCard(
            id="coding-read-release-status-001",
            suite=MODEL_CODING_DEV_V0_SUITE,
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read RELEASE.md and state the current release status in one sentence. Do not modify files.",),
            initial_files={"RELEASE.md": "# Release\n\nStatus: ready for review\n"},
            required_response_facts=("ready for review",),
            description="A read-only task whose acceptance checks both the fact and zero changes.",
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: read-only state inspection",
            template_id="read-release-status",
        ),
        ModelTaskCard(
            id="coding-patch-tax-001",
            suite=MODEL_CODING_DEV_V0_SUITE,
            kind=ModelTaskKind.PATCH,
            prompts=("Fix the calculation in tax.py. Change only tax.py and keep the function name unchanged.",),
            initial_files={"tax.py": "def total(amount: int) -> int:\n    return amount\n"},
            allowed_paths=("tax.py",),
            required_file_contents={"tax.py": "return amount + 1"},
            description="A single-file repair with a stable public function name.",
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="legacy v0 task; retained for evidence only",
            template_id="single-file-arithmetic-repair",
        ),
        ModelTaskCard(
            id="coding-cross-turn-interface-001",
            suite=MODEL_CODING_DEV_V0_SUITE,
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Inspect formatter.py. The public function signature must remain unchanged.",
                "Now fix the rendered label to use the word 'Pilot'. Change only formatter.py and preserve the public signature.",
            ),
            initial_files={"formatter.py": "def format_label(value: str) -> str:\n    return f'Agent: {value}'\n"},
            allowed_paths=("formatter.py",),
            required_file_contents={
                "formatter.py": "def format_label(value: str) -> str:",
                "formatter.py#rendered-label": "return f'Pilot: {value}'",
            },
            description="A two-turn scoped repair that retains an earlier interface constraint.",
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: preserve prior public interface constraint",
            template_id="cross-turn-interface-preservation",
        ),
    )


def build_model_coding_tax_repair_v1_cards() -> tuple[ModelTaskCard, ...]:
    """A versioned repair card replacing the v0 task with an explicit rule.

    The v0 prompt said only to "fix the calculation" but its evaluator
    expected a hidden ``+ 1`` rule.  This card preserves that evidence in the
    v0 report and makes the revised requirement visible to the model.
    """

    return (
        ModelTaskCard(
            id="coding-patch-tax-002",
            suite=MODEL_CODING_TAX_REPAIR_V1_SUITE,
            kind=ModelTaskKind.PATCH,
            prompts=(
                (
                    "Fix tax.py. The product rule is a fixed tax surcharge of 1 unit: "
                    "total(10) must return 11. Change only tax.py and keep the function name unchanged."
                ),
            ),
            initial_files={"tax.py": "def total(amount: int) -> int:\n    return amount\n"},
            allowed_paths=("tax.py",),
            required_file_contents={"tax.py": "return amount + 1"},
            description="Versioned replacement for v0's under-specified tax repair task.",
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.OBSERVED_REGRESSION,
            source_reference="model-coding-dev-v0/coding-patch-tax-001 hidden-rule diagnosis",
            template_id="single-file-arithmetic-repair",
        ),
    )


def build_model_coding_v1_seed_cards() -> tuple[ModelTaskCard, ...]:
    """First reviewed batch for the future 100-card Coding snapshot deck.

    This is intentionally a seed suite, not the frozen ``model-coding-v1``
    baseline.  Every card has a distinct behavior mechanism and independent
    acceptance assertions; later batches must add coverage rather than merely
    rename files or symbols.
    """

    cards = (
        _seed_card(
            id="coding-v1-read-release-001",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read RELEASE.md and state the release status in one sentence. Do not modify files.",),
            initial_files={"RELEASE.md": "# Release\n\nStatus: pilot-ready\n"},
            required_response_facts=("pilot-ready",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: read-only state inspection",
            template_id="read-release-status",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-read-timeout-002",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Inspect config.py. State the default request timeout in seconds. Do not modify files.",),
            initial_files={"config.py": "DEFAULT_REQUEST_TIMEOUT_SECONDS = 45\n"},
            required_response_facts=("45",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: configuration fact extraction",
            template_id="read-configuration-fact",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-read-cli-flag-003",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read cli.py and state which flag enables safe mode. Do not modify files.",),
            initial_files={"cli.py": "SAFE_MODE_FLAG = '--safe'\n"},
            required_response_facts=("--safe",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: visible CLI behavior inspection",
            template_id="read-configuration-fact",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-read-feature-004",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Inspect features.py. Is bulk export enabled by default? Answer yes or no and do not modify files.",),
            initial_files={"features.py": "BULK_EXPORT_ENABLED = False\n"},
            required_response_facts=("no",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled feature-flag default fixture",
            template_id="read-feature-flag",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-read-validation-owner-005",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read validators.py and name the function that emits 'name is required'. Do not modify files.",),
            initial_files={
                "validators.py": (
                    "def validate_name(value: str) -> str:\n"
                    "    if not value:\n"
                    "        return 'name is required'\n"
                    "    return 'ok'\n"
                ),
            },
            required_response_facts=("validate_name",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: locate behavior owner",
            template_id="read-behavior-owner",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-clamp-006",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix limits.py. clamp(value) must return a value from 0 through 100 inclusive. Change only limits.py.",),
            initial_files={"limits.py": "def clamp(value: int) -> int:\n    return min(value, 100)\n"},
            allowed_paths=("limits.py",),
            required_file_contents={"limits.py": "return max(0, min(value, 100))"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled missing-lower-bound injection",
            template_id="single-file-boundary-repair",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-retry-default-007",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix retry.py. If retry_count is None, retry_count_or_default must return 3; otherwise return retry_count. Change only retry.py.",),
            initial_files={
                "retry.py": (
                    "def retry_count_or_default(retry_count: int | None) -> int:\n"
                    "    return retry_count if retry_count is not None else 0\n"
                ),
            },
            allowed_paths=("retry.py",),
            required_file_contents={"retry.py": "else 3"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: explicit default correction",
            template_id="single-file-default-repair",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-role-case-008",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix auth.py. is_admin must accept case-insensitive role input. Keep its public signature and change only auth.py.",),
            initial_files={"auth.py": "def is_admin(role: str) -> bool:\n    return role == 'admin'\n"},
            allowed_paths=("auth.py",),
            required_file_contents={"auth.py": "return role.casefold() == 'admin'"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled case-normalization injection",
            template_id="single-file-normalization-repair",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-boundary-009",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix quota.py. reached(value, limit) must return true when value is equal to or greater than limit. Change only quota.py.",),
            initial_files={"quota.py": "def reached(value: int, limit: int) -> bool:\n    return value > limit\n"},
            allowed_paths=("quota.py",),
            required_file_contents={"quota.py": "return value >= limit"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: inclusive boundary correction",
            template_id="single-file-boundary-repair",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-serializer-010",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix serializer.py. as_payload must return both id and name. Change only serializer.py and keep the function signature unchanged.",),
            initial_files={
                "serializer.py": (
                    "def as_payload(user: dict[str, str]) -> dict[str, str]:\n"
                    "    return {'id': user['id']}\n"
                ),
            },
            allowed_paths=("serializer.py",),
            required_file_contents={"serializer.py": "'name': user['name']"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: incomplete serialization repair",
            template_id="single-file-required-field-repair",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-normalize-011",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix normalize.py. normalize must strip surrounding whitespace and convert to lowercase. Change only normalize.py; keep its public signature unchanged.",),
            initial_files={"normalize.py": "def normalize(value: str) -> str:\n    return value\n"},
            allowed_paths=("normalize.py",),
            required_file_contents={
                "normalize.py": "def normalize(value: str) -> str:",
                "normalize.py#normalized-return": "return value.strip().lower()",
            },
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled missing-normalization injection",
            template_id="single-file-normalization-repair",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-lookup-012",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix lookup.py. find_label must return None when a key is absent, rather than the string 'unknown'. Change only lookup.py.",),
            initial_files={
                "lookup.py": (
                    "def find_label(labels: dict[str, str], key: str) -> str | None:\n"
                    "    return labels.get(key, 'unknown')\n"
                ),
            },
            allowed_paths=("lookup.py",),
            required_file_contents={"lookup.py": "return labels.get(key)"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: sentinel-value correction",
            template_id="single-file-default-repair",
            split=ModelTaskSplit.DEVELOPMENT,
        ),
        _seed_card(
            id="coding-v1-repair-message-013",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix messages.py. The user-visible title must be 'Connection failed'. Change only messages.py.",),
            initial_files={"messages.py": "CONNECTION_FAILURE_TITLE = 'Connection error'\n"},
            allowed_paths=("messages.py",),
            required_file_contents={"messages.py": "CONNECTION_FAILURE_TITLE = 'Connection failed'"},
            family=ModelTaskFamily.PERMISSION_BOUNDARY,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: narrow write scope with visible output change",
            template_id="scoped-constant-repair",
            split=ModelTaskSplit.HOLDOUT,
        ),
        _seed_card(
            id="coding-v1-repair-multifile-014",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix the greeting in formatter.py to use the public product name from constants.py. Change only formatter.py and constants.py.",),
            initial_files={
                "constants.py": "PRODUCT_NAME = 'TechPilot'\n",
                "formatter.py": "def greeting(name: str) -> str:\n    return f'Hello {name} from Pilot'\n",
            },
            allowed_paths=("formatter.py", "constants.py"),
            required_file_contents={
                "formatter.py": "from constants import PRODUCT_NAME",
                "formatter.py#product-name-usage": "return f'Hello {name} from {PRODUCT_NAME}'",
            },
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: cross-module constant consistency",
            template_id="two-file-constant-consistency",
            split=ModelTaskSplit.HOLDOUT,
        ),
        _seed_card(
            id="coding-v1-cross-turn-mode-015",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Inspect policy.py. The public signature of resolve_mode must remain unchanged.",
                "Now change the fallback returned by resolve_mode from 'fast' to 'safe'. Change only policy.py and preserve the public signature.",
            ),
            initial_files={"policy.py": "def resolve_mode(value: str | None) -> str:\n    return value or 'fast'\n"},
            allowed_paths=("policy.py",),
            required_file_contents={
                "policy.py": "def resolve_mode(value: str | None) -> str:",
                "policy.py#safe-fallback": "return value or 'safe'",
            },
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: preserve cross-turn public interface constraint",
            template_id="cross-turn-interface-preservation",
            split=ModelTaskSplit.HOLDOUT,
        ),
        _seed_card(
            id="coding-v1-cross-turn-template-016",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Inspect template.py. Keep render's parameters unchanged.",
                "Change the default prefix displayed by render to 'Report'. Change only template.py and keep the parameters unchanged.",
            ),
            initial_files={"template.py": "def render(value: str, prefix: str = 'Draft') -> str:\n    return f'{prefix}: {value}'\n"},
            allowed_paths=("template.py",),
            required_file_contents={
                "template.py": "def render(value: str, prefix: str = 'Report') -> str:",
                "template.py#render-body": "return f'{prefix}: {value}'",
            },
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled default-label change with retained interface",
            template_id="cross-turn-default-preservation",
            split=ModelTaskSplit.HOLDOUT,
        ),
        _seed_card(
            id="coding-v1-cross-turn-summary-017",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Read summary.py. Do not change the name or parameters of build_summary.",
                "Now make build_summary use '; ' between items rather than ', '. Change only summary.py and retain the earlier interface constraint.",
            ),
            initial_files={"summary.py": "def build_summary(items: list[str]) -> str:\n    return ', '.join(items)\n"},
            allowed_paths=("summary.py",),
            required_file_contents={
                "summary.py": "def build_summary(items: list[str]) -> str:",
                "summary.py#separator": "return '; '.join(items)",
            },
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: preserve API while changing output behavior",
            template_id="cross-turn-interface-preservation",
            split=ModelTaskSplit.HOLDOUT,
        ),
        _seed_card(
            id="coding-v1-cross-turn-tax-018",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Inspect tax.py. The function name total must remain unchanged.",
                "Apply the explicit fixed surcharge rule: total(10) must return 11. Change only tax.py and retain the function name.",
            ),
            initial_files={"tax.py": "def total(amount: int) -> int:\n    return amount\n"},
            allowed_paths=("tax.py",),
            required_file_contents={
                "tax.py": "def total(amount: int) -> int:",
                "tax.py#explicit-surcharge": "return amount + 1",
            },
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.OBSERVED_REGRESSION,
            source_reference="model-coding-dev-v0/coding-patch-tax-001 under-specified-rule diagnosis",
            template_id="cross-turn-explicit-rule-repair",
            split=ModelTaskSplit.REGRESSION,
        ),
        _seed_card(
            id="coding-v1-regression-normalize-019",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix name.py. canonical_name must strip whitespace and use lowercase. Change only name.py and keep its public signature unchanged.",),
            initial_files={"name.py": "def canonical_name(value: str) -> str:\n    return value\n"},
            allowed_paths=("name.py",),
            required_file_contents={
                "name.py": "def canonical_name(value: str) -> str:",
                "name.py#canonical-return": "return value.strip().lower()",
            },
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled normalization regression guard",
            template_id="single-file-normalization-repair",
            split=ModelTaskSplit.REGRESSION,
        ),
        _seed_card(
            id="coding-v1-regression-scope-020",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix status.py. current_status must return 'ready'. Change only status.py; do not modify README.md.",),
            initial_files={
                "status.py": "def current_status() -> str:\n    return 'draft'\n",
                "README.md": "Status is documented elsewhere.\n",
            },
            allowed_paths=("status.py",),
            required_file_contents={"status.py": "return 'ready'"},
            family=ModelTaskFamily.PERMISSION_BOUNDARY,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: scoped write must leave adjacent documentation untouched",
            template_id="scoped-constant-repair",
            split=ModelTaskSplit.REGRESSION,
        ),
    )
    return cards


def build_model_coding_v1_seed_qa_v1_cards() -> tuple[ModelTaskCard, ...]:
    """Versioned QA-approved successor to the already executed seed deck.

    The original ``model-coding-v1-seed`` remains untouched so its historical
    report retains an exact case-set digest.  This successor fixes the
    serializer oracle exposed by the quality gate and adds private QA evidence.
    """

    behavior_checks = {
        "coding-v1-repair-clamp-006": (
            BehaviorCheck("below", "limits.py", "clamp", (-5,), 0),
            BehaviorCheck("middle", "limits.py", "clamp", (50,), 50),
            BehaviorCheck("above", "limits.py", "clamp", (150,), 100),
        ),
        "coding-v1-repair-retry-default-007": (
            BehaviorCheck("none-default", "retry.py", "retry_count_or_default", (None,), 3),
            BehaviorCheck("explicit-value", "retry.py", "retry_count_or_default", (7,), 7),
        ),
        "coding-v1-repair-role-case-008": (
            BehaviorCheck("upper-admin", "auth.py", "is_admin", ("ADMIN",), True),
            BehaviorCheck("non-admin", "auth.py", "is_admin", ("editor",), False),
        ),
        "coding-v1-repair-boundary-009": (
            BehaviorCheck("equal-limit", "quota.py", "reached", (5, 5), True),
            BehaviorCheck("below-limit", "quota.py", "reached", (4, 5), False),
            BehaviorCheck("above-limit", "quota.py", "reached", (6, 5), True),
        ),
        "coding-v1-repair-normalize-011": (
            BehaviorCheck("trim-and-lower", "normalize.py", "normalize", ("  Ready ",), "ready"),
            BehaviorCheck("already-normalized", "normalize.py", "normalize", ("ok",), "ok"),
        ),
        "coding-v1-repair-lookup-012": (
            BehaviorCheck("present-key", "lookup.py", "find_label", ({"a": "alpha"}, "a"), "alpha"),
            BehaviorCheck("missing-key", "lookup.py", "find_label", ({"a": "alpha"}, "b"), None),
        ),
    }
    upgraded: list[ModelTaskCard] = []
    for card in build_model_coding_v1_seed_cards():
        if card.id == "coding-v1-repair-serializer-010":
            card = replace(
                card,
                required_file_contents={
                    "serializer.py#id": "'id': user['id']",
                    "serializer.py#name": "'name': user['name']",
                },
            )
        checks = behavior_checks.get(card.id, ())
        upgraded.append(replace(
            card,
            suite=MODEL_CODING_V1_SEED_QA_V1_SUITE,
            acceptance_level=(ModelTaskAcceptanceLevel.BEHAVIORAL if checks else ModelTaskAcceptanceLevel.STATIC_REGRESSION),
            behavior_checks=checks,
            required_file_contents={} if checks else card.required_file_contents,
        ))
    return _with_seed_quality_evidence(tuple(upgraded))


def _with_seed_quality_evidence(cards: Sequence[ModelTaskCard]) -> tuple[ModelTaskCard, ...]:
    """Attach maintainer-reviewed golden and negative outcomes to the seed deck."""

    evidence = {
        "coding-v1-read-release-001": _seed_qa(reference_response="pilot-ready", negative_response="not ready"),
        "coding-v1-read-timeout-002": _seed_qa(reference_response="45 seconds", negative_response="30 seconds"),
        "coding-v1-read-cli-flag-003": _seed_qa(reference_response="--safe", negative_response="--debug"),
        "coding-v1-read-feature-004": _seed_qa(reference_response="no", negative_response="yes"),
        "coding-v1-read-validation-owner-005": _seed_qa(reference_response="validate_name", negative_response="validate_email"),
        "coding-v1-repair-clamp-006": _seed_qa(
            reference_files={"limits.py": "def clamp(value: int) -> int:\n    return max(0, min(value, 100))\n"},
            negative_files={"limits.py": "def clamp(value: int) -> int:\n    return min(max(value, 1), 100)\n"},
        ),
        "coding-v1-repair-retry-default-007": _seed_qa(
            reference_files={"retry.py": "def retry_count_or_default(retry_count: int | None) -> int:\n    return retry_count if retry_count is not None else 3\n"},
            negative_files={"retry.py": "def retry_count_or_default(retry_count: int | None) -> int:\n    return retry_count if retry_count is not None else 1\n"},
        ),
        "coding-v1-repair-role-case-008": _seed_qa(
            reference_files={"auth.py": "def is_admin(role: str) -> bool:\n    return role.casefold() == 'admin'\n"},
            negative_files={"auth.py": "def is_admin(role: str) -> bool:\n    return role.lower() == 'root'\n"},
        ),
        "coding-v1-repair-boundary-009": _seed_qa(
            reference_files={"quota.py": "def reached(value: int, limit: int) -> bool:\n    return value >= limit\n"},
            negative_files={"quota.py": "def reached(value: int, limit: int) -> bool:\n    return value == limit\n"},
        ),
        "coding-v1-repair-serializer-010": _seed_qa(
            reference_files={"serializer.py": "def as_payload(user: dict[str, str]) -> dict[str, str]:\n    return {'id': user['id'], 'name': user['name']}\n"},
            negative_files={"serializer.py": "def as_payload(user: dict[str, str]) -> dict[str, str]:\n    return {'name': user['name']}\n"},
        ),
        "coding-v1-repair-normalize-011": _seed_qa(
            reference_files={"normalize.py": "def normalize(value: str) -> str:\n    return value.strip().lower()\n"},
            negative_files={"normalize.py": "def normalize(value: str) -> str:\n    return value.strip()\n"},
        ),
        "coding-v1-repair-lookup-012": _seed_qa(
            reference_files={"lookup.py": "def find_label(labels: dict[str, str], key: str) -> str | None:\n    return labels.get(key)\n"},
            negative_files={"lookup.py": "def find_label(labels: dict[str, str], key: str) -> str | None:\n    return labels.get(key, 'missing')\n"},
        ),
        "coding-v1-repair-message-013": _seed_qa(
            reference_files={"messages.py": "CONNECTION_FAILURE_TITLE = 'Connection failed'\n"},
            negative_files={"messages.py": "CONNECTION_FAILURE_TITLE = 'Connection unavailable'\n"},
            frozen=True,
        ),
        "coding-v1-repair-multifile-014": _seed_qa(
            reference_files={"formatter.py": "from constants import PRODUCT_NAME\n\ndef greeting(name: str) -> str:\n    return f'Hello {name} from {PRODUCT_NAME}'\n"},
            negative_files={"formatter.py": "def greeting(name: str) -> str:\n    return f'Hi {name} from Pilot'\n"},
            frozen=True,
        ),
        "coding-v1-cross-turn-mode-015": _seed_qa(
            reference_files={"policy.py": "def resolve_mode(value: str | None) -> str:\n    return value or 'safe'\n"},
            negative_files={"policy.py": "def resolve_mode(value: str | None) -> str:\n    return value or 'secure'\n"},
            frozen=True,
        ),
        "coding-v1-cross-turn-template-016": _seed_qa(
            reference_files={"template.py": "def render(value: str, prefix: str = 'Report') -> str:\n    return f'{prefix}: {value}'\n"},
            negative_files={"template.py": "def render(value: str, prefix: str = 'Summary') -> str:\n    return f'{prefix}: {value}'\n"},
            frozen=True,
        ),
        "coding-v1-cross-turn-summary-017": _seed_qa(
            reference_files={"summary.py": "def build_summary(items: list[str]) -> str:\n    return '; '.join(items)\n"},
            negative_files={"summary.py": "def build_summary(items: list[str]) -> str:\n    return ' | '.join(items)\n"},
            frozen=True,
        ),
        "coding-v1-cross-turn-tax-018": _seed_qa(
            reference_files={"tax.py": "def total(amount: int) -> int:\n    return amount + 1\n"},
            negative_files={"tax.py": "def total(amount: int) -> int:\n    return amount + 2\n"},
        ),
        "coding-v1-regression-normalize-019": _seed_qa(
            reference_files={"name.py": "def canonical_name(value: str) -> str:\n    return value.strip().lower()\n"},
            negative_files={"name.py": "def canonical_name(value: str) -> str:\n    return value.lower()\n"},
        ),
        "coding-v1-regression-scope-020": _seed_qa(
            reference_files={"status.py": "def current_status() -> str:\n    return 'ready'\n"},
            negative_files={"status.py": "def current_status() -> str:\n    return 'pending'\n"},
        ),
    }
    selected = tuple(cards)
    if set(evidence) != {card.id for card in selected}:
        raise ValueError("seed quality evidence must cover exactly the seed cards")
    enriched = tuple(replace(card, quality=evidence[card.id]) for card in selected)
    return require_valid_model_task_deck(enriched)


def _seed_qa(
    *,
    reference_files: Mapping[str, str] | None = None,
    reference_response: str = "",
    negative_files: Mapping[str, str] | None = None,
    negative_response: str = "",
    frozen: bool = False,
) -> ModelTaskQualityEvidence:
    return ModelTaskQualityEvidence(
        reference_files=reference_files or {},
        reference_response=reference_response,
        negative_examples=(
            ModelTaskNegativeExample(
                id="semantic-wrong-outcome",
                files=negative_files or {},
                response=negative_response,
                rationale="A plausible answer or in-scope repair that misses the stated acceptance rule.",
            ),
        ),
        reviewed_by="techpilot-maintainer",
        review_note="Reference and semantic negative checked against the independent local oracle.",
        review_state=ModelTaskReviewState.FROZEN if frozen else ModelTaskReviewState.REVIEWED,
    )


def select_model_task_cards(
    cards: Sequence[ModelTaskCard],
    split: ModelTaskSplit,
) -> tuple[ModelTaskCard, ...]:
    """Return a declared split without changing card order or hidden task facts."""

    return tuple(card for card in cards if card.split is split)


def _seed_card(
    *,
    id: str,
    kind: ModelTaskKind,
    prompts: tuple[str, ...],
    initial_files: Mapping[str, str],
    allowed_paths: tuple[str, ...] = (),
    required_file_contents: Mapping[str, str] | None = None,
    required_response_facts: tuple[str, ...] = (),
    family: ModelTaskFamily,
    source: ModelTaskSource,
    source_reference: str,
    template_id: str,
    split: ModelTaskSplit,
) -> ModelTaskCard:
    return ModelTaskCard(
        id=id,
        suite=MODEL_CODING_V1_SEED_SUITE,
        kind=kind,
        prompts=prompts,
        initial_files=initial_files,
        allowed_paths=allowed_paths,
        required_file_contents=required_file_contents or {},
        required_response_facts=required_response_facts,
        family=family,
        source=source,
        source_reference=source_reference,
        template_id=template_id,
        split=split,
    )


def validate_model_task_deck(cards: Sequence[ModelTaskCard]) -> ModelTaskQualityReport:
    """Check that evaluator-only evidence makes each card reproducible and discriminating.

    This gate is intentionally deterministic: it does not call a provider and
    it does not treat an LLM's own proposed solution as proof of correctness.
    """

    selected = tuple(cards)
    issues: list[ModelTaskQualityIssue] = []

    def issue(card: ModelTaskCard, code: str, message: str) -> None:
        issues.append(ModelTaskQualityIssue(card.id, code, message))

    ids = [card.id for card in selected]
    for card_id in sorted({card_id for card_id in ids if ids.count(card_id) > 1}):
        issues.append(ModelTaskQualityIssue(card_id, "duplicate-id", "task ids must be unique within a deck"))
    fingerprints: dict[str, str] = {}
    for card in selected:
        body = {
            "kind": card.kind.value,
            "prompts": list(card.prompts),
            "initial_files": dict(card.initial_files),
            "allowed_paths": list(card.allowed_paths),
            "required_file_contents": dict(card.required_file_contents),
            "required_response_facts": list(card.required_response_facts),
        }
        fingerprint = _digest(body)
        if fingerprint in fingerprints:
            issue(card, "duplicate-task-body", f"same visible task body as {fingerprints[fingerprint]}")
        else:
            fingerprints[fingerprint] = card.id

        evidence = card.quality
        if evidence is None:
            issue(card, "missing-quality-evidence", "card needs reference and negative acceptance evidence")
            continue
        if card.split is ModelTaskSplit.HOLDOUT and evidence.review_state is not ModelTaskReviewState.FROZEN:
            issue(card, "holdout-not-frozen", "holdout cards must be reviewed and frozen before execution")
        if len({example.id for example in evidence.negative_examples}) != len(evidence.negative_examples):
            issue(card, "duplicate-negative-id", "negative example ids must be unique per card")
        if not evidence.negative_examples:
            issue(card, "missing-negative-example", "card needs at least one plausible unacceptable outcome")

        before = {path: content.encode("utf-8") for path, content in card.initial_files.items()}
        reference = _overlay_files(before, evidence.reference_files)
        reference_checks = _score(card, before, reference, evidence.reference_response)
        if not all(reference_checks.values()):
            issue(card, "reference-rejected", "reference outcome does not satisfy every acceptance check")
        baseline_checks = _score(card, before, before, "")
        if all(baseline_checks.values()):
            issue(card, "original-accepted", "unmodified workspace with no answer must not pass")

        reference_changes = _changed_paths(before, reference)
        if card.kind is ModelTaskKind.READ_ONLY:
            if evidence.reference_files:
                issue(card, "read-only-reference-writes", "read-only reference must not change workspace files")
        elif not reference_changes or not all(_is_allowed_path(path, set(card.allowed_paths)) for path in reference_changes):
            issue(card, "reference-scope", "patch reference must make an in-scope workspace change")

        for example in evidence.negative_examples:
            after = _overlay_files(before, example.files)
            checks = _score(card, before, after, example.response)
            if all(checks.values()):
                issue(card, f"negative-accepted:{example.id}", "negative example satisfies the current oracle")
            changes = _changed_paths(before, after)
            if card.kind is ModelTaskKind.READ_ONLY and example.files:
                issue(card, f"negative-read-only-writes:{example.id}", "read-only negative must test an answer, not a file change")
            if card.kind is ModelTaskKind.PATCH:
                if not changes:
                    issue(card, f"negative-no-change:{example.id}", "patch negative must model a plausible changed but wrong repair")
                elif not all(_is_allowed_path(path, set(card.allowed_paths)) for path in changes):
                    issue(card, f"negative-out-of-scope:{example.id}", "semantic negative must stay inside declared write scope")

        if card.kind is ModelTaskKind.PATCH:
            scope_probe = dict(reference)
            scope_probe["__qa_forbidden__/probe.txt"] = b"not allowed\n"
            if all(_score(card, before, scope_probe, evidence.reference_response).values()):
                issue(card, "scope-probe-accepted", "oracle did not reject an out-of-scope write")

    return ModelTaskQualityReport(cards=selected, issues=tuple(issues))


def require_valid_model_task_deck(cards: Sequence[ModelTaskCard]) -> tuple[ModelTaskCard, ...]:
    """Return cards only if the deterministic QA gate accepts the entire deck."""

    selected = tuple(cards)
    report = validate_model_task_deck(selected)
    if not report.passed:
        summary = "; ".join(f"{item.card_id}:{item.code}" for item in report.issues)
        raise ValueError(f"model task deck quality gate failed: {summary}")
    return selected


_MODEL_CODING_V1_STATIC_DISPOSITIONS = {
    **dict.fromkeys((
        "coding-v1-read-timeout-002",
        "coding-v1-read-validation-owner-005",
        "coding-v1-repair-serializer-010",
        "coding-v1-repair-multifile-014",
        "coding-v1-cross-turn-summary-017",
        "coding-v1b1-read-product-label-022",
        "coding-v1b1-repair-url-join-026",
        "coding-v1b1-repair-record-field-029",
        "coding-v1b1-repair-stable-labels-031",
        "coding-v1b1-holdout-error-code-035",
        "coding-v1b2-read-service-default-043",
        "coding-v1b2-repair-label-list-050",
        "coding-v1b2-holdout-error-059",
        "coding-v1b2-holdout-record-064",
    ), ModelTaskStaticDisposition.SOURCE_EVIDENCE_REQUIRED),
    **dict.fromkeys((
        "coding-v1-read-release-001",
        "coding-v1-read-cli-flag-003",
        "coding-v1-read-feature-004",
        "coding-v1-repair-message-013",
        "coding-v1-regression-scope-020",
        "coding-v1b1-read-timeout-precedence-021",
        "coding-v1b1-read-error-owner-023",
        "coding-v1b1-read-required-field-024",
        "coding-v1b1-holdout-scope-037",
        "coding-v1b1-holdout-read-source-038",
        "coding-v1b1-regression-write-scope-040",
        "coding-v1b2-read-retry-delay-041",
        "coding-v1b2-read-redaction-field-042",
        "coding-v1b2-read-validator-044",
        "coding-v1b2-holdout-read-policy-058",
        "coding-v1b2-holdout-scope-061",
        "coding-v1b2-holdout-read-format-063",
        "coding-v1b2-regression-scope-068",
        "coding-v1b3-read-version-071",
        "coding-v1b3-read-cache-072",
        "coding-v1b3-read-roles-073",
        "coding-v1b3-read-paths-074",
        "coding-v1b3-read-modes-075",
        "coding-v1b3-read-holdout_region-088",
        "coding-v1b3-read-holdout_limit-089",
        "coding-v1b3-repair-reg_scope_one-096",
        "coding-v1b3-repair-reg_scope_two-097",
    ), ModelTaskStaticDisposition.SCOPE_OR_READ_ONLY),
    **dict.fromkeys((
        "coding-v1-repair-clamp-006",
        "coding-v1-cross-turn-mode-015",
        "coding-v1-cross-turn-template-016",
        "coding-v1b1-repair-port-range-027",
        "coding-v1b1-repair-closed-range-030",
        "coding-v1b1-cross-turn-mode-033",
        "coding-v1b1-holdout-cross-turn-prefix-036",
        "coding-v1b2-repair-latency-047",
        "coding-v1b2-repair-first-present-053",
        "coding-v1b2-holdout-cross-turn-title-060",
        "coding-v1b2-holdout-normalize-062",
        "coding-v1b2-holdout-context-065",
        "coding-v1b2-regression-boundary-069",
        "coding-v1b3-cross-formats-085",
        "coding-v1b3-cross-states-086",
        "coding-v1b3-repair-holdout_label-090",
        "coding-v1b3-repair-holdout_flag-091",
        "coding-v1b3-repair-holdout_status-093",
        "coding-v1b3-cross-holdout_render-094",
        "coding-v1b3-repair-reg_none-098",
        "coding-v1b3-repair-quota-078",
        "coding-v1b3-repair-names-079",
        "coding-v1b3-repair-records-080",
        "coding-v1b3-repair-ports-082",
        "coding-v1b3-cross-summaries-084",
        "coding-v1b3-cross-rows-087",
        "coding-v1b3-repair-holdout_range-092",
        "coding-v1b3-cross-holdout_join-095",
        "coding-v1b3-cross-reg_cross-100",
    ), ModelTaskStaticDisposition.EXTEND_BEHAVIOR_ORACLE),
}


def apply_model_coding_v1_static_dispositions(cards: Sequence[ModelTaskCard]) -> tuple[ModelTaskCard, ...]:
    """Attach the reviewed, card-level disposition to the v1 working decks.

    Raw batches intentionally remain unchanged so their historical QA evidence
    can still be reconstructed. This overlay is only for the versioned working
    aggregates and raises if a static card was added without a review decision.
    """

    selected = tuple(cards)
    static_ids = {
        card.id
        for card in selected
        if card.acceptance_level is ModelTaskAcceptanceLevel.STATIC_REGRESSION
    }
    missing = sorted(static_ids - set(_MODEL_CODING_V1_STATIC_DISPOSITIONS))
    if missing:
        raise ValueError(f"static model task disposition missing for: {', '.join(missing)}")
    return tuple(
        replace(card, static_disposition=_MODEL_CODING_V1_STATIC_DISPOSITIONS[card.id])
        if card.acceptance_level is ModelTaskAcceptanceLevel.STATIC_REGRESSION
        else card
        for card in selected
    )


def _overlay_files(before: Mapping[str, bytes], files: Mapping[str, str]) -> dict[str, bytes]:
    after = dict(before)
    after.update({path: content.encode("utf-8") for path, content in files.items()})
    return after


def _changed_paths(before: Mapping[str, bytes], after: Mapping[str, bytes]) -> set[str]:
    return {path for path in set(before) | set(after) if before.get(path) != after.get(path)}


def model_task_set_digest(cards: Sequence[ModelTaskCard]) -> str:
    return _digest([card.to_dict() for card in cards])


def _score(
    card: ModelTaskCard,
    before: Mapping[str, bytes],
    after: Mapping[str, bytes],
    response: str,
) -> dict[str, bool]:
    changed = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    allowed = set(card.allowed_paths)
    checks: dict[str, bool] = {
        "changed_paths_allowed": all(_is_allowed_path(path, allowed) for path in changed),
    }
    if card.kind is ModelTaskKind.READ_ONLY:
        checks["read_only_workspace_unchanged"] = not changed
    for raw_path, required in card.required_file_contents.items():
        path = raw_path.split("#", 1)[0]
        content = after.get(path, b"").decode("utf-8", errors="replace")
        checks[f"required_content:{raw_path}"] = required in content
    response_lower = response.casefold()
    for fact in card.required_response_facts:
        checks[f"response_fact:{fact}"] = fact.casefold() in response_lower
    for check in card.behavior_checks:
        source = after.get(check.file_path, b"").decode("utf-8", errors="replace")
        workspace_sources = {
            path: content.decode("utf-8", errors="replace")
            for path, content in after.items()
        }
        checks[f"behavior:{check.id}"] = evaluate_behavior_check(source, check, workspace_sources=workspace_sources)
    return checks


def _is_allowed_path(path: str, allowed: set[str]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in allowed)


def _write_initial_files(root: Path, files: Mapping[str, str]) -> None:
    for relative, content in files.items():
        target = root / _relative_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _read_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_patch(path: Path, before: Mapping[str, bytes], after: Mapping[str, bytes]) -> int:
    parts: list[str] = []
    for relative in sorted(set(before) | set(after)):
        old = before.get(relative, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        new = after.get(relative, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        parts.extend(difflib.unified_diff(old, new, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    rendered = "".join(parts)
    path.write_text(rendered, encoding="utf-8")
    return len(rendered.encode("utf-8"))


def _write_events(path: Path, events: Sequence[RuntimeEvent]) -> None:
    path.write_text("".join(json.dumps(event.to_dict(), ensure_ascii=False) + "\n" for event in events), encoding="utf-8")


class _EvaluationOutputLease:
    """An output-directory lease that prevents concurrent billable evaluation runs.

    Progress files make a run resumable after interruption, but are not a
    coordination primitive: two processes could otherwise restore the same
    checkpoint and both start a Provider call.  The lock is acquired before
    progress is read and held until the report is durable or the caller exits.
    """

    _FILENAME = ".model-evaluation.lock"

    def __init__(self, root: Path) -> None:
        self.path = root / self._FILENAME
        self.owner = {
            "schema_version": 1,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "lease_id": uuid.uuid4().hex,
        }
        self._acquired = False

    def __enter__(self):
        while True:
            try:
                descriptor = json.dumps(self.owner, ensure_ascii=False).encode("utf-8")
                handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                try:
                    os.write(handle, descriptor)
                finally:
                    os.close(handle)
                self._acquired = True
                return self
            except FileExistsError:
                owner = _read_evaluation_output_lease(self.path)
                if owner is None or _evaluation_output_lease_is_active(owner):
                    raise ModelEvaluationOutputLockedError(
                        f"model evaluation output is already active: {self.path}; "
                        "wait for its owner to exit before resuming"
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    continue

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if not self._acquired:
            return
        owner = _read_evaluation_output_lease(self.path)
        if owner is not None and owner.get("lease_id") == self.owner["lease_id"]:
            self.path.unlink(missing_ok=True)
        self._acquired = False


def _read_evaluation_output_lease(path: Path) -> Mapping[str, Any] | None:
    """Return a valid lease descriptor, or ``None`` for malformed lock data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("host"), str)
        or not isinstance(payload.get("pid"), int)
        or not isinstance(payload.get("lease_id"), str)
    ):
        return None
    return payload


def _evaluation_output_lease_is_active(owner: Mapping[str, Any]) -> bool:
    """Fail closed unless a same-host owner is definitely no longer running."""

    if owner["host"] != socket.gethostname():
        return True
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _load_completed_attempt(attempt_root: Path, *, task_id: str, attempt: int) -> ModelAttemptOutcome | None:
    """Reuse an attempt persisted before its parent progress checkpoint."""

    attempt_path = attempt_root / "attempt.json"
    if not attempt_path.exists():
        return None
    try:
        outcome = _outcome_from_dict(json.loads(attempt_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"model evaluation artifact cannot be recovered: {attempt_path}: {error}") from error
    if outcome.task_id != task_id or outcome.attempt != attempt:
        raise ValueError(f"model evaluation artifact belongs to a different task attempt: {attempt_root}")
    return outcome


def _archive_incomplete_attempt(attempt_root: Path) -> Path:
    """Preserve an interrupted attempt while freeing its exact retry location."""

    index = 1
    while True:
        archive_root = attempt_root.with_name(f"{attempt_root.name}-interrupted-{index}")
        if not archive_root.exists():
            attempt_root.replace(archive_root)
            return archive_root
        index += 1


def _write_progress(
    root: Path,
    *,
    cards: Sequence[ModelTaskCard],
    outcomes: Sequence[ModelAttemptOutcome],
    attempts_per_task: int,
) -> None:
    """Persist completed attempts even if the runner stops before final summary."""

    _write_json(root / "progress.json", {
        "schema_version": 1,
        "planned_case_count": len(cards),
        "planned_case_set_digest": model_task_set_digest(cards),
        "planned_attempt_count": len(cards) * attempts_per_task,
        "completed_attempt_count": len(outcomes),
        "outcomes": [outcome.to_dict() for outcome in outcomes],
    })


def _restore_or_initialize_progress(
    root: Path,
    *,
    manifest: ModelEvaluationManifest,
    cards: Sequence[ModelTaskCard],
    attempts_per_task: int,
) -> list[ModelAttemptOutcome]:
    """Resume only an exactly matching interrupted run; never mix task decks."""

    manifest_path = root / "manifest.json"
    progress_path = root / "progress.json"
    if not manifest_path.exists() and not progress_path.exists():
        _write_json(manifest_path, manifest.to_dict())
        outcomes: list[ModelAttemptOutcome] = []
        _write_progress(root, cards=cards, outcomes=outcomes, attempts_per_task=attempts_per_task)
        return outcomes
    if not manifest_path.exists() or not progress_path.exists():
        raise ValueError("model evaluation output has incomplete resume metadata")
    try:
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"model evaluation output cannot resume: {error}") from error
    if saved_manifest != manifest.to_dict():
        raise ValueError("model evaluation output manifest does not match this requested run")
    expected_digest = model_task_set_digest(cards)
    if (
        progress.get("planned_case_count") != len(cards)
        or progress.get("planned_case_set_digest") != expected_digest
        or progress.get("planned_attempt_count") != len(cards) * attempts_per_task
    ):
        raise ValueError("model evaluation progress does not match this requested task deck or attempt count")
    raw_outcomes = progress.get("outcomes")
    if not isinstance(raw_outcomes, list):
        raise TypeError("model evaluation progress outcomes must be a list")
    outcomes = [_outcome_from_dict(value) for value in raw_outcomes]
    if progress.get("completed_attempt_count") != len(outcomes):
        raise ValueError("model evaluation progress completed count does not match its outcomes")
    expected_pairs = {(card.id, attempt) for card in cards for attempt in range(1, attempts_per_task + 1)}
    actual_pairs = {(outcome.task_id, outcome.attempt) for outcome in outcomes}
    if len(actual_pairs) != len(outcomes) or not actual_pairs <= expected_pairs:
        raise ValueError("model evaluation progress contains duplicate or unexpected task attempts")
    return outcomes


def _outcome_from_dict(value: object) -> ModelAttemptOutcome:
    if not isinstance(value, dict):
        raise TypeError("model evaluation progress outcome must be an object")
    try:
        status = ModelAttemptStatus(str(value["status"]))
        task_id = value["task_id"]
        attempt = value["attempt"]
        response = value["response"]
        checks = value["checks"]
        failure_categories = value["failure_categories"]
        provider_calls = value["provider_calls"]
        elapsed_seconds = value["elapsed_seconds"]
        artifact_directory = value["artifact_directory"]
    except KeyError as error:
        raise ValueError(f"model evaluation progress outcome is missing {error.args[0]}") from error
    if (
        not isinstance(task_id, str)
        or not isinstance(attempt, int)
        or not isinstance(response, str)
        or not isinstance(checks, dict)
        or not all(isinstance(name, str) and isinstance(passed, bool) for name, passed in checks.items())
        or not isinstance(failure_categories, list)
        or not all(isinstance(category, str) for category in failure_categories)
        or not isinstance(provider_calls, int)
        or not isinstance(elapsed_seconds, int | float)
        or not isinstance(artifact_directory, str)
    ):
        raise ValueError("model evaluation progress outcome has invalid field types")
    return ModelAttemptOutcome(
        task_id=task_id,
        attempt=attempt,
        status=status,
        response=response,
        checks=checks,
        failure_categories=tuple(failure_categories),
        provider_calls=provider_calls,
        prompt_tokens=_optional_int(value.get("prompt_tokens")),
        completion_tokens=_optional_int(value.get("completion_tokens")),
        estimated_cost=_optional_float(value.get("estimated_cost")),
        elapsed_seconds=float(elapsed_seconds),
        artifact_directory=artifact_directory,
        runtime_status=value.get("runtime_status") if isinstance(value.get("runtime_status"), str) else None,
    )


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"model evaluation path must stay inside the task workspace: {value}")
    return path


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Return the deterministic nearest-rank percentile for an ordered sample."""

    if not values:
        raise ValueError("nearest-rank percentile requires values")
    index = max(0, min(len(values) - 1, int((len(values) * percentile) - 1e-12)))
    return values[index]


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _merged_limits(card_limits: RuntimeLimits, override: RuntimeLimits | None) -> RuntimeLimits:
    if override is None:
        return card_limits
    return replace(
        card_limits,
        **{
            name: value
            for name in RuntimeLimits.__dataclass_fields__
            if (value := getattr(override, name)) is not None
        },
    )
