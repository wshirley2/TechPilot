"""Run deterministic Runtime Replay or explicitly authorized model-evaluation suites."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from techpilot.config.user import resolve_runtime_config
from techpilot.engine.runtime_control import RuntimeLimits
from techpilot.runtime import RuntimeBootstrap

from .cases import (
    CORE_V0_SUITE,
    LONG_TASK_OBSERVABILITY_V0_SUITE,
    LONG_TASK_RUNTIME_SMOKE_SUITE,
    LONG_TASK_RUNTIME_V0_SUITE,
    ROLE_RUNTIME_VALIDATION_SUITE,
    RUNNER_VALIDATION_SUITE,
    build_core_v0_cases,
    build_long_task_observability_v0_cases,
    build_long_task_runtime_smoke_cases,
    build_long_task_runtime_v0_cases,
    build_role_runtime_validation_cases,
    build_runner_validation_cases,
)
from .contracts import BaselineReference
from .holdout import (
    HoldoutFormatError,
    default_holdout_report_path,
    holdout_case_set_metadata,
    inspect_holdout_case_schema,
    inspect_long_task_holdout_design,
    run_holdout,
    write_holdout_summary,
)
from .long_task_holdout import (
    default_long_task_holdout_report_path,
    long_task_holdout_case_set_metadata,
    run_long_task_holdout,
)
from .model_task_batch_01 import (
    MODEL_CODING_V1_BATCH_01_SUITE,
    MODEL_CODING_V1_WORKING_40_SUITE,
    build_model_coding_v1_batch_01_cards,
    build_model_coding_v1_working_40_cards,
)
from .model_task_batch_02 import (
    MODEL_CODING_V1_BATCH_02_SUITE,
    MODEL_CODING_V1_WORKING_70_SUITE,
    build_model_coding_v1_batch_02_cards,
    build_model_coding_v1_working_70_cards,
)
from .model_task_batch_03 import MODEL_CODING_V1_WORKING_100_SUITE, build_model_coding_v1_working_100_cards
from .model_task_v2_formal import (
    MODEL_CODING_V2_FORMAL_100_SUITE,
    build_model_coding_v2_formal_100_cards,
)
from .model_tasks import (
    MODEL_CODING_DEV_V0_SUITE,
    MODEL_CODING_TAX_REPAIR_V1_SUITE,
    MODEL_CODING_V1_FORMAL_CANDIDATE_IDS,
    MODEL_CODING_V1_SEED_QA_V1_SUITE,
    MODEL_CODING_V1_SEED_SUITE,
    ModelEvaluationManifest,
    ModelEvaluationRunner,
    ModelTaskCard,
    ModelTaskSplit,
    build_model_coding_dev_v0_cards,
    build_model_coding_tax_repair_v1_cards,
    build_model_coding_v1_seed_cards,
    build_model_coding_v1_seed_qa_v1_cards,
    model_task_set_digest,
    select_model_task_cards,
)
from .runner import ReplayRunner


def _per_attempt_token_limit(
    max_total_tokens: int | None,
    *,
    card_count: int,
    attempts_per_task: int,
) -> int | None:
    """Divide a run-level token ceiling across isolated Runtime attempts.

    A card may contain several user prompts but runs inside one Runtime session, so
    applying a per-prompt share as that session's total budget artificially
    penalizes cross-turn cards.
    """

    if max_total_tokens is None:
        return None
    return max_total_tokens // (card_count * attempts_per_task)


def _select_model_task_ids(cards: Sequence[ModelTaskCard], task_ids: Sequence[str]) -> tuple[ModelTaskCard, ...]:
    """Keep an explicitly named subset in deck order for a bounded smoke run."""

    if not task_ids:
        return tuple(cards)
    requested = set(task_ids)
    available = {card.id for card in cards}
    unknown = requested - available
    if unknown:
        raise ValueError(f"--model-task-id is not available in the selected task deck: {', '.join(sorted(unknown))}")
    return tuple(card for card in cards if card.id in requested)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TechPilot Runtime Replay or explicitly authorized model-evaluation cases.")
    parser.add_argument(
        "--suite",
        default=CORE_V0_SUITE,
        choices=[CORE_V0_SUITE, RUNNER_VALIDATION_SUITE, ROLE_RUNTIME_VALIDATION_SUITE, LONG_TASK_RUNTIME_SMOKE_SUITE, LONG_TASK_RUNTIME_V0_SUITE, LONG_TASK_OBSERVABILITY_V0_SUITE, MODEL_CODING_DEV_V0_SUITE, MODEL_CODING_TAX_REPAIR_V1_SUITE, MODEL_CODING_V1_BATCH_01_SUITE, MODEL_CODING_V1_BATCH_02_SUITE, MODEL_CODING_V1_SEED_SUITE, MODEL_CODING_V1_SEED_QA_V1_SUITE, MODEL_CODING_V1_WORKING_40_SUITE, MODEL_CODING_V1_WORKING_70_SUITE, MODEL_CODING_V1_WORKING_100_SUITE, MODEL_CODING_V2_FORMAL_100_SUITE],
    )
    parser.add_argument("--output", type=Path, help="Write the structured result manifest to this JSON path.")
    parser.add_argument("--run-model", action="store_true", help="Explicitly permit a billable real-model evaluation run.")
    parser.add_argument("--model-output", type=Path, help="Directory for retained model-evaluation workspaces and reports.")
    parser.add_argument("--model", help="Model used only with --run-model; otherwise user configuration supplies it.")
    parser.add_argument("--model-base-url", help="Provider base URL used only with --run-model.")
    parser.add_argument("--model-api-key", help="Provider API key used only with --run-model; prefer user configuration.")
    parser.add_argument("--attempts-per-task", type=int, default=1, help="Repeated attempts per model task card (default: 1).")
    parser.add_argument(
        "--model-task-split",
        choices=[split.value for split in ModelTaskSplit],
        help="Required for a split-aware model suite; selects development, holdout, or regression cards.",
    )
    parser.add_argument("--max-cost-usd", type=float, help="Optional USD limit for a real-model evaluation run.")
    parser.add_argument("--max-total-tokens", type=int, help="Optional total Token budget allocated across planned task attempts.")
    parser.add_argument("--formal-candidate-prebaseline", action="store_true", help="Run only non-Holdout formal candidates from working-100.")
    parser.add_argument("--model-task-id", action="append", default=[], help="Run only this task ID; repeat to select a small explicit subset.")
    parser.add_argument(
        "--baseline-v0",
        action="store_true",
        help="Require a clean, fully passing core-v0 run before writing its formal baseline.",
    )
    parser.add_argument("--summary", action="store_true", help="Print only the suite result and case-set digest.")
    parser.add_argument(
        "--compare-baseline",
        type=Path,
        help="Compare only against a baseline report with the identical suite, track, and case-set digest.",
    )
    parser.add_argument(
        "--holdout-root",
        type=Path,
        help="Run an external holdout-v0 directory and print/write only a redacted summary.",
    )
    parser.add_argument(
        "--holdout-case-set-metadata",
        type=Path,
        help="Print only the count and digest needed to complete an external holdout manifest.",
    )
    parser.add_argument(
        "--holdout-case-schema",
        type=Path,
        help="Print only JSONL case count and field names, never private case values.",
    )
    parser.add_argument(
        "--long-task-holdout-design-metadata",
        type=Path,
        help="Validate a private long-task design deck and print only its count, field names, and digest.",
    )
    parser.add_argument(
        "--long-task-holdout-root",
        type=Path,
        help="Run a private long-task-holdout-v0 directory and print/write only a redacted summary.",
    )
    parser.add_argument(
        "--long-task-holdout-case-set-metadata",
        type=Path,
        help="Print only count and digest for a private long-task-holdout-v0 manifest.",
    )
    args = parser.parse_args(argv)
    model_suites = {MODEL_CODING_DEV_V0_SUITE, MODEL_CODING_TAX_REPAIR_V1_SUITE, MODEL_CODING_V1_BATCH_01_SUITE, MODEL_CODING_V1_BATCH_02_SUITE, MODEL_CODING_V1_SEED_SUITE, MODEL_CODING_V1_SEED_QA_V1_SUITE, MODEL_CODING_V1_WORKING_40_SUITE, MODEL_CODING_V1_WORKING_70_SUITE, MODEL_CODING_V1_WORKING_100_SUITE, MODEL_CODING_V2_FORMAL_100_SUITE}
    if args.suite in model_suites:
        if not args.run_model:
            parser.error(f"--suite {args.suite} requires --run-model")
        if args.model_output is None:
            parser.error("--run-model requires --model-output")
        if args.output is not None:
            parser.error("--run-model writes its report inside --model-output; do not pass --output")
        if args.max_cost_usd is None and args.max_total_tokens is None:
            parser.error("--run-model requires --max-cost-usd or --max-total-tokens")
        if args.max_cost_usd is not None and args.max_cost_usd <= 0:
            parser.error("--max-cost-usd must be positive")
        if args.max_total_tokens is not None and args.max_total_tokens <= 0:
            parser.error("--max-total-tokens must be positive")
        if args.attempts_per_task <= 0:
            parser.error("--attempts-per-task must be positive")
        split_aware_suites = {MODEL_CODING_V1_BATCH_01_SUITE, MODEL_CODING_V1_BATCH_02_SUITE, MODEL_CODING_V1_SEED_SUITE, MODEL_CODING_V1_SEED_QA_V1_SUITE, MODEL_CODING_V1_WORKING_40_SUITE, MODEL_CODING_V1_WORKING_70_SUITE, MODEL_CODING_V1_WORKING_100_SUITE, MODEL_CODING_V2_FORMAL_100_SUITE}
        if args.formal_candidate_prebaseline and (args.suite != MODEL_CODING_V1_WORKING_100_SUITE or args.model_task_split is not None):
            parser.error("--formal-candidate-prebaseline requires working-100 without --model-task-split")
        if args.suite in split_aware_suites and args.model_task_split is None and not args.formal_candidate_prebaseline:
            parser.error(f"--suite {args.suite} requires --model-task-split")
        if args.suite not in split_aware_suites and args.model_task_split is not None:
            parser.error("--model-task-split is supported only by split-aware model suites")
        if any((args.baseline_v0, args.compare_baseline, args.holdout_root, args.long_task_holdout_root)):
            parser.error("--run-model cannot be combined with baseline or holdout options")
        config = resolve_runtime_config(
            model=args.model,
            base_url=args.model_base_url,
            api_key=args.model_api_key,
        )
        cards = (
            tuple(card for card in build_model_coding_v1_working_100_cards() if card.id in MODEL_CODING_V1_FORMAL_CANDIDATE_IDS and card.split is not ModelTaskSplit.HOLDOUT)
            if args.formal_candidate_prebaseline
            else
            build_model_coding_dev_v0_cards()
            if args.suite == MODEL_CODING_DEV_V0_SUITE
            else build_model_coding_tax_repair_v1_cards()
            if args.suite == MODEL_CODING_TAX_REPAIR_V1_SUITE
            else select_model_task_cards(
                build_model_coding_v1_batch_01_cards()
                if args.suite == MODEL_CODING_V1_BATCH_01_SUITE
                else build_model_coding_v1_batch_02_cards()
                if args.suite == MODEL_CODING_V1_BATCH_02_SUITE
                else build_model_coding_v1_seed_cards()
                if args.suite == MODEL_CODING_V1_SEED_SUITE
                else build_model_coding_v1_seed_qa_v1_cards()
                if args.suite == MODEL_CODING_V1_SEED_QA_V1_SUITE
                else build_model_coding_v1_working_40_cards()
                if args.suite == MODEL_CODING_V1_WORKING_40_SUITE
                else build_model_coding_v1_working_100_cards()
                if args.suite == MODEL_CODING_V1_WORKING_100_SUITE
                else build_model_coding_v2_formal_100_cards()
                if args.suite == MODEL_CODING_V2_FORMAL_100_SUITE
                else build_model_coding_v1_working_70_cards(),
                ModelTaskSplit(args.model_task_split),
            )
        )
        cards = _select_model_task_ids(cards, args.model_task_id)
        per_attempt_token_limit = _per_attempt_token_limit(
            args.max_total_tokens,
            card_count=len(cards),
            attempts_per_task=args.attempts_per_task,
        )
        if per_attempt_token_limit == 0:
            parser.error("--max-total-tokens is too small for the selected task and attempt count")
        manifest = ModelEvaluationManifest(
            provider=config.provider,
            model=config.model,
            parameters={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "max_cost_usd": args.max_cost_usd,
                "max_total_tokens": args.max_total_tokens,
                "per_attempt_token_limit": per_attempt_token_limit,
                "attempts_per_task": args.attempts_per_task,
                "task_split": args.model_task_split,
            },
            suite=args.suite,
            case_set_digest=model_task_set_digest(cards),
        )
        try:
            report = ModelEvaluationRunner(RuntimeBootstrap()).run(
                cards,
                manifest=manifest,
                output_directory=args.model_output,
                attempts_per_task=args.attempts_per_task,
                limits_override=RuntimeLimits(
                    max_cost_usd=args.max_cost_usd,
                    max_total_tokens=per_attempt_token_limit,
                ),
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(f"{manifest.suite}: {report.passed}/{report.started} passed; tasks={len(cards)}; attempts_per_task={report.attempts_per_task}")
        print(f"report: {args.model_output / 'report.json'}")
        return 0 if report.passed == report.started else 1
    if args.long_task_holdout_design_metadata is not None:
        if any((args.holdout_root, args.holdout_case_set_metadata, args.holdout_case_schema, args.long_task_holdout_root, args.long_task_holdout_case_set_metadata, args.baseline_v0, args.compare_baseline)):
            parser.error("--long-task-holdout-design-metadata cannot be combined with another holdout or baseline option")
        try:
            metadata = inspect_long_task_holdout_design(args.long_task_holdout_design_metadata)
        except HoldoutFormatError as error:
            parser.error(str(error))
        print(f"case_count: {metadata.case_count}")
        print("case_fields: " + ", ".join(metadata.fields))
        print(f"case_set_digest: {metadata.case_set_digest}")
        print("status: design_only; convert to long-task-holdout-v0 before execution")
        return 0
    if args.long_task_holdout_case_set_metadata is not None:
        if any((args.holdout_root, args.holdout_case_set_metadata, args.holdout_case_schema, args.long_task_holdout_root, args.baseline_v0, args.compare_baseline)):
            parser.error("--long-task-holdout-case-set-metadata cannot be combined with another holdout or baseline option")
        try:
            count, digest = long_task_holdout_case_set_metadata(args.long_task_holdout_case_set_metadata)
        except HoldoutFormatError as error:
            parser.error(str(error))
        print(f"case_count: {count}")
        print(f"case_set_digest: {digest}")
        return 0
    if args.long_task_holdout_root is not None:
        if args.baseline_v0 or args.compare_baseline is not None or args.holdout_root is not None:
            parser.error("--long-task-holdout-root cannot be combined with another holdout or baseline option")
        try:
            summary = run_long_task_holdout(args.long_task_holdout_root)
        except HoldoutFormatError as error:
            parser.error(str(error))
        output = args.output or default_long_task_holdout_report_path(args.long_task_holdout_root)
        write_holdout_summary(summary, output)
        print(f"{summary.suite}: {summary.passed}/{summary.total} passed; source_case_set_digest={summary.case_set_digest}")
        print(f"replay_case_set_digest: {summary.replay_case_set_digest}")
        print(f"categories: {json.dumps(summary.categories, ensure_ascii=False, sort_keys=True)}")
        print("failed_case_ids: " + (", ".join(summary.failed_case_ids) if summary.failed_case_ids else "none"))
        if summary.failure_kinds:
            print(f"failure_kinds: {json.dumps(summary.failure_kinds, ensure_ascii=False, sort_keys=True)}")
            print(f"failure_kind_by_case: {json.dumps(summary.failure_kind_by_case, ensure_ascii=False, sort_keys=True)}")
        if summary.observed_by_case:
            print(f"observed_by_case: {json.dumps(summary.observed_by_case, ensure_ascii=False, sort_keys=True)}")
        print(f"summary: {output}")
        return 0 if summary.passed == summary.total else 1
    if args.holdout_case_schema is not None:
        if args.holdout_root is not None or args.holdout_case_set_metadata is not None:
            parser.error("--holdout-case-schema cannot be combined with another holdout option")
        try:
            schema = inspect_holdout_case_schema(args.holdout_case_schema)
        except HoldoutFormatError as error:
            parser.error(str(error))
        print(f"case_count: {schema.case_count}")
        print("case_fields: " + ", ".join(schema.fields))
        return 0
    if args.holdout_case_set_metadata is not None:
        if args.holdout_root is not None or args.baseline_v0 or args.compare_baseline is not None:
            parser.error("--holdout-case-set-metadata cannot be combined with holdout run or baseline options")
        try:
            count, digest = holdout_case_set_metadata(args.holdout_case_set_metadata)
        except HoldoutFormatError as error:
            parser.error(str(error))
        print(f"case_count: {count}")
        print(f"case_set_digest: {digest}")
        return 0
    if args.holdout_root is not None:
        if args.baseline_v0 or args.compare_baseline is not None:
            parser.error("--holdout-root cannot be combined with baseline options")
        try:
            summary = run_holdout(args.holdout_root)
        except HoldoutFormatError as error:
            parser.error(str(error))
        output = args.output or default_holdout_report_path(args.holdout_root)
        write_holdout_summary(summary, output)
        print(
            f"{summary.suite}: {summary.passed}/{summary.total} passed; "
            f"case_set_digest={summary.case_set_digest}"
        )
        print(f"categories: {json.dumps(summary.categories, ensure_ascii=False, sort_keys=True)}")
        print("failed_case_ids: " + (", ".join(summary.failed_case_ids) if summary.failed_case_ids else "none"))
        print(f"summary: {output}")
        return 0 if summary.passed == summary.total else 1
    cases = (
        build_core_v0_cases()
        if args.suite == CORE_V0_SUITE
        else build_role_runtime_validation_cases()
        if args.suite == ROLE_RUNTIME_VALIDATION_SUITE
        else build_long_task_runtime_smoke_cases()
        if args.suite == LONG_TASK_RUNTIME_SMOKE_SUITE
        else build_long_task_runtime_v0_cases()
        if args.suite == LONG_TASK_RUNTIME_V0_SUITE
        else build_long_task_observability_v0_cases()
        if args.suite == LONG_TASK_OBSERVABILITY_V0_SUITE
        else build_runner_validation_cases()
    )
    report = ReplayRunner().run(cases)
    payload = report.to_dict()
    comparison = None
    if args.compare_baseline is not None:
        try:
            comparison = BaselineReference.from_path(args.compare_baseline).compare(report)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        payload["comparison"] = comparison.to_dict()
    if args.baseline_v0:
        if args.output is None:
            parser.error("--baseline-v0 requires --output")
        try:
            ReplayRunner.write_baseline(report, args.output)
        except ValueError as error:
            parser.error(str(error))
    elif args.output is not None:
        ReplayRunner.write_report(report, args.output)
    if args.summary:
        print(f"{report.suite}: {report.passed}/{report.total} passed; case_set_digest={report.case_set_digest}")
        if comparison is not None:
            status = "comparable" if comparison.comparable else f"not-comparable:{comparison.reason}"
            print(f"baseline comparison: {status}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if comparison is not None and not comparison.comparable:
        return 2
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
