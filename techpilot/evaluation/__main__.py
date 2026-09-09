"""Run a deterministic TechPilot Runtime Replay suite from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from .runner import ReplayRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline TechPilot Runtime Replay cases.")
    parser.add_argument(
        "--suite",
        default=CORE_V0_SUITE,
        choices=[CORE_V0_SUITE, RUNNER_VALIDATION_SUITE, ROLE_RUNTIME_VALIDATION_SUITE, LONG_TASK_RUNTIME_SMOKE_SUITE, LONG_TASK_RUNTIME_V0_SUITE, LONG_TASK_OBSERVABILITY_V0_SUITE],
    )
    parser.add_argument("--output", type=Path, help="Write the structured result manifest to this JSON path.")
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
