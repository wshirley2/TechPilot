"""Local calibration for the Goal model pilot's independent oracle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from techpilot.evaluation.goal_model_oracle import score_pilot_case
from techpilot.evaluation.goal_model_v0 import load_deck, prepare_pilot


def _prepare(tmp_path: Path) -> Path:
    root = tmp_path / "calibration"
    prepare_pilot(root, model="calibration-only", max_cost_usd=0.01, max_total_tokens=None)
    return root


def _reference_solution(workspace: Path, *, ensure_ascii: bool = False) -> None:
    cli = workspace / "src" / "cli_data_tool" / "cli.py"
    cli.write_text(
        """import argparse
import json

from cli_data_tool.exporter import export_text

DEFAULT_ITEMS = ["alpha", "beta", "gamma"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="data-tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--items", nargs="+", default=DEFAULT_ITEMS)
    export_parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.format == "json":
        print(json.dumps({"items": args.items, "count": len(args.items)}, ensure_ascii=ENSURE_ASCII))
    else:
        print(export_text(args.items))
    return 0
""".replace("ENSURE_ASCII", repr(ensure_ascii)),
        encoding="utf-8",
    )
    readme = workspace / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n## JSON usage\n\n```powershell\n"
        + "python -m cli_data_tool.cli export --format json\n"
        + "```\n",
        encoding="utf-8",
    )


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace_files(workspace: Path) -> dict[str, bytes]:
    return {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*") if path.is_file()
    }


def test_unmodified_fixture_fails_without_touching_workspace(tmp_path: Path) -> None:
    root = _prepare(tmp_path)
    workspace = root / "normal-json-export" / "workspace"
    before = _workspace_files(workspace)

    record = _record(score_pilot_case(root, case_id="normal-json-export", label="baseline"))

    assert record["passed"] is False
    assert record["checks"]["default_text"]["passed"] is True
    assert record["checks"]["frozen_files_unchanged"]["passed"] is True
    assert record["checks"]["readme_example"]["passed"] is False
    assert record["checks"]["json_export"]["passed"] is False
    assert _workspace_files(workspace) == before
    assert record["case_set_digest"] == load_deck().digest


@pytest.mark.parametrize("case_id", [
    "normal-json-export", "restart-json-export", "constraint-json-export",
])
def test_reference_solution_passes_each_case_without_touching_workspace(
    tmp_path: Path, case_id: str,
) -> None:
    root = _prepare(tmp_path)
    workspace = root / case_id / "workspace"
    _reference_solution(workspace)
    before = _workspace_files(workspace)

    record = _record(score_pilot_case(root, case_id=case_id, label="reference"))

    assert record["passed"] is True, record["checks"]
    assert all(check["passed"] for check in record["checks"].values())
    assert ("unicode_json" in record["checks"]) is (case_id == "constraint-json-export")
    assert _workspace_files(workspace) == before


def test_constraint_case_rejects_escaped_unicode(tmp_path: Path) -> None:
    root = _prepare(tmp_path)
    workspace = root / "constraint-json-export" / "workspace"
    _reference_solution(workspace, ensure_ascii=True)

    record = _record(score_pilot_case(root, case_id="constraint-json-export", label="escaped"))

    assert record["passed"] is False
    assert record["checks"]["json_export"]["passed"] is True
    assert record["checks"]["unicode_json"]["passed"] is False


def test_changed_frozen_file_fails_and_record_cannot_be_overwritten(tmp_path: Path) -> None:
    root = _prepare(tmp_path)
    workspace = root / "normal-json-export" / "workspace"
    _reference_solution(workspace)
    frozen_test = workspace / "tests" / "test_export.py"
    frozen_test.write_text("# changed\n", encoding="utf-8")

    output = score_pilot_case(root, case_id="normal-json-export", label="changed")
    original_record = output.read_bytes()

    assert _record(output)["passed"] is False
    assert _record(output)["checks"]["frozen_files_unchanged"]["changed_paths"] == ["tests/test_export.py"]
    with pytest.raises(FileExistsError):
        score_pilot_case(root, case_id="normal-json-export", label="changed")
    assert output.read_bytes() == original_record


def test_unapproved_source_file_fails_even_when_functionality_passes(tmp_path: Path) -> None:
    root = _prepare(tmp_path)
    workspace = root / "normal-json-export" / "workspace"
    _reference_solution(workspace)
    (workspace / "src" / "cli_data_tool" / "extra.py").write_text("VALUE = 1\n", encoding="utf-8")

    record = _record(score_pilot_case(root, case_id="normal-json-export", label="extra"))

    assert record["passed"] is False
    assert record["checks"]["json_export"]["passed"] is True
    assert record["checks"]["allowed_file_scope"] == {
        "passed": False,
        "unapproved_paths": ["src/cli_data_tool/extra.py"],
    }
