"""Score a prepared Goal pilot candidate independently of Goal self-assessment.

The scorer executes only a temporary copy of the frozen fixture plus permitted
candidate files. It never calls a model and never runs code in the source workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from techpilot.evaluation.goal_model_v0 import SCHEMA_VERSION, SUITE, GoalPilotDeck, load_deck

_LABEL = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_MAX_EXCERPT = 2000
_COMMAND_TIMEOUT_SECONDS = 10
_RUNTIME_DIRECTORIES = {".techpilot", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
_ENTRY = (
    "import sys; "
    "sys.path.insert(0, sys.argv[1]); "
    "from cli_data_tool.cli import main; "
    "raise SystemExit(main(sys.argv[2:]))"
)


def _candidate_files(workspace: Path, paths: set[str]) -> dict[str, bytes | None]:
    result: dict[str, bytes | None] = {}
    for name in sorted(paths):
        source = (workspace / name).resolve()
        if not source.is_relative_to(workspace):
            raise ValueError(f"Goal pilot candidate path escapes the workspace: {name}")
        result[name] = source.read_bytes() if source.is_file() else None
    return result


def _candidate_digest(files: dict[str, bytes | None], unapproved_paths: list[str]) -> str:
    hashes = {
        name: hashlib.sha256(content).hexdigest() if content is not None else None
        for name, content in files.items()
    }
    return hashlib.sha256(json.dumps(
        {"candidate_hashes": hashes, "unapproved_paths": unapproved_paths},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def candidate_digest_for_workspace(workspace: Path, *, deck: GoalPilotDeck | None = None) -> str:
    """Fingerprint exactly the frozen fixture and permitted candidate files."""

    selected = deck or load_deck()
    names = set(selected.fixture_hashes) | set(selected.raw["allowed_paths"])
    files = _candidate_files(workspace.resolve(), names)
    return _candidate_digest(files, _unapproved_paths(workspace.resolve(), names))


def _unapproved_paths(workspace: Path, approved: set[str]) -> list[str]:
    unexpected = []
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if any(part in _RUNTIME_DIRECTORIES for part in relative.parts):
            continue
        if (path.is_file() or path.is_symlink()) and relative.as_posix() not in approved:
            unexpected.append(relative.as_posix())
    return sorted(unexpected)


def _command(candidate_files: dict[str, bytes | None], arguments: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="techpilot-goal-oracle-") as temporary:
        working_copy = Path(temporary)
        for name, content in candidate_files.items():
            if content is None:
                continue
            target = working_copy / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        environment = {
            key: value for key, value in os.environ.items()
            if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR"}
        }
        environment.update({
            "TMP": str(working_copy),
            "TEMP": str(working_copy),
            "TMPDIR": str(working_copy),
        })
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _ENTRY, str(working_copy / "src"), *arguments],
                cwd=working_copy,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "error": None,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            return {
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": type(error).__name__,
            }


def _record_command(result: dict[str, Any], passed: bool) -> dict[str, Any]:
    return {
        "passed": passed,
        "returncode": result["returncode"],
        "stdout_excerpt": result["stdout"][:_MAX_EXCERPT],
        "stderr_excerpt": result["stderr"][:_MAX_EXCERPT],
        "error": result["error"],
    }


def _json_result(result: dict[str, Any], items: list[str]) -> bool:
    if result["returncode"] != 0 or result["error"] is not None:
        return False
    try:
        payload = json.loads(result["stdout"])
    except (TypeError, ValueError):
        return False
    return payload == {"items": items, "count": len(items)}


def score_pilot_case(
    prepared_root: Path, *, case_id: str, label: str, deck: GoalPilotDeck | None = None,
) -> Path:
    """Run the frozen external oracle and write a new, separate result record."""

    if not _LABEL.fullmatch(label):
        raise ValueError("Goal pilot assessment label must be a safe lowercase name")
    selected = deck or load_deck()
    case = next((item for item in selected.cases if item.case_id == case_id), None)
    if case is None:
        raise ValueError("Goal pilot case is not part of the frozen deck")
    root = prepared_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("record_type") != "preparation"
        or manifest.get("suite") != SUITE
        or manifest.get("case_set_digest") != selected.digest
        or not isinstance(manifest.get("cases"), list)
        or case_id not in manifest["cases"]
    ):
        raise ValueError("Goal pilot manifest does not match this frozen deck")
    case_root = (root / case_id).resolve()
    workspace = (case_root / "workspace").resolve()
    if not case_root.is_relative_to(root) or not workspace.is_relative_to(case_root) or not workspace.is_dir():
        raise ValueError("Goal pilot candidate workspace is missing or outside the prepared output")
    acceptance = selected.raw["external_acceptance"]
    allowed = set(selected.raw["allowed_paths"])
    fixture_paths = set(selected.fixture_hashes)
    candidate_files = _candidate_files(workspace, fixture_paths | allowed)
    candidate_hashes = {
        name: hashlib.sha256(content).hexdigest() if content is not None else None
        for name, content in candidate_files.items()
    }
    unapproved = _unapproved_paths(workspace, fixture_paths | allowed)
    candidate_digest = _candidate_digest(candidate_files, unapproved)
    unchanged = {
        name: candidate_hashes[name] == expected
        for name, expected in selected.fixture_hashes.items() if name not in allowed
    }
    try:
        readme_has_example = acceptance["readme_example"] in (candidate_files["README.md"] or b"").decode("utf-8")
    except UnicodeDecodeError:
        readme_has_example = False
    checks: dict[str, dict[str, Any]] = {
        "frozen_files_unchanged": {
            "passed": all(unchanged.values()),
            "changed_paths": sorted(name for name, passed in unchanged.items() if not passed),
        },
        "allowed_file_scope": {"passed": not unapproved, "unapproved_paths": unapproved},
        "readme_example": {"passed": readme_has_example},
    }
    default_text = _command(candidate_files, ["export"])
    checks["default_text"] = _record_command(
        default_text,
        default_text["returncode"] == 0 and default_text["stdout"] == acceptance["default_text_stdout"],
    )
    explicit_text = _command(candidate_files, ["export", "--format", "text", "--items", "one", "two"])
    checks["explicit_text"] = _record_command(
        explicit_text,
        explicit_text["returncode"] == 0 and explicit_text["stdout"] == acceptance["explicit_text_stdout"],
    )
    items = acceptance["json_items"]
    json_export = _command(candidate_files, ["export", "--format", "json", "--items", *items])
    checks["json_export"] = _record_command(json_export, _json_result(json_export, items))
    invalid = _command(candidate_files, ["export", "--format", acceptance["invalid_format"]])
    checks["invalid_format"] = _record_command(
        invalid,
        invalid["returncode"] is not None and invalid["returncode"] != 0,
    )
    if case.kind == "constraint_after_stage":
        unicode_item = acceptance["unicode_item"]
        unicode_export = _command(candidate_files, ["export", "--format", "json", "--items", unicode_item])
        checks["unicode_json"] = _record_command(
            unicode_export,
            _json_result(unicode_export, [unicode_item]) and unicode_item in unicode_export["stdout"],
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "track": "model",
        "record_type": "external_acceptance",
        "case_id": case_id,
        "case_set_digest": selected.digest,
        "candidate_digest": candidate_digest,
        "label": label,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
    assessments = case_root / "assessments"
    assessments.mkdir(exist_ok=True)
    output = assessments / f"{label}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a prepared Goal pilot candidate without calling a model.")
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--deck", type=Path)
    args = parser.parse_args(argv)
    deck = load_deck(args.deck) if args.deck else load_deck()
    print(score_pilot_case(args.prepared_root, case_id=args.case_id, label=args.label, deck=deck))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
