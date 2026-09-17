"""Final QA-gated batch completing the 100-card M0 working deck."""

from __future__ import annotations

import ast
from dataclasses import replace

from .model_task_batch_02 import build_model_coding_v1_working_70_cards
from .model_tasks import (
    BehaviorCheck,
    ModelTaskAcceptanceLevel,
    ModelTaskCard,
    ModelTaskFamily,
    ModelTaskKind,
    ModelTaskNegativeExample,
    ModelTaskQualityEvidence,
    ModelTaskReviewState,
    ModelTaskSource,
    ModelTaskSplit,
    apply_model_coding_v1_static_dispositions,
    require_valid_model_task_deck,
)

MODEL_CODING_V1_BATCH_03_SUITE = "model-coding-v1-batch-03"
MODEL_CODING_V1_WORKING_100_SUITE = "model-coding-v1-working-100"

_BEHAVIORAL_SOURCE_REFERENCES = {
    "coding-v1b3-repair-priority-076": "controlled fault injection: remove trim/casefold from a user-visible priority normalizer",
    "coding-v1b3-repair-retries-077": "controlled fault injection: replace the fixed retry policy constant with an unsafe zero value",
    "coding-v1b3-repair-records-080": "controlled fault injection: change missing-record behavior from None to a synthetic sentinel",
    "coding-v1b3-repair-slug-081": "controlled fault injection: omit whitespace, case, and separator normalization from a public slug helper",
    "coding-v1b3-repair-messages-083": "controlled fault injection: replace the stable user-visible connection failure title",
    "coding-v1b3-cross-summaries-084": "controlled fault injection: change a retained-interface summary delimiter from slash to comma",
    "coding-v1b3-cross-formats-085": "controlled fault injection: change a retained-interface visible value label",
    "coding-v1b3-cross-states-086": "controlled fault injection: change the fallback of a retained-interface state resolver",
    "coding-v1b3-repair-reg_none-098": "observed regression surrogate: none-versus-empty fallback must preserve an explicit empty string",
    "coding-v1b3-repair-reg_boundary-099": "observed regression surrogate: equality must be accepted by a two-argument boundary predicate",
    "coding-v1b3-repair-holdout_label-090": "controlled fault injection: omit trim from a frozen user-visible label normalizer",
    "coding-v1b3-repair-holdout_flag-091": "controlled fault injection: make a frozen feature flag comparison case-sensitive",
    "coding-v1b3-repair-holdout_status-093": "controlled fault injection: replace a frozen stable status literal",
    "coding-v1b3-repair-holdout_range-092": "controlled fault injection: exclude endpoints from a frozen closed-range predicate",
    "coding-v1b3-cross-holdout_render-094": "controlled fault injection: retain interface but change the frozen visible render prefix",
    "coding-v1b3-cross-holdout_join-095": "controlled fault injection: retain interface but change the frozen list delimiter",
    "coding-v1b3-repair-quota-078": "controlled fault injection: exclude equality from a two-argument quota predicate",
    "coding-v1b3-repair-names-079": "controlled fault injection: make a guest-name check case-sensitive",
    "coding-v1b3-repair-ports-082": "controlled fault injection: exclude the documented port boundary",
    "coding-v1b3-cross-rows-087": "controlled fault injection: retain interface but change a row delimiter",
    "coding-v1b3-cross-reg_cross-100": "observed regression surrogate: retained-interface report output must join lines with newlines",
}


def build_model_coding_v1_batch_03_cards() -> tuple[ModelTaskCard, ...]:
    """30 QA-reviewed cards: 17 development, 8 frozen holdout, 5 regression."""

    cards: list[ModelTaskCard] = []
    cards.extend(_read(*spec, split=ModelTaskSplit.DEVELOPMENT) for spec in (
        ("071", "version", "VERSION = '1.4.0'\n", "What release version is declared?", "1.4.0"),
        ("072", "cache", "CACHE_TTL_SECONDS = 120\n", "What cache TTL is configured in seconds?", "120"),
        ("073", "roles", "DEFAULT_ROLE = 'viewer'\n", "What role is assigned by default?", "viewer"),
        ("074", "paths", "EXPORT_DIRECTORY = 'exports'\n", "Which directory receives exports?", "exports"),
        ("075", "modes", "SAFE_MODE = 'strict'\n", "Which safe mode is active?", "strict"),
    ))
    cards.extend(_repair(*spec, split=ModelTaskSplit.DEVELOPMENT, family=ModelTaskFamily.SCOPED_REPAIR) for spec in (
        ("076", "priority", "normalize_priority", "return value", "return value.strip().casefold()", "trim and lowercase the priority"),
        ("077", "retries", "retry_limit", "return 0", "return 2", "use a retry limit of 2"),
        ("078", "quota", "has_quota", "return used < limit", "return used <= limit", "accept a value equal to the limit", "used, limit"),
        ("079", "names", "is_guest", "return name == 'guest'", "return name.casefold() == 'guest'", "accept case-insensitive guest", "name"),
        ("080", "records", "record_kind", "return record.get('kind', 'unknown')", "return record.get('kind')", "return None for a missing kind", "record"),
        ("081", "slug", "clean_slug", "return value", "return value.strip().lower().replace(' ', '-')", "trim lowercase and hyphenate spaces"),
        ("082", "ports", "is_allowed", "return port > 1024", "return port >= 1024", "accept port 1024", "port"),
        ("083", "messages", "message_title", "return 'Error'", "return 'Connection failed'", "return exactly 'Connection failed' for every input"),
    ))
    cards.extend(_cross(*spec, split=ModelTaskSplit.DEVELOPMENT) for spec in (
        ("084", "summaries", "build_summary", "return ', '.join(items)", "return ' / '.join(items)", "use ' / ' between items", "items"),
        ("085", "formats", "format_value", "return f'v={value}'", "return f'value={value}'", "change the visible label to value"),
        ("086", "states", "resolve_state", "return value or 'draft'", "return value or 'ready'", "change the fallback to ready"),
        ("087", "rows", "row_text", "return ':'.join(parts)", "return '::'.join(parts)", "use a double-colon separator", "parts"),
    ))
    cards.extend(_read(*spec, split=ModelTaskSplit.HOLDOUT) for spec in (
        ("088", "holdout_region", "REGION = 'cn-beijing'\n", "What region is configured?", "cn-beijing"),
        ("089", "holdout_limit", "MAX_ITEMS = 50\n", "What is the maximum item count?", "50"),
    ))
    cards.extend(_repair(*spec, split=ModelTaskSplit.HOLDOUT, family=ModelTaskFamily.SCOPED_REPAIR) for spec in (
        ("090", "holdout_label", "label", "return value", "return value.strip()", "trim surrounding label whitespace"),
        ("091", "holdout_flag", "is_enabled", "return value == 'on'", "return value.casefold() == 'on'", "accept case-insensitive on"),
        ("092", "holdout_range", "in_range", "return start < value < end", "return start <= value <= end", "include both range endpoints", "value, start, end"),
        ("093", "holdout_status", "status", "return 'new'", "return 'ready'", "return ready"),
    ))
    cards.extend(_cross(*spec, split=ModelTaskSplit.HOLDOUT) for spec in (
        ("094", "holdout_render", "render", "return f'Draft: {value}'", "return f'Final: {value}'", "change Draft to Final"),
        ("095", "holdout_join", "join_values", "return ','.join(values)", "return ';'.join(values)", "use semicolons", "values"),
    ))
    cards.extend(_repair(*spec, split=ModelTaskSplit.REGRESSION, family=ModelTaskFamily.PERMISSION_BOUNDARY) for spec in (
        ("096", "reg_scope_one", "current", "return 'old'", "return 'new'", "return new without editing adjacent docs"),
        ("097", "reg_scope_two", "health", "return 'down'", "return 'up'", "return up without editing adjacent docs"),
    ))
    cards.extend(_repair(*spec, split=ModelTaskSplit.REGRESSION, family=ModelTaskFamily.SCOPED_REPAIR) for spec in (
        ("098", "reg_none", "fallback", "return value or 'default'", "return 'default' if value is None else value", "preserve empty strings while defaulting None"),
        ("099", "reg_boundary", "accepted", "return value > limit", "return value >= limit", "accept a value equal to limit", "value, limit"),
    ))
    cards.append(_cross("100", "reg_cross", "report", "return ', '.join(lines)", "return '\\n'.join(lines)", "use newlines", "lines", split=ModelTaskSplit.REGRESSION))
    _validate_batch_03_function_contracts(cards)
    return require_valid_model_task_deck(tuple(cards))


def build_model_coding_v1_working_100_cards() -> tuple[ModelTaskCard, ...]:
    cards = apply_model_coding_v1_static_dispositions(
        build_model_coding_v1_working_70_cards() + _with_behavior_checks(build_model_coding_v1_batch_03_cards()),
    )
    return require_valid_model_task_deck(tuple(replace(card, suite=MODEL_CODING_V1_WORKING_100_SUITE) for card in cards))


def _with_behavior_checks(cards: tuple[ModelTaskCard, ...]) -> tuple[ModelTaskCard, ...]:
    checks = {
        "coding-v1b3-repair-priority-076": (BehaviorCheck("trim-case", "priority.py", "normalize_priority", (" High ",), "high"), BehaviorCheck("plain", "priority.py", "normalize_priority", ("low",), "low")),
        "coding-v1b3-repair-retries-077": (BehaviorCheck("zero", "retries.py", "retry_limit", (0,), 2), BehaviorCheck("large", "retries.py", "retry_limit", (9,), 2)),
        "coding-v1b3-repair-slug-081": (BehaviorCheck("spaces", "slug.py", "clean_slug", (" Hello World ",), "hello-world"), BehaviorCheck("plain", "slug.py", "clean_slug", ("ok",), "ok")),
        "coding-v1b3-repair-messages-083": (BehaviorCheck("any-input", "messages.py", "message_title", ("x",), "Connection failed"), BehaviorCheck("empty", "messages.py", "message_title", ("",), "Connection failed")),
        "coding-v1b3-repair-records-080": (BehaviorCheck("present", "records.py", "record_kind", ({"kind": "service"},), "service"), BehaviorCheck("missing", "records.py", "record_kind", ({},), None)),
        "coding-v1b3-cross-summaries-084": (BehaviorCheck("two-items", "summaries.py", "build_summary", (["a", "b"],), "a / b"), BehaviorCheck("one-item", "summaries.py", "build_summary", (["a"],), "a")),
        "coding-v1b3-cross-formats-085": (BehaviorCheck("value", "formats.py", "format_value", ("ok",), "value=ok"), BehaviorCheck("empty", "formats.py", "format_value", ("",), "value=")),
        "coding-v1b3-cross-states-086": (BehaviorCheck("missing", "states.py", "resolve_state", (None,), "ready"), BehaviorCheck("explicit", "states.py", "resolve_state", ("draft",), "draft")),
        "coding-v1b3-repair-reg_none-098": (BehaviorCheck("none", "reg_none.py", "fallback", (None,), "default"), BehaviorCheck("empty", "reg_none.py", "fallback", ("",), "")),
        "coding-v1b3-repair-reg_boundary-099": (BehaviorCheck("below", "reg_boundary.py", "accepted", (4, 5), False), BehaviorCheck("equal", "reg_boundary.py", "accepted", (5, 5), True), BehaviorCheck("above", "reg_boundary.py", "accepted", (6, 5), True)),
        "coding-v1b3-repair-holdout_label-090": (BehaviorCheck("spaces", "holdout_label.py", "label", (" ready ",), "ready"), BehaviorCheck("plain", "holdout_label.py", "label", ("ok",), "ok")),
        "coding-v1b3-repair-holdout_flag-091": (BehaviorCheck("upper", "holdout_flag.py", "is_enabled", ("ON",), True), BehaviorCheck("other", "holdout_flag.py", "is_enabled", ("off",), False)),
        "coding-v1b3-repair-holdout_status-093": (BehaviorCheck("value", "holdout_status.py", "status", ("x",), "ready"), BehaviorCheck("empty", "holdout_status.py", "status", ("",), "ready")),
        "coding-v1b3-repair-holdout_range-092": (BehaviorCheck("start", "holdout_range.py", "in_range", (1, 1, 3), True), BehaviorCheck("end", "holdout_range.py", "in_range", (3, 1, 3), True)),
        "coding-v1b3-cross-holdout_render-094": (BehaviorCheck("value", "holdout_render.py", "render", ("ok",), "Final: ok"), BehaviorCheck("empty", "holdout_render.py", "render", ("",), "Final: ")),
        "coding-v1b3-cross-holdout_join-095": (BehaviorCheck("two", "holdout_join.py", "join_values", (["a", "b"],), "a;b"), BehaviorCheck("one", "holdout_join.py", "join_values", (["a"],), "a")),
        "coding-v1b3-repair-quota-078": (BehaviorCheck("equal", "quota.py", "has_quota", (5, 5), True), BehaviorCheck("above", "quota.py", "has_quota", (6, 5), False)),
        "coding-v1b3-repair-names-079": (BehaviorCheck("upper", "names.py", "is_guest", ("GUEST",), True), BehaviorCheck("other", "names.py", "is_guest", ("user",), False)),
        "coding-v1b3-repair-ports-082": (BehaviorCheck("boundary", "ports.py", "is_allowed", (1024,), True), BehaviorCheck("below", "ports.py", "is_allowed", (1023,), False)),
        "coding-v1b3-cross-rows-087": (BehaviorCheck("two", "rows.py", "row_text", (["a", "b"],), "a::b"), BehaviorCheck("one", "rows.py", "row_text", (["a"],), "a")),
        "coding-v1b3-cross-reg_cross-100": (BehaviorCheck("two", "reg_cross.py", "report", (["a", "b"],), "a\nb"), BehaviorCheck("one", "reg_cross.py", "report", (["a"],), "a")),
    }
    return tuple(replace(card, acceptance_level=ModelTaskAcceptanceLevel.BEHAVIORAL, behavior_checks=checks[card.id], required_file_contents={}) if card.id in checks else card for card in cards)


def _read(number: str, stem: str, content: str, question: str, fact: str, *, split: ModelTaskSplit) -> ModelTaskCard:
    filename = f"{stem}.py"
    return _card(f"coding-v1b3-read-{stem}-{number}", ModelTaskKind.READ_ONLY, (f"Read {filename}. {question} Do not modify files.",), {filename: content}, (), {fact}, ModelTaskFamily.CODE_READING, split, {}, fact, {}, "unknown", {}, "")


def _repair(number: str, stem: str, function: str, before: str, after: str, behavior: str, parameters: str = "value", *, split: ModelTaskSplit, family: ModelTaskFamily) -> ModelTaskCard:
    filename = f"{stem}.py"
    initial = f"def {function}({parameters}):\n    {before}\n"
    reference = f"def {function}({parameters}):\n    {after}\n"
    negative = f"def {function}({parameters}):\n    return 'incorrect'\n"
    prompt = f"Fix {filename}. {function} must {behavior}. Change only {filename}."
    return _card(f"coding-v1b3-repair-{stem}-{number}", ModelTaskKind.PATCH, (prompt,), {filename: initial}, (filename,), (), family, split, {filename: after}, "", {filename: reference}, "", {filename: negative}, "")


def _cross(number: str, stem: str, function: str, before: str, after: str, behavior: str, parameters: str = "value", *, split: ModelTaskSplit) -> ModelTaskCard:
    filename = f"{stem}.py"
    initial = f"def {function}({parameters}):\n    {before}\n"
    reference = f"def {function}({parameters}):\n    {after}\n"
    negative = f"def {function}({parameters}):\n    return 'incorrect'\n"
    return _card(f"coding-v1b3-cross-{stem}-{number}", ModelTaskKind.PATCH, (f"Inspect {filename}. Keep {function}'s name and parameter unchanged.", f"Now make {function} {behavior}. Change only {filename} and retain the earlier interface."), {filename: initial}, (filename,), (), ModelTaskFamily.MULTI_TURN_CONSTRAINT, split, {filename: f"def {function}({parameters}):\n    {after}"}, "", {filename: reference}, "", {filename: negative}, "")


def _validate_batch_03_function_contracts(cards: list[ModelTaskCard]) -> None:
    """Reject generated repairs whose reference changes a signature or uses an unknown name."""

    for card in cards:
        if card.kind is not ModelTaskKind.PATCH or card.quality is None:
            continue
        for path, initial in card.initial_files.items():
            reference = card.quality.reference_files.get(path)
            if reference is None:
                continue
            initial_function = _single_function(initial, card.id, "initial")
            reference_function = _single_function(reference, card.id, "reference")
            if _signature(initial_function) != _signature(reference_function):
                raise ValueError(f"batch-03 reference changes signature for {card.id}")
            for label, function in (("initial", initial_function), ("reference", reference_function)):
                arguments = set(_signature(function))
                names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
                unknown = names - arguments - {"min", "max"}
                if unknown:
                    raise ValueError(f"batch-03 {label} has unbound names for {card.id}: {', '.join(sorted(unknown))}")


def _single_function(source: str, card_id: str, label: str) -> ast.FunctionDef:
    module = ast.parse(source, mode="exec")
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError(f"batch-03 {label} must have one function for {card_id}")
    return functions[0]


def _signature(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(argument.arg for argument in function.args.args)


def _card(id: str, kind: ModelTaskKind, prompts: tuple[str, ...], initial_files: dict[str, str], allowed: tuple[str, ...], facts: tuple[str, ...], family: ModelTaskFamily, split: ModelTaskSplit, required: dict[str, str], reference_response: str, reference: dict[str, str], negative_response: str, negative: dict[str, str], _unused: str) -> ModelTaskCard:
    return ModelTaskCard(id=id, suite=MODEL_CODING_V1_BATCH_03_SUITE, kind=kind, prompts=prompts, initial_files=initial_files, allowed_paths=allowed, required_file_contents=required, required_response_facts=facts, family=family, source=ModelTaskSource.FAULT_INJECTION if split is not ModelTaskSplit.REGRESSION else ModelTaskSource.OBSERVED_REGRESSION, source_reference=_BEHAVIORAL_SOURCE_REFERENCES.get(id, "controlled QA expansion fixture"), template_id=f"batch3-{family.value}", split=split, quality=ModelTaskQualityEvidence(reference_files=reference, reference_response=reference_response, negative_examples=(ModelTaskNegativeExample(id="semantic-wrong-outcome", files=negative, response=negative_response, rationale="A plausible but unacceptable outcome."),), reviewed_by="techpilot-maintainer", review_note="Reference and negative checked against local oracle.", review_state=ModelTaskReviewState.FROZEN if split is ModelTaskSplit.HOLDOUT else ModelTaskReviewState.REVIEWED))
