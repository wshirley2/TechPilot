"""First S1+S2 expansion batch for the formal M0 v2 Coding deck."""

from __future__ import annotations

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
    require_valid_model_task_deck,
)

MODEL_CODING_V2_BATCH_01_SUITE = "model-coding-v2-batch-01"


def build_model_coding_v2_batch_01_cards() -> tuple[ModelTaskCard, ...]:
    """Return 20 reviewed S1+S2 cards: 12 development, 6 Holdout, 2 regression."""

    cards = (
        _single(
            "coding-v2b1-dev-normalize-alias-101", "alias", "canonical_alias", "value: str", "return value", "return value.strip().lower()", "strip whitespace and lowercase the alias",
            (BehaviorCheck("spaces", "alias.py", "canonical_alias", (" Ready ",), "ready"), BehaviorCheck("plain", "alias.py", "canonical_alias", ("ok",), "ok")),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: omit trim and case normalization from a public alias", "return value.lower()",
        ),
        _single(
            "coding-v2b1-dev-clamp-score-102", "score", "bounded_score", "value: int", "return value", "return max(0, min(value, 10))", "clamp the score to the inclusive range 0 through 10",
            (BehaviorCheck("below", "score.py", "bounded_score", (-1,), 0), BehaviorCheck("middle", "score.py", "bounded_score", (5,), 5), BehaviorCheck("above", "score.py", "bounded_score", (11,), 10)),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: remove both public score bounds", "return min(value, 10)",
        ),
        _single(
            "coding-v2b1-dev-retry-boundary-103", "retries", "can_retry", "attempt: int, maximum: int", "return attempt < maximum", "return attempt <= maximum", "allow the final configured retry attempt",
            (BehaviorCheck("below", "retries.py", "can_retry", (2, 3), True), BehaviorCheck("equal", "retries.py", "can_retry", (3, 3), True), BehaviorCheck("above", "retries.py", "can_retry", (4, 3), False)),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.OBSERVED_REGRESSION, "observed regression surrogate: final retry was excluded by a strict inequality", "return attempt == maximum",
        ),
        _single(
            "coding-v2b1-dev-caption-none-104", "caption", "resolve_caption", "value: str | None", "return value or 'untitled'", "return 'untitled' if value is None else value", "default only None while preserving an explicit empty caption",
            (BehaviorCheck("none", "caption.py", "resolve_caption", (None,), "untitled"), BehaviorCheck("empty", "caption.py", "resolve_caption", ("",), "")),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: collapse explicit empty text into a fallback", "return 'untitled' if not value else value",
        ),
        _single(
            "coding-v2b1-dev-columns-tab-105", "columns", "render_columns", "columns: list[str]", "return ', '.join(columns)", "return '\\t'.join(columns)", "join columns with a tab separator",
            (BehaviorCheck("two", "columns.py", "render_columns", (["id", "name"],), "id\tname"), BehaviorCheck("one", "columns.py", "render_columns", (["id"],), "id")),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.MANUAL_CONTRACT, "manual output-format contract: export columns use tabs", "return ' '.join(columns)", family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
        ),
        _single(
            "coding-v2b1-dev-total-format-106", "total", "format_total", "value: int", "return f'count={value}'", "return f'total={value}'", "use total as the visible label",
            (BehaviorCheck("positive", "total.py", "format_total", (2,), "total=2"), BehaviorCheck("zero", "total.py", "format_total", (0,), "total=0")),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: replace the stable total label", "return f'total: {value}'",
        ),
        _single(
            "coding-v2b1-dev-event-fields-107", "event", "event_summary", "event: dict[str, object]", "return {'id': event['id'], 'status': event['status']}", "return {'id': event['id'], 'status': event['status'], 'active': event['active']}", "return id, status, and active fields",
            (BehaviorCheck("active", "event.py", "event_summary", ({"id": "e-1", "status": "open", "active": True},), {"id": "e-1", "status": "open", "active": True}), BehaviorCheck("inactive", "event.py", "event_summary", ({"id": "e-2", "status": "closed", "active": False},), {"id": "e-2", "status": "closed", "active": False})),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.OBSERVED_REGRESSION, "observed regression surrogate: serializer omitted a promised boolean field", "return {'id': event['id'], 'active': event['active']}",
        ),
        _constant(
            "coding-v2b1-dev-constant-scheme-108", "scheme", "DEFAULT_SCHEME", "https", "link", "default_scheme", "return 'http'", "return DEFAULT_SCHEME", "use the public default scheme as the only source of truth",
            (BehaviorCheck("configured", "link.py", "default_scheme", (), "https", constant_imports=(("scheme", "DEFAULT_SCHEME"),)), BehaviorCheck("source-binding", "link.py", "default_scheme", (), "https", constant_imports=(("scheme", "DEFAULT_SCHEME"),))),
            ModelTaskSplit.DEVELOPMENT, "manual source-of-truth contract: a client default must bind its sibling public constant", "return 'https'",
        ),
        _single(
            "coding-v2b1-dev-public-key-109", "keys", "public_key", "value: str", "return value", "return value.removeprefix('public_')", "remove exactly one public_ prefix",
            (BehaviorCheck("one", "keys.py", "public_key", ("public_alpha",), "alpha"), BehaviorCheck("two", "keys.py", "public_key", ("public_public_alpha",), "public_alpha")),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: retain a public storage prefix in returned identifiers", "return value.replace('public_', '')",
        ),
        _single(
            "coding-v2b1-dev-tag-membership-110", "tags", "has_tag", "tags: dict[str, str], tag: str", "return False", "return tag in tags", "report whether the declared tag is present",
            (BehaviorCheck("present", "tags.py", "has_tag", ({"prod": "enabled"}, "prod"), True), BehaviorCheck("missing", "tags.py", "has_tag", ({"prod": "enabled"}, "test"), False)),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: replace membership behavior with a fixed false result", "return tag not in tags",
        ),
        _single(
            "coding-v2b1-dev-percent-range-111", "percent", "is_percent", "value: int", "return 0 < value < 100", "return 0 <= value <= 100", "include both public percentage boundaries",
            (BehaviorCheck("zero", "percent.py", "is_percent", (0,), True), BehaviorCheck("hundred", "percent.py", "is_percent", (100,), True), BehaviorCheck("above", "percent.py", "is_percent", (101,), False)),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: exclude documented percentage endpoints", "return 0 < value <= 100",
        ),
        _single(
            "coding-v2b1-dev-mode-default-112", "mode", "selected_mode", "value: str | None", "return value or 'draft'", "return value or 'safe'", "change the fallback to safe while preserving explicit modes",
            (BehaviorCheck("missing", "mode.py", "selected_mode", (None,), "safe"), BehaviorCheck("explicit", "mode.py", "selected_mode", ("fast",), "fast")),
            ModelTaskSplit.DEVELOPMENT, ModelTaskSource.MANUAL_CONTRACT, "manual cross-turn contract: preserve the function interface while updating its fallback", "return 'safe'",
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
        ),
        _single(
            "coding-v2b1-holdout-slug-113", "holdout_slug", "slug", "value: str", "return value", "return value.strip().lower().replace(' ', '-')", "trim, lowercase, and hyphenate the frozen slug",
            (BehaviorCheck("spaces", "holdout_slug.py", "slug", (" Hello World ",), "hello-world"), BehaviorCheck("plain", "holdout_slug.py", "slug", ("ok",), "ok")),
            ModelTaskSplit.HOLDOUT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: omit frozen slug normalization", "return value.strip().lower()",
        ),
        _single(
            "coding-v2b1-holdout-switch-114", "holdout_switch", "is_enabled", "value: str", "return value == 'enabled'", "return value.casefold() == 'enabled'", "accept case-insensitive enabled",
            (BehaviorCheck("upper", "holdout_switch.py", "is_enabled", ("ENABLED",), True), BehaviorCheck("other", "holdout_switch.py", "is_enabled", ("disabled",), False)),
            ModelTaskSplit.HOLDOUT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: make a frozen switch comparison case-sensitive", "return value.lower() == 'disabled'",
        ),
        _single(
            "coding-v2b1-holdout-template-115", "holdout_template", "render", "value: str, prefix: str = 'Draft'", "return f'{prefix}: {value}'", "return f'{prefix}: {value}'", "change the default prefix from Draft to Final while retaining the interface",
            (BehaviorCheck("default", "holdout_template.py", "render", ("ok",), "Final: ok"), BehaviorCheck("custom", "holdout_template.py", "render", ("ok", "Alert"), "Alert: ok")),
            ModelTaskSplit.HOLDOUT, ModelTaskSource.MANUAL_CONTRACT, "frozen cross-turn contract: update only a default while retaining caller override behavior", "return f'Final: {value}'",
            reference_parameters="value: str, prefix: str = 'Final'", family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
        ),
        _constant(
            "coding-v2b1-holdout-constant-title-116", "titles", "PUBLIC_TITLE", "Pilot", "banner", "title", "return 'Draft'", "return PUBLIC_TITLE", "use the frozen public title as the only source of truth",
            (BehaviorCheck("visible", "banner.py", "title", (), "Pilot", constant_imports=(("titles", "PUBLIC_TITLE"),)), BehaviorCheck("source-binding", "banner.py", "title", (), "Pilot", constant_imports=(("titles", "PUBLIC_TITLE"),))),
            ModelTaskSplit.HOLDOUT, "frozen source-of-truth contract: banner title must bind a sibling public constant", "return 'Pilot'",
        ),
        _single(
            "coding-v2b1-holdout-range-117", "holdout_range", "within", "value: int, lower: int, upper: int", "return lower < value < upper", "return lower <= value <= upper", "include both frozen range boundaries",
            (BehaviorCheck("lower", "holdout_range.py", "within", (1, 1, 3), True), BehaviorCheck("upper", "holdout_range.py", "within", (3, 1, 3), True), BehaviorCheck("outside", "holdout_range.py", "within", (4, 1, 3), False)),
            ModelTaskSplit.HOLDOUT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: exclude frozen closed-range endpoints", "return lower < value <= upper",
        ),
        _single(
            "coding-v2b1-holdout-label-118", "holdout_label", "display_label", "value: str | None", "return value or 'pending'", "return 'pending' if value is None else value", "default only None while preserving an explicit empty label",
            (BehaviorCheck("none", "holdout_label.py", "display_label", (None,), "pending"), BehaviorCheck("empty", "holdout_label.py", "display_label", ("",), "")),
            ModelTaskSplit.HOLDOUT, ModelTaskSource.FAULT_INJECTION, "controlled fault injection: collapse frozen explicit empty label into fallback", "return 'pending' if value == '' else value",
        ),
        _single(
            "coding-v2b1-regression-falsy-119", "regression_falsy", "fallback", "value: bool | None", "return value or True", "return True if value is None else value", "preserve false while defaulting only None",
            (BehaviorCheck("none", "regression_falsy.py", "fallback", (None,), True), BehaviorCheck("false", "regression_falsy.py", "fallback", (False,), False)),
            ModelTaskSplit.REGRESSION, ModelTaskSource.OBSERVED_REGRESSION, "observed regression surrogate: a false feature choice was replaced by its fallback", "return False if value is None else value",
        ),
        _single(
            "coding-v2b1-regression-lines-120", "regression_lines", "report", "lines: list[str]", "return ', '.join(lines)", "return '\\n'.join(lines)", "join report lines with newlines",
            (BehaviorCheck("two", "regression_lines.py", "report", (["a", "b"],), "a\nb"), BehaviorCheck("one", "regression_lines.py", "report", (["a"],), "a")),
            ModelTaskSplit.REGRESSION, ModelTaskSource.OBSERVED_REGRESSION, "observed regression surrogate: a retained-interface report changed newline output to commas", "return ';'.join(lines)", family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
        ),
    )
    return require_valid_model_task_deck(cards)


def _single(
    card_id: str,
    stem: str,
    function: str,
    parameters: str,
    before: str,
    after: str,
    behavior: str,
    checks: tuple[BehaviorCheck, ...],
    split: ModelTaskSplit,
    source: ModelTaskSource,
    source_reference: str,
    negative: str,
    *,
    reference_parameters: str | None = None,
    family: ModelTaskFamily = ModelTaskFamily.SCOPED_REPAIR,
) -> ModelTaskCard:
    filename = f"{stem}.py"
    initial = f"def {function}({parameters}):\n    {before}\n"
    reference = f"def {function}({reference_parameters or parameters}):\n    {after}\n"
    negative_source = f"def {function}({parameters}):\n    {negative}\n"
    prompts = (
        (f"Inspect {filename}. Keep {function}'s name and parameters unchanged.", f"Now make {function} {behavior}. Change only {filename} and retain the earlier interface.")
        if family is ModelTaskFamily.MULTI_TURN_CONSTRAINT
        else (f"Fix {filename}. {function} must {behavior}. Change only {filename} and retain its public interface.",)
    )
    return _card(card_id, prompts, {filename: initial}, (filename,), checks, family, split, source, source_reference, {filename: reference}, {filename: negative_source})


def _constant(
    card_id: str,
    module: str,
    constant: str,
    value: str,
    stem: str,
    function: str,
    before: str,
    after: str,
    behavior: str,
    checks: tuple[BehaviorCheck, ...],
    split: ModelTaskSplit,
    source_reference: str,
    negative: str,
) -> ModelTaskCard:
    module_path = f"{module}.py"
    filename = f"{stem}.py"
    initial = {module_path: f"{constant} = '{value}'\n", filename: f"def {function}() -> str:\n    {before}\n"}
    reference = {filename: f"from {module} import {constant}\n\ndef {function}() -> str:\n    {after}\n"}
    negative_source = {filename: f"def {function}() -> str:\n    {negative}\n"}
    prompts = (f"Read {module_path}. {constant} must remain the public source of truth.", f"Update {filename} so {function} {behavior}. Change only {filename} and {module_path}.")
    return _card(card_id, prompts, initial, (module_path, filename), checks, ModelTaskFamily.CONTEXT_CONTINUITY, split, ModelTaskSource.MANUAL_CONTRACT, source_reference, reference, negative_source)


def _card(
    card_id: str,
    prompts: tuple[str, ...],
    initial_files: dict[str, str],
    allowed_paths: tuple[str, ...],
    checks: tuple[BehaviorCheck, ...],
    family: ModelTaskFamily,
    split: ModelTaskSplit,
    source: ModelTaskSource,
    source_reference: str,
    reference_files: dict[str, str],
    negative_files: dict[str, str],
) -> ModelTaskCard:
    return ModelTaskCard(
        id=card_id,
        suite=MODEL_CODING_V2_BATCH_01_SUITE,
        kind=ModelTaskKind.PATCH,
        prompts=prompts,
        initial_files=initial_files,
        allowed_paths=allowed_paths,
        family=family,
        source=source,
        source_reference=source_reference,
        template_id=f"v2b1-{family.value}",
        split=split,
        acceptance_level=ModelTaskAcceptanceLevel.BEHAVIORAL,
        behavior_checks=checks,
        quality=ModelTaskQualityEvidence(
            reference_files=reference_files,
            negative_examples=(ModelTaskNegativeExample("semantic-wrong-outcome", files=negative_files, rationale="A plausible in-scope repair that violates the declared behavior or source-of-truth contract."),),
            reviewed_by="techpilot-maintainer",
            review_note="Reference and semantic negative are checked by the deterministic behavioral oracle before any model run.",
            review_state=ModelTaskReviewState.FROZEN if split is ModelTaskSplit.HOLDOUT else ModelTaskReviewState.REVIEWED,
        ),
    )
