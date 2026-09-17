"""First QA-gated expansion batch for the Coding Agent task deck.

These are small, self-contained snapshots.  They do not copy SWE-bench issues
or patches; public material informs only the issue-to-snapshot-to-oracle shape.
Every evaluator asset stays outside the model prompt and workspace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

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
    build_model_coding_v1_seed_qa_v1_cards,
    require_valid_model_task_deck,
)

MODEL_CODING_V1_BATCH_01_SUITE = "model-coding-v1-batch-01"
MODEL_CODING_V1_WORKING_40_SUITE = "model-coding-v1-working-40"


def build_model_coding_v1_batch_01_cards() -> tuple[ModelTaskCard, ...]:
    """Return 20 reviewed cards: 14 development, 4 frozen holdout, 2 regression."""

    return require_valid_model_task_deck((
        _card(
            id="coding-v1b1-read-timeout-precedence-021",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read config.py. What timeout is used when no value is supplied? Do not modify files.",),
            initial_files={
                "config.py": (
                    "DEFAULT_TIMEOUT_SECONDS = 30\n\n"
                    "def effective_timeout(value: int | None) -> int:\n"
                    "    return DEFAULT_TIMEOUT_SECONDS if value is None else value\n"
                ),
            },
            required_response_facts=("30",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: trace a visible default through its owner function",
            template_id="read-default-precedence",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_response="30 seconds",
            negative_response="60 seconds",
        ),
        _card(
            id="coding-v1b1-read-product-label-022",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Inspect constants.py and formatter.py. State the public product label. Do not modify files.",),
            initial_files={
                "constants.py": "PRODUCT_LABEL = 'TechPilot Local'\n",
                "formatter.py": "from constants import PRODUCT_LABEL\n\ndef header() -> str:\n    return PRODUCT_LABEL\n",
            },
            required_response_facts=("techpilot local",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: trace a public value across a local module boundary",
            template_id="read-cross-module-constant",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_response="The public product label is TechPilot Local.",
            negative_response="The public product label is Pilot.",
        ),
        _card(
            id="coding-v1b1-read-error-owner-023",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read tokens.py. Name the exception raised for an empty token. Do not modify files.",),
            initial_files={
                "tokens.py": (
                    "class EmptyTokenError(ValueError):\n"
                    "    pass\n\n"
                    "def require_token(token: str) -> str:\n"
                    "    if not token:\n"
                    "        raise EmptyTokenError('token is required')\n"
                    "    return token\n"
                ),
            },
            required_response_facts=("emptytokenerror",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: locate the owner of an explicit error path",
            template_id="read-behavior-owner",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_response="EmptyTokenError is raised.",
            negative_response="ValueError is raised directly.",
        ),
        _card(
            id="coding-v1b1-read-required-field-024",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read schema.py. Which field is required in every exported record? Do not modify files.",),
            initial_files={
                "schema.py": "REQUIRED_EXPORT_FIELDS = ('id', 'created_at')\n\ndef required_fields() -> tuple[str, ...]:\n    return REQUIRED_EXPORT_FIELDS\n",
            },
            required_response_facts=("created_at",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled schema fixture: distinguish a mandatory field from nearby optional prose",
            template_id="read-required-schema-field",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_response="created_at is required.",
            negative_response="display_name is required.",
        ),
        _card(
            id="coding-v1b1-repair-none-default-025",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix defaults.py. resolve_label(None) must return 'auto', while an empty string must remain an empty string. Change only defaults.py.",),
            initial_files={"defaults.py": "def resolve_label(value: str | None) -> str:\n    return value or 'auto'\n"},
            allowed_paths=("defaults.py",),
            required_file_contents={"defaults.py": "return 'auto' if value is None else value"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled falsy-versus-none default regression",
            template_id="single-file-none-default-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"defaults.py": "def resolve_label(value: str | None) -> str:\n    return 'auto' if value is None else value\n"},
            negative_files={"defaults.py": "def resolve_label(value: str | None) -> str:\n    return value or 'safe'\n"},
        ),
        _card(
            id="coding-v1b1-repair-url-join-026",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix urls.py. join_url must produce exactly one slash between base and path. Change only urls.py and keep the signature unchanged.",),
            initial_files={"urls.py": "def join_url(base: str, path: str) -> str:\n    return f'{base}/{path}'\n"},
            allowed_paths=("urls.py",),
            required_file_contents={"urls.py": "return f\"{base.rstrip('/')}/{path.lstrip('/')}\""},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: normalize a boundary between two caller inputs",
            template_id="single-file-url-boundary-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"urls.py": "def join_url(base: str, path: str) -> str:\n    return f\"{base.rstrip('/')}/{path.lstrip('/')}\"\n"},
            negative_files={"urls.py": "def join_url(base: str, path: str) -> str:\n    return f'{base.rstrip('/')}/{path}'\n"},
        ),
        _card(
            id="coding-v1b1-repair-port-range-027",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix ports.py. is_valid_port must accept ports 1 through 65535 inclusive. Change only ports.py.",),
            initial_files={"ports.py": "def is_valid_port(port: int) -> bool:\n    return 0 < port < 65535\n"},
            allowed_paths=("ports.py",),
            required_file_contents={"ports.py": "return 1 <= port <= 65535"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled upper-bound exclusivity injection",
            template_id="single-file-range-boundary-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"ports.py": "def is_valid_port(port: int) -> bool:\n    return 1 <= port <= 65535\n"},
            negative_files={"ports.py": "def is_valid_port(port: int) -> bool:\n    return 1 <= port < 65535\n"},
        ),
        _card(
            id="coding-v1b1-repair-tag-normalize-028",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix tags.py. canonical_tag must trim whitespace, use lowercase, and replace internal spaces with hyphens. Change only tags.py.",),
            initial_files={"tags.py": "def canonical_tag(value: str) -> str:\n    return value\n"},
            allowed_paths=("tags.py",),
            required_file_contents={"tags.py": "return value.strip().lower().replace(' ', '-')"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled multi-step text normalization injection",
            template_id="single-file-composed-normalization-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"tags.py": "def canonical_tag(value: str) -> str:\n    return value.strip().lower().replace(' ', '-')\n"},
            negative_files={"tags.py": "def canonical_tag(value: str) -> str:\n    return value.strip().lower()\n"},
        ),
        _card(
            id="coding-v1b1-repair-record-field-029",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix records.py. to_record must return id, name, and active from the input mapping. Change only records.py.",),
            initial_files={"records.py": "def to_record(user: dict[str, object]) -> dict[str, object]:\n    return {'id': user['id'], 'name': user['name']}\n"},
            allowed_paths=("records.py",),
            required_file_contents={
                "records.py#id": "'id': user['id']",
                "records.py#name": "'name': user['name']",
                "records.py#active": "'active': user['active']",
            },
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: preserve all declared payload fields",
            template_id="single-file-required-field-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"records.py": "def to_record(user: dict[str, object]) -> dict[str, object]:\n    return {'id': user['id'], 'name': user['name'], 'active': user['active']}\n"},
            negative_files={"records.py": "def to_record(user: dict[str, object]) -> dict[str, object]:\n    return {'id': user['id'], 'active': user['active']}\n"},
        ),
        _card(
            id="coding-v1b1-repair-closed-range-030",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix ranges.py. contains(value, start, end) must include both endpoints. Change only ranges.py.",),
            initial_files={"ranges.py": "def contains(value: int, start: int, end: int) -> bool:\n    return start < value < end\n"},
            allowed_paths=("ranges.py",),
            required_file_contents={"ranges.py": "return start <= value <= end"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled two-sided inclusive-boundary injection",
            template_id="single-file-closed-range-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"ranges.py": "def contains(value: int, start: int, end: int) -> bool:\n    return start <= value <= end\n"},
            negative_files={"ranges.py": "def contains(value: int, start: int, end: int) -> bool:\n    return start <= value < end\n"},
        ),
        _card(
            id="coding-v1b1-repair-stable-labels-031",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix labels.py. unique_labels must remove duplicates while retaining the first occurrence order. Change only labels.py.",),
            initial_files={"labels.py": "def unique_labels(labels: list[str]) -> list[str]:\n    return sorted(set(labels))\n"},
            allowed_paths=("labels.py",),
            required_file_contents={"labels.py": "return list(dict.fromkeys(labels))"},
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: preserve caller-observable order while removing duplicates",
            template_id="single-file-stable-dedup-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"labels.py": "def unique_labels(labels: list[str]) -> list[str]:\n    return list(dict.fromkeys(labels))\n"},
            negative_files={"labels.py": "def unique_labels(labels: list[str]) -> list[str]:\n    return sorted(set(labels), reverse=True)\n"},
        ),
        _card(
            id="coding-v1b1-cross-turn-delimiter-032",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Inspect rows.py. Keep render_row's name and parameters unchanged.",
                "Now make render_row use ' | ' between cells. Change only rows.py and retain the earlier interface constraint.",
            ),
            initial_files={"rows.py": "def render_row(cells: list[str], delimiter: str = ', ') -> str:\n    return delimiter.join(cells)\n"},
            allowed_paths=("rows.py",),
            required_file_contents={
                "rows.py#signature": "def render_row(cells: list[str], delimiter: str = ' | ') -> str:",
                "rows.py#body": "return delimiter.join(cells)",
            },
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: retain the first-turn function interface while changing a later default",
            template_id="cross-turn-default-preservation",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"rows.py": "def render_row(cells: list[str], delimiter: str = ' | ') -> str:\n    return delimiter.join(cells)\n"},
            negative_files={"rows.py": "def render_row(cells: list[str]) -> str:\n    return ' | '.join(cells)\n"},
        ),
        _card(
            id="coding-v1b1-cross-turn-mode-033",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Inspect modes.py. The public signature of selected_mode must stay unchanged.",
                "Change the fallback from 'preview' to 'standard'. Change only modes.py and retain the earlier signature constraint.",
            ),
            initial_files={"modes.py": "def selected_mode(value: str | None) -> str:\n    return value or 'preview'\n"},
            allowed_paths=("modes.py",),
            required_file_contents={
                "modes.py#signature": "def selected_mode(value: str | None) -> str:",
                "modes.py#fallback": "return value or 'standard'",
            },
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled fallback change with retained cross-turn API contract",
            template_id="cross-turn-interface-preservation",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"modes.py": "def selected_mode(value: str | None) -> str:\n    return value or 'standard'\n"},
            negative_files={"modes.py": "def selected_mode() -> str:\n    return 'standard'\n"},
        ),
        _card(
            id="coding-v1b1-context-product-name-034",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Read constants.py. The public product name must remain the single source of truth.",
                "Update greeting.py to use that public product name. Change only greeting.py and constants.py.",
            ),
            initial_files={
                "constants.py": "PRODUCT_NAME = 'TechPilot'\n",
                "greeting.py": "def greeting(name: str) -> str:\n    return f'Hello {name} from Pilot'\n",
            },
            allowed_paths=("constants.py", "greeting.py"),
            required_file_contents={
                "greeting.py#import": "from constants import PRODUCT_NAME",
                "greeting.py#usage": "return f'Hello {name} from {PRODUCT_NAME}'",
            },
            family=ModelTaskFamily.CONTEXT_CONTINUITY,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: retain a first-turn source-of-truth constraint in a later multi-file repair",
            template_id="cross-turn-source-of-truth-repair",
            split=ModelTaskSplit.DEVELOPMENT,
            reference_files={"greeting.py": "from constants import PRODUCT_NAME\n\ndef greeting(name: str) -> str:\n    return f'Hello {name} from {PRODUCT_NAME}'\n"},
            negative_files={"greeting.py": "def greeting(name: str) -> str:\n    return f'Hello {name} from TechPilot'\n"},
        ),
        _card(
            id="coding-v1b1-holdout-error-code-035",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix presenter.py so format_error imports and displays the public CONNECTION_ERROR_CODE from errors.py. Change only presenter.py and errors.py.",),
            initial_files={
                "errors.py": "CONNECTION_ERROR_CODE = 'E_CONN'\n",
                "presenter.py": "def format_error() -> str:\n    return 'connection failed'\n",
            },
            allowed_paths=("errors.py", "presenter.py"),
            required_file_contents={
                "presenter.py#import": "from errors import CONNECTION_ERROR_CODE",
                "presenter.py#usage": "return f'{CONNECTION_ERROR_CODE}: connection failed'",
            },
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.PUBLIC_PATTERN_ADAPTED,
            source_reference="SWE-bench issue-resolution pattern: use a public cross-module constant instead of duplicating a literal",
            template_id="two-file-constant-consistency",
            split=ModelTaskSplit.HOLDOUT,
            reference_files={"presenter.py": "from errors import CONNECTION_ERROR_CODE\n\ndef format_error() -> str:\n    return f'{CONNECTION_ERROR_CODE}: connection failed'\n"},
            negative_files={"presenter.py": "def format_error() -> str:\n    return 'E_CONN: connection failed'\n"},
        ),
        _card(
            id="coding-v1b1-holdout-cross-turn-prefix-036",
            kind=ModelTaskKind.PATCH,
            prompts=(
                "Inspect notices.py. Keep build_notice's function name and parameters unchanged.",
                "Make the default prefix 'Notice'. Change only notices.py and retain the earlier interface constraint.",
            ),
            initial_files={"notices.py": "def build_notice(message: str, prefix: str = 'Info') -> str:\n    return f'{prefix}: {message}'\n"},
            allowed_paths=("notices.py",),
            required_file_contents={
                "notices.py#signature": "def build_notice(message: str, prefix: str = 'Notice') -> str:",
                "notices.py#body": "return f'{prefix}: {message}'",
            },
            family=ModelTaskFamily.MULTI_TURN_CONSTRAINT,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: frozen multi-turn interface preservation",
            template_id="cross-turn-default-preservation",
            split=ModelTaskSplit.HOLDOUT,
            reference_files={"notices.py": "def build_notice(message: str, prefix: str = 'Notice') -> str:\n    return f'{prefix}: {message}'\n"},
            negative_files={"notices.py": "def build_notice(message: str) -> str:\n    return f'Notice: {message}'\n"},
        ),
        _card(
            id="coding-v1b1-holdout-scope-037",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix readiness.py so current_readiness returns 'ready'. Change only readiness.py; do not modify README.md.",),
            initial_files={
                "readiness.py": "def current_readiness() -> str:\n    return 'draft'\n",
                "README.md": "Readiness is documented here.\n",
            },
            allowed_paths=("readiness.py",),
            required_file_contents={"readiness.py": "return 'ready'"},
            family=ModelTaskFamily.PERMISSION_BOUNDARY,
            source=ModelTaskSource.MANUAL_CONTRACT,
            source_reference="manual contract: frozen narrow-write repair with adjacent documentation",
            template_id="scoped-constant-repair",
            split=ModelTaskSplit.HOLDOUT,
            reference_files={"readiness.py": "def current_readiness() -> str:\n    return 'ready'\n"},
            negative_files={"readiness.py": "def current_readiness() -> str:\n    return 'almost-ready'\n"},
        ),
        _card(
            id="coding-v1b1-holdout-read-source-038",
            kind=ModelTaskKind.READ_ONLY,
            prompts=("Read identity.py. State the identifier prefix used for service accounts. Do not modify files.",),
            initial_files={"identity.py": "SERVICE_ACCOUNT_PREFIX = 'svc_'\n\ndef service_id(name: str) -> str:\n    return f'{SERVICE_ACCOUNT_PREFIX}{name}'\n"},
            required_response_facts=("svc_",),
            family=ModelTaskFamily.CODE_READING,
            source=ModelTaskSource.FAULT_INJECTION,
            source_reference="controlled frozen identifier-prefix fixture",
            template_id="read-configuration-fact",
            split=ModelTaskSplit.HOLDOUT,
            reference_response="The prefix is svc_.",
            negative_response="The prefix is service_.",
        ),
        _card(
            id="coding-v1b1-regression-serializer-039",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix exports.py. export_user must return id, name, and email. Change only exports.py.",),
            initial_files={"exports.py": "def export_user(user: dict[str, str]) -> dict[str, str]:\n    return {'id': user['id'], 'name': user['name']}\n"},
            allowed_paths=("exports.py",),
            required_file_contents={
                "exports.py#id": "'id': user['id']",
                "exports.py#name": "'name': user['name']",
                "exports.py#email": "'email': user['email']",
            },
            family=ModelTaskFamily.SCOPED_REPAIR,
            source=ModelTaskSource.OBSERVED_REGRESSION,
            source_reference="model-coding-v1-seed serializer oracle audit: require every promised field",
            template_id="single-file-complete-serialization-repair",
            split=ModelTaskSplit.REGRESSION,
            reference_files={"exports.py": "def export_user(user: dict[str, str]) -> dict[str, str]:\n    return {'id': user['id'], 'name': user['name'], 'email': user['email']}\n"},
            negative_files={"exports.py": "def export_user(user: dict[str, str]) -> dict[str, str]:\n    return {'id': user['id'], 'email': user['email']}\n"},
        ),
        _card(
            id="coding-v1b1-regression-write-scope-040",
            kind=ModelTaskKind.PATCH,
            prompts=("Fix state.py so state_name returns 'active'. Change only state.py; do not edit docs.md.",),
            initial_files={
                "state.py": "def state_name() -> str:\n    return 'inactive'\n",
                "docs.md": "State names are listed here.\n",
            },
            allowed_paths=("state.py",),
            required_file_contents={"state.py": "return 'active'"},
            family=ModelTaskFamily.PERMISSION_BOUNDARY,
            source=ModelTaskSource.OBSERVED_REGRESSION,
            source_reference="model task QA scope-probe regression guard",
            template_id="scoped-constant-repair",
            split=ModelTaskSplit.REGRESSION,
            reference_files={"state.py": "def state_name() -> str:\n    return 'active'\n"},
            negative_files={"state.py": "def state_name() -> str:\n    return 'enabled'\n"},
        ),
    ))


def build_model_coding_v1_working_40_cards() -> tuple[ModelTaskCard, ...]:
    """Combine the QA seed and batch 01 under one unexecuted, versioned work deck."""

    cards = apply_model_coding_v1_static_dispositions(
        _with_behavior_checks(build_model_coding_v1_seed_qa_v1_cards() + build_model_coding_v1_batch_01_cards()),
    )
    return require_valid_model_task_deck(
        tuple(replace(card, suite=MODEL_CODING_V1_WORKING_40_SUITE) for card in cards),
    )


def _with_behavior_checks(cards: tuple[ModelTaskCard, ...]) -> tuple[ModelTaskCard, ...]:
    checks = {
        "coding-v1-cross-turn-tax-018": (BehaviorCheck("surcharge", "tax.py", "total", (10,), 11), BehaviorCheck("zero", "tax.py", "total", (0,), 1)),
        "coding-v1-regression-normalize-019": (BehaviorCheck("spaces", "name.py", "canonical_name", (" Ready ",), "ready"), BehaviorCheck("plain", "name.py", "canonical_name", ("ok",), "ok")),
        "coding-v1b1-repair-none-default-025": (BehaviorCheck("none", "defaults.py", "resolve_label", (None,), "auto"), BehaviorCheck("empty", "defaults.py", "resolve_label", ("",), "")),
        "coding-v1b1-repair-tag-normalize-028": (BehaviorCheck("spaces", "tags.py", "canonical_tag", ("  Ready Set ",), "ready-set"), BehaviorCheck("plain", "tags.py", "canonical_tag", ("ok",), "ok")),
        "coding-v1b1-repair-port-range-027": (BehaviorCheck("lowest", "ports.py", "is_valid_port", (1,), True), BehaviorCheck("highest", "ports.py", "is_valid_port", (65535,), True), BehaviorCheck("below", "ports.py", "is_valid_port", (0,), False)),
        "coding-v1b1-repair-closed-range-030": (BehaviorCheck("start", "ranges.py", "contains", (1, 1, 3), True), BehaviorCheck("end", "ranges.py", "contains", (3, 1, 3), True), BehaviorCheck("below", "ranges.py", "contains", (0, 1, 3), False)),
        "coding-v1b1-cross-turn-mode-033": (BehaviorCheck("missing", "modes.py", "selected_mode", (None,), "standard"), BehaviorCheck("explicit", "modes.py", "selected_mode", ("custom",), "custom")),
        "coding-v1b1-cross-turn-delimiter-032": (BehaviorCheck("default", "rows.py", "render_row", (["a", "b"],), "a | b"), BehaviorCheck("custom", "rows.py", "render_row", (["a", "b"], ": "), "a: b")),
        "coding-v1b1-regression-serializer-039": (BehaviorCheck("first-user", "exports.py", "export_user", ({"id": "u-1", "name": "Ada", "email": "ada@example.test"},), {"id": "u-1", "name": "Ada", "email": "ada@example.test"}), BehaviorCheck("second-user", "exports.py", "export_user", ({"id": "u-2", "name": "Bo", "email": "bo@example.test"},), {"id": "u-2", "name": "Bo", "email": "bo@example.test"})),
        "coding-v1b1-context-product-name-034": (BehaviorCheck("ada", "greeting.py", "greeting", ("Ada",), "Hello Ada from TechPilot", constant_imports=(("constants", "PRODUCT_NAME"),)), BehaviorCheck("bo", "greeting.py", "greeting", ("Bo",), "Hello Bo from TechPilot", constant_imports=(("constants", "PRODUCT_NAME"),))),
        "coding-v1b1-holdout-cross-turn-prefix-036": (BehaviorCheck("default", "notices.py", "build_notice", ("ok",), "Notice: ok"), BehaviorCheck("custom", "notices.py", "build_notice", ("ok", "Alert"), "Alert: ok")),
    }
    return tuple(replace(card, acceptance_level=ModelTaskAcceptanceLevel.BEHAVIORAL, behavior_checks=checks[card.id], required_file_contents={}) if card.id in checks else card for card in cards)


def _card(
    *,
    id: str,
    kind: ModelTaskKind,
    prompts: tuple[str, ...],
    initial_files: Mapping[str, str],
    required_response_facts: tuple[str, ...] = (),
    allowed_paths: tuple[str, ...] = (),
    required_file_contents: Mapping[str, str] | None = None,
    family: ModelTaskFamily,
    source: ModelTaskSource,
    source_reference: str,
    template_id: str,
    split: ModelTaskSplit,
    reference_files: Mapping[str, str] | None = None,
    reference_response: str = "",
    negative_files: Mapping[str, str] | None = None,
    negative_response: str = "",
) -> ModelTaskCard:
    return ModelTaskCard(
        id=id,
        suite=MODEL_CODING_V1_BATCH_01_SUITE,
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
        quality=ModelTaskQualityEvidence(
            reference_files=reference_files or {},
            reference_response=reference_response,
            negative_examples=(
                ModelTaskNegativeExample(
                    id="semantic-wrong-outcome",
                    files=negative_files or {},
                    response=negative_response,
                    rationale="A plausible answer or in-scope repair that misses the declared acceptance rule.",
                ),
            ),
            reviewed_by="techpilot-maintainer",
            review_note="Reference and semantic negative checked against the independent local oracle.",
            review_state=(ModelTaskReviewState.FROZEN if split is ModelTaskSplit.HOLDOUT else ModelTaskReviewState.REVIEWED),
        ),
    )
