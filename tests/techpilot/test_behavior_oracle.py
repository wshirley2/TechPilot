from __future__ import annotations

import pytest

from techpilot.evaluation.behavior_oracle import (
    BEHAVIOR_ORACLE_VERSION,
    BehaviorCheck,
    behavior_oracle_identity,
    evaluate_behavior_check,
)
from techpilot.evaluation.model_tasks import ModelTaskAcceptanceLevel, ModelTaskCard, ModelTaskKind, _score


def test_behavior_oracle_interprets_only_a_pure_allowlisted_expression() -> None:
    check = BehaviorCheck("none-default", "defaults.py", "resolve", (None,), "auto")
    source = "def resolve(value):\n    return 'auto' if value is None else value\n"

    assert evaluate_behavior_check(source, check)
    assert not evaluate_behavior_check(source, BehaviorCheck("empty", "defaults.py", "resolve", ("",), "auto"))


def test_behavior_oracle_identity_is_versioned_and_content_addressed() -> None:
    identity = behavior_oracle_identity()

    assert identity["version"] == BEHAVIOR_ORACLE_VERSION
    assert len(identity["source_sha256"]) == 64
    assert set(identity["source_sha256"]) <= set("0123456789abcdef")


def test_behavior_oracle_returns_the_actual_short_circuit_value() -> None:
    source = "def resolve(value):\n    return value or 'ready'\n"

    assert evaluate_behavior_check(source, BehaviorCheck("missing", "state.py", "resolve", (None,), "ready"))
    assert evaluate_behavior_check(source, BehaviorCheck("explicit", "state.py", "resolve", ("custom",), "custom"))


def test_behavior_oracle_rejects_code_outside_its_ast_subset_without_execution() -> None:
    check = BehaviorCheck("unsafe", "unsafe.py", "resolve", ("x",), "x")

    assert not evaluate_behavior_check("import os\ndef resolve(value):\n    return os.getcwd()\n", check)
    assert not evaluate_behavior_check("def resolve(value):\n    value = os.getcwd()\n    return value\n", check)
    assert not evaluate_behavior_check("def resolve(value):\n    return value.encode()\n", check)


def test_behavior_oracle_interprets_chain_comparisons_without_executing_code() -> None:
    check = BehaviorCheck("upper-bound", "ports.py", "is_valid_port", (65535,), True)
    source = "def is_valid_port(port):\n    return 1 <= port <= 65535\n"

    assert evaluate_behavior_check(source, check)
    assert not evaluate_behavior_check(source, BehaviorCheck("above", "ports.py", "is_valid_port", (65536,), True))


def test_behavior_oracle_interprets_only_two_numeric_min_max_arguments() -> None:
    check = BehaviorCheck("lower-bound", "limits.py", "clamp", (-1,), 0)
    source = "def clamp(value):\n    return max(0, min(value, 100))\n"

    assert evaluate_behavior_check(source, check)
    assert not evaluate_behavior_check("def clamp(value):\n    return max(value, 0, 100)\n", check)
    assert not evaluate_behavior_check("def clamp(value):\n    return sorted((value, 0))[0]\n", check)


def test_behavior_oracle_interprets_join_and_simple_scalar_f_strings_only() -> None:
    assert evaluate_behavior_check("def summary(items):\n    return ' / '.join(items)\n", BehaviorCheck("join", "summary.py", "summary", (["a", "b"],), "a / b"))
    assert evaluate_behavior_check("def label(value):\n    return f'value={value}'\n", BehaviorCheck("label", "label.py", "label", ("ok",), "value=ok"))
    assert evaluate_behavior_check("def label(value):\n    return f'value={value}'\n", BehaviorCheck("number", "label.py", "label", (2,), "value=2"))
    assert not evaluate_behavior_check("def label(value):\n    return f'{value!r}'\n", BehaviorCheck("repr", "label.py", "label", ("ok",), "'ok'"))


def test_behavior_oracle_interprets_only_numeric_scalar_addition() -> None:
    check = BehaviorCheck("surcharge", "tax.py", "total", (10,), 11)

    assert evaluate_behavior_check("def total(amount):\n    return amount + 1\n", check)
    assert not evaluate_behavior_check("def total(amount):\n    return amount + '1'\n", check)


def test_behavior_oracle_interprets_only_scalar_dictionary_literals_and_subscripts() -> None:
    source = "def public(user):\n    return {'id': user['id'], 'name': user['name']}\n"
    user = {"id": "u-1", "name": "Ada"}

    assert evaluate_behavior_check(source, BehaviorCheck("all-fields", "payload.py", "public", (user,), user))
    assert not evaluate_behavior_check(source, BehaviorCheck("wrong-name", "payload.py", "public", (user,), {"id": "u-1", "name": "Bo"}))
    assert not evaluate_behavior_check("def public(user):\n    return user['id'].encode()\n", BehaviorCheck("method", "payload.py", "public", (user,), "u-1"))


def test_behavior_oracle_interprets_declared_scalar_constant_imports_without_importing_code() -> None:
    check = BehaviorCheck(
        "source-of-truth",
        "greeting.py",
        "greeting",
        ("Ada",),
        "Hello Ada from TechPilot",
        constant_imports=(("constants", "PRODUCT_NAME"),),
    )
    source = "from constants import PRODUCT_NAME\n\ndef greeting(name):\n    return f'Hello {name} from {PRODUCT_NAME}'\n"
    workspace = {"constants.py": "PRODUCT_NAME = 'TechPilot'\n"}

    assert evaluate_behavior_check(source, check, workspace_sources=workspace)
    assert not evaluate_behavior_check("def greeting(name):\n    return f'Hello {name} from TechPilot'\n", check, workspace_sources=workspace)
    assert not evaluate_behavior_check(source, check, workspace_sources={"constants.py": "PRODUCT_NAME = os.getenv('PRODUCT_NAME')\n"})


def test_behavior_oracle_interprets_safe_if_branches_and_no_argument_split() -> None:
    assert evaluate_behavior_check(
        "def resolve(value):\n    if value is None:\n        return 'auto'\n    return value\n",
        BehaviorCheck("none", "defaults.py", "resolve", (None,), "auto"),
    )
    assert evaluate_behavior_check(
        "def canonical(value):\n    return '-'.join(value.strip().lower().split())\n",
        BehaviorCheck("spaces", "tags.py", "canonical", ("  Ready Set ",), "ready-set"),
    )
    assert evaluate_behavior_check(
        "def resolve(value):\n    if value is None:\n        value = 'auto'\n    return value\n",
        BehaviorCheck("assignment", "defaults.py", "resolve", (None,), "auto"),
    )


def test_behavior_oracle_interprets_safe_scalar_constants_slices_and_type_guards() -> None:
    source = """_PREFIX = 'tp_'\n\ndef clean(value):\n    \"\"\"Remove at most one transport prefix.\"\"\"\n    prefix = _PREFIX\n    if isinstance(value, str) and value.startswith(prefix):\n        return str(value)[len(prefix):]\n    return value\n"""

    assert evaluate_behavior_check(source, BehaviorCheck("one", "keys.py", "clean", ("tp_alpha",), "alpha"))
    assert evaluate_behavior_check(source, BehaviorCheck("two", "keys.py", "clean", ("tp_tp_alpha",), "tp_alpha"))
    assert evaluate_behavior_check(source, BehaviorCheck("plain", "keys.py", "clean", ("alpha",), "alpha"))
    assert not evaluate_behavior_check(
        "value = open('secret')\n\ndef clean(value):\n    return value\n",
        BehaviorCheck("unsafe-constant", "keys.py", "clean", ("alpha",), "alpha"),
    )


def test_behavior_oracle_interprets_a_scalar_bool_conversion() -> None:
    source = "def response(value):\n    return {'retryable': bool(value.get('retryable', False))}\n"

    assert evaluate_behavior_check(source, BehaviorCheck("true", "response.py", "response", ({"retryable": True},), {"retryable": True}))
    assert evaluate_behavior_check(source, BehaviorCheck("false", "response.py", "response", ({"retryable": False},), {"retryable": False}))


def test_behavioral_card_scores_behavior_and_scope_without_exact_reference_source_text() -> None:
    check = BehaviorCheck("none", "defaults.py", "resolve", (None,), "auto")
    card = ModelTaskCard(
        id="behavior-card", suite="behavior-v0", kind=ModelTaskKind.PATCH,
        prompts=("Fix defaults.py.",), initial_files={"defaults.py": "def resolve(value):\n    return value\n"},
        allowed_paths=("defaults.py",),
        acceptance_level=ModelTaskAcceptanceLevel.BEHAVIORAL, behavior_checks=(check,),
    )
    before = {"defaults.py": b"def resolve(value):\n    return value\n"}
    after = {"defaults.py": b"def resolve(value):\n    if value is None:\n        return 'auto'\n    return value\n"}

    assert all(_score(card, before, after, "").values())

    with pytest.raises(ValueError, match="exact source text"):
        ModelTaskCard(
            id="invalid-behavior-card", suite="behavior-v0", kind=ModelTaskKind.PATCH,
            prompts=("Fix defaults.py.",), initial_files={"defaults.py": "def resolve(value):\n    return value\n"},
            allowed_paths=("defaults.py",), required_file_contents={"defaults.py": "return value"},
            acceptance_level=ModelTaskAcceptanceLevel.BEHAVIORAL, behavior_checks=(check,),
        )
