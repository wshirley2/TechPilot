"""Unified Task Runtime identity contract tests."""

import pytest

from techpilot.runtime.contracts import (
    RuntimeResultStatus,
    TaskRuntimeIdentity,
    TaskRuntimePaths,
    TaskRuntimeResult,
)


def test_chat_runtime_identity_normalizes_paths(tmp_path):
    identity = TaskRuntimeIdentity(
        session_id=" session-1 ",
        working_directory=tmp_path,
    )

    assert identity.session_id == "session-1"
    assert identity.working_directory == tmp_path.resolve()
    assert identity.to_dict() == {
        "session_id": "session-1",
        "working_directory": str(tmp_path.resolve()),
    }


def test_chat_runtime_paths_keep_sessions_under_repository_by_default(tmp_path):
    paths = TaskRuntimePaths.for_runtime(tmp_path)

    assert paths.working_directory == tmp_path.resolve()
    assert paths.session_directory == tmp_path.resolve() / ".techpilot" / "sessions"
    assert paths.to_dict() == {
        "working_directory": str(tmp_path.resolve()),
        "session_directory": str(tmp_path.resolve() / ".techpilot" / "sessions"),
    }


def test_runtime_identity_rejects_blank_session_id(tmp_path):
    with pytest.raises(ValueError, match="non-empty session id"):
        TaskRuntimeIdentity(
            session_id=" ",
            working_directory=tmp_path,
        )


@pytest.mark.parametrize(
    ("event_type", "payload", "status"),
    [
        ("turn_completed", {"content": "done"}, RuntimeResultStatus.SUCCEEDED),
        ("turn_interrupted", {"reason": "cancelled"}, RuntimeResultStatus.CANCELLED),
        (
            "turn_failed",
            {"error_type": "RuntimeError", "error": "provider failed"},
            RuntimeResultStatus.FAILED,
        ),
        (
            "turn_limit_reached",
            {"limit": "provider_calls", "actual": 1, "maximum": 1},
            RuntimeResultStatus.LIMIT_REACHED,
        ),
    ],
)
def test_terminal_events_project_to_unified_turn_results(event_type, payload, status):
    result = TaskRuntimeResult.from_terminal_event(event_type, payload)

    assert result is not None
    assert result.status is status
    assert TaskRuntimeResult.from_dict(result.to_dict()) == result


def test_limit_result_requires_explicit_limit_facts():
    with pytest.raises(ValueError, match="requires a limit name"):
        TaskRuntimeResult(
            status=RuntimeResultStatus.LIMIT_REACHED,
            reason="stopped",
        )


def test_escalation_required_is_a_non_failure_runtime_result():
    result = TaskRuntimeResult(
        status=RuntimeResultStatus.ESCALATION_REQUIRED,
        response="需要隔离执行",
        reason="源仓库未执行",
    )

    assert TaskRuntimeResult.from_dict(result.to_dict()) == result
