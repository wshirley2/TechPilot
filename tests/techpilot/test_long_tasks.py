"""Deterministic RS-6.1 safety tests for long Task persistence and recovery."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from techpilot.runtime.long_tasks import (
    EffectDisposition,
    LongTaskBudget,
    LongTaskEvent,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskStore,
)
from techpilot.runtime.sessions import SessionEvent, SessionStore


def _create(store: LongTaskStore, root: Path, *, budget: LongTaskBudget | None = None):
    SessionStore.for_repository(root).create("chat-session", repository_root=root, model="fake-model")
    return store.create(
        "repair-task",
        goal="Repair the deterministic fixture and validate it.",
        repository_root=root,
        session_id="chat-session",
        budget=budget,
    )


def _session_cursor(root: Path, session_id: str = "chat-session") -> str:
    return SessionStore.for_repository(root).replay(session_id).events[-1].event_id


def _start_effect(store: LongTaskStore, action_id: str):
    with store.acquire_lease("repair-task", owner_id="test-runner") as lease:
        return store.start_effect("repair-task", action_id, lease=lease)


def test_checkpoint_indexes_task_facts_without_overwriting_raw_session_facts(tmp_path):
    sessions = SessionStore(tmp_path / ".techpilot" / "sessions")
    sessions.create("chat-session", repository_root=tmp_path, model="fake-model")
    sessions.append(SessionEvent("user_fact", "chat-session", {"marker": "raw-session-fact"}))
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="inspect", kind="agent_turn")

    checkpoint = store.checkpoint(
        "repair-task",
        checkpoint_id="before-inspect",
        message_projection=[{"role": "system", "content": "[summary] raw-session-fact"}],
        recovery_reason="before deterministic inspection",
        session_event_cursor=_session_cursor(tmp_path),
    )

    projection = store.replay("repair-task")
    assert checkpoint.event_cursor == projection.events[-2].event_id
    assert checkpoint.session_event_cursor == _session_cursor(tmp_path)
    assert checkpoint.pending_action_ids == ("inspect",)
    assert sessions.replay("chat-session").events[-1].payload == {"marker": "raw-session-fact"}
    assert store.path_for("repair-task") != sessions.path_for("chat-session")


def test_resume_skips_a_durably_completed_effect_without_calling_it_again(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="write-fix", kind="tool_effect", effect_id="write-fix-v1")
    assert store.effect_disposition("repair-task", "write-fix") is EffectDisposition.EXECUTE
    _start_effect(store, "write-fix")
    store.complete_effect("repair-task", "write-fix", result="patch written")
    store.checkpoint(
        "repair-task",
        message_projection=[],
        recovery_reason="after patch",
        session_event_cursor=_session_cursor(tmp_path),
    )
    store.pause("repair-task", reason="simulated restart")

    recovered = LongTaskStore.for_repository(tmp_path).resume("repair-task", reason="restart recovered")

    assert recovered.status is LongTaskStatus.RUNNING
    assert recovered.completed_effect_ids == ("write-fix-v1",)
    assert store.effect_disposition("repair-task", "write-fix") is EffectDisposition.SKIP_SUCCEEDED
    with pytest.raises(LongTaskStateError, match="skip_succeeded"):
        _start_effect(store, "write-fix")


def test_interruption_after_effect_start_requires_reconciliation_never_auto_retries(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="execute-migration", kind="tool_effect", effect_id="migration-v1")
    _start_effect(store, "execute-migration")
    externally_observed_calls = ["migration-v1"]  # The actual external effect happened before process loss.

    recovered = LongTaskStore.for_repository(tmp_path).resume("repair-task", reason="restart recovered")

    assert recovered.status is LongTaskStatus.RECOVERY_REQUIRED
    assert recovered.ambiguous_effect_ids == ("migration-v1",)
    assert store.effect_disposition("repair-task", "execute-migration") is EffectDisposition.RECONCILE_REQUIRED
    assert externally_observed_calls == ["migration-v1"]
    with pytest.raises(LongTaskStateError, match="reconcile_required"):
        _start_effect(store, "execute-migration")


def test_explicitly_rejected_effect_is_not_recorded_as_completed(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="write-fix", kind="tool_effect", effect_id="write-fix-v1")
    _start_effect(store, "write-fix")

    projection = store.fail_effect("repair-task", "write-fix", reason="Permission denied write_file: user rejected")

    assert projection.completed_effect_ids == ()
    assert projection.actions["write-fix"].status.value == "failed"
    assert store.effect_disposition("repair-task", "write-fix") is EffectDisposition.BLOCKED_FAILED


def test_restart_recovers_a_running_task_when_no_pause_event_was_durable(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="inspect", kind="agent_turn")

    recovered = LongTaskStore.for_repository(tmp_path).resume("repair-task", reason="process restarted")

    assert recovered.status is LongTaskStatus.RUNNING
    assert recovered.actions["inspect"].status.value == "pending"


def test_completed_agent_turn_unblocks_a_dependent_side_effect(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="inspect", kind="agent_turn")
    store.complete_action("repair-task", "inspect", result="failure located")
    store.plan_action(
        "repair-task",
        action_id="write-fix",
        kind="tool_effect",
        effect_id="write-fix-v1",
        dependencies=("inspect",),
    )

    assert store.effect_disposition("repair-task", "write-fix") is EffectDisposition.EXECUTE


def test_task_success_is_terminal_and_blocks_later_effects(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="inspect", kind="agent_turn")
    store.complete_action("repair-task", "inspect")
    completed = store.succeed("repair-task", result="validated")

    assert completed.status is LongTaskStatus.SUCCEEDED
    with pytest.raises(LongTaskStateError, match="not running"):
        store.plan_action("repair-task", action_id="later", kind="tool_effect", effect_id="later-v1")
    with pytest.raises(LongTaskStateError, match="interrupted Tasks"):
        store.resume("repair-task", reason="must not resume")


def test_cancelled_task_never_starts_a_later_side_effect(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="write-fix", kind="tool_effect", effect_id="write-fix-v1")
    store.cancel("repair-task", reason="user cancelled")

    assert store.effect_disposition("repair-task", "write-fix") is EffectDisposition.BLOCKED_CANCELLED
    with pytest.raises(LongTaskStateError, match="terminal Task"):
        _start_effect(store, "write-fix")


def test_effect_budget_stops_before_the_next_side_effect_and_survives_replay(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path, budget=LongTaskBudget(max_effects=1))
    store.plan_action("repair-task", action_id="write-fix", kind="tool_effect", effect_id="write-fix-v1")
    _start_effect(store, "write-fix")
    store.complete_effect("repair-task", "write-fix")
    store.plan_action("repair-task", action_id="run-command", kind="tool_effect", effect_id="test-v1", dependencies=("write-fix",))

    assert store.effect_disposition("repair-task", "run-command") is EffectDisposition.BLOCKED_LIMIT
    recovered = LongTaskStore.for_repository(tmp_path).replay("repair-task")
    assert recovered.status is LongTaskStatus.LIMIT_REACHED
    assert recovered.effects_used == 1


def test_effect_start_requires_a_live_lease_and_rejects_a_second_process(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="write-fix", kind="tool_effect", effect_id="write-fix-v1")
    with pytest.raises(TypeError):
        store.start_effect("repair-task", "write-fix")  # type: ignore[call-arg]

    with store.acquire_lease("repair-task", owner_id="primary") as lease:
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from pathlib import Path\n"
                    "from techpilot.runtime.long_tasks import LongTaskLeaseError, LongTaskStore\n"
                    "store = LongTaskStore.for_repository(Path(sys.argv[1]))\n"
                    "try:\n"
                    "    store.acquire_lease('repair-task')\n"
                    "except LongTaskLeaseError:\n"
                    "    sys.exit(0)\n"
                    "sys.exit(1)\n"
                ),
                str(tmp_path),
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        store.start_effect("repair-task", "write-fix", lease=lease)

    with LongTaskStore.for_repository(tmp_path).acquire_lease("repair-task", owner_id="after-release") as later:
        assert later.active


def test_task_lease_is_released_when_the_owner_process_exits(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from pathlib import Path\n"
                "from techpilot.runtime.long_tasks import LongTaskStore\n"
                "lease = LongTaskStore.for_repository(Path(sys.argv[1])).acquire_lease('repair-task')\n"
                "sys.exit(0)\n"
            ),
            str(tmp_path),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr

    with store.acquire_lease("repair-task", owner_id="recovered-owner") as recovered:
        assert recovered.active


def test_checkpoint_binds_to_a_real_event_in_its_own_session(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    cursor = _session_cursor(tmp_path)

    checkpoint = store.checkpoint(
        "repair-task",
        message_projection=[{"role": "user", "content": "repair fixture"}],
        recovery_reason="before tool planning",
        session_event_cursor=cursor,
    )

    assert checkpoint.session_event_cursor == cursor
    assert any(event.event_id == cursor for event in SessionStore.for_repository(tmp_path).replay("chat-session").events)


def test_checkpoint_rejects_cursor_from_another_session_or_missing_event(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    other = SessionStore.for_repository(tmp_path).create("other-session", repository_root=tmp_path, model="fake-model")
    other_cursor = other.events[-1].event_id

    with pytest.raises(LongTaskStateError, match="does not exist"):
        store.checkpoint(
            "repair-task",
            message_projection=[],
            recovery_reason="invalid cursor",
            session_event_cursor=other_cursor,
        )
    with pytest.raises(LongTaskStateError, match="does not exist"):
        store.checkpoint(
            "repair-task",
            message_projection=[],
            recovery_reason="missing cursor",
            session_event_cursor="missing-event",
        )


def test_legacy_checkpoint_without_session_cursor_fails_closed(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    projection = store.replay("repair-task")
    store.append(
        LongTaskEvent(
            task_id="repair-task",
            event_type="checkpoint_recorded",
            payload={
                "checkpoint_id": "legacy-checkpoint",
                "event_cursor": projection.events[-1].event_id,
                "message_projection": [],
                "completed_effect_ids": [],
                "pending_action_ids": [],
                "recovery_reason": "legacy fixture",
                "effects_used": 0,
                "max_effects": None,
            },
        )
    )

    recovered = LongTaskStore.for_repository(tmp_path).resume("repair-task", reason="restart recovered")

    assert recovered.status is LongTaskStatus.RECOVERY_REQUIRED
    assert recovered.recovery_blockers == ["checkpoint_session_cursor_missing"]


def test_replay_ignores_an_incomplete_trailing_task_event(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    with store.path_for("repair-task").open("ab") as stream:
        stream.write(b'{"schema_version":1')

    projection = LongTaskStore.for_repository(tmp_path).replay("repair-task")

    assert projection.goal.startswith("Repair")
    assert projection.warnings == ["Ignored incomplete trailing long Task event"]


def test_invalid_middle_task_event_fails_closed_and_cannot_resume(tmp_path):
    store = LongTaskStore.for_repository(tmp_path)
    _create(store, tmp_path)
    store.plan_action("repair-task", action_id="write-fix", kind="tool_effect", effect_id="write-fix-v1")
    with store.path_for("repair-task").open("ab") as stream:
        stream.write(b"not-json\n")
    store.append(
        # A later valid event proves the invalid line is in the middle, not a partial tail.
        LongTaskEvent(
            task_id="repair-task",
            event_type="task_paused",
            payload={"reason": "after corruption"},
        )
    )

    recovered = LongTaskStore.for_repository(tmp_path).resume("repair-task", reason="restart recovered")

    assert recovered.status is LongTaskStatus.RECOVERY_REQUIRED
    assert recovered.recovery_blockers == ["event_log_invalid"]
    with pytest.raises(LongTaskStateError, match="not running"):
        store.effect_disposition("repair-task", "write-fix")
