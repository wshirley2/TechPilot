"""Negative controls proving that core-v0 detects representative Runtime regressions."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from techpilot.engine.agent import Agent
from techpilot.engine.context import ContextManager
from techpilot.engine.tool_execution import ToolExecutionPlan, ToolExecutionWave
from techpilot.evaluation import ReplayCategory, ReplayRunner, build_core_v0_cases, build_long_task_runtime_v0_cases
from techpilot.runtime import long_task_workflow
from techpilot.runtime.extensions import RoleRegistry
from techpilot.runtime.long_tasks import EffectDisposition, LongTaskStatus, LongTaskStore
from techpilot.runtime.sessions import SessionStore


def _category_cases(category: ReplayCategory):
    return tuple(case for case in build_core_v0_cases() if case.category is category)


def _assert_detected(runner: ReplayRunner, category: ReplayCategory) -> None:
    report = runner.run(_category_cases(category))
    assert report.passed < report.total


def _long_task_cases(*modes: str):
    return tuple(case for case in build_long_task_runtime_v0_cases() if case.input["mode"] in modes)


def _assert_long_task_detected(*modes: str) -> None:
    report = ReplayRunner(Path(__file__).parents[2]).run(_long_task_cases(*modes))
    assert report.passed < report.total


def test_core_detects_c5_write_barrier_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsafe_build(cls, descriptions):
        return cls(tuple(descriptions), (ToolExecutionWave(tuple(range(len(descriptions))), concurrent=True),))

    monkeypatch.setattr(ToolExecutionPlan, "build", classmethod(unsafe_build))
    _assert_detected(ReplayRunner(Path(__file__).parents[2]), ReplayCategory.SCHEDULING)


def test_core_detects_tool_snip_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ContextManager, "_snip_tool_outputs", staticmethod(lambda messages: False))
    _assert_detected(ReplayRunner(Path(__file__).parents[2]), ReplayCategory.CONTEXT)


def test_core_detects_summary_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ContextManager, "_summarize_old", lambda self, messages, llm, keep_recent=8: False)
    report = ReplayRunner(Path(__file__).parents[2]).run(
        tuple(case for case in _category_cases(ReplayCategory.CONTEXT) if case.scenario == "context-summary")
    )
    assert report.passed < report.total


def test_core_detects_hard_collapse_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ContextManager, "_hard_collapse", lambda self, messages, llm: None)
    report = ReplayRunner(Path(__file__).parents[2]).run(
        tuple(case for case in _category_cases(ReplayCategory.CONTEXT) if case.scenario == "context-collapse")
    )
    assert report.passed < report.total


def test_core_detects_session_projection_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    original_replay = SessionStore.replay

    def without_model_projection(self, session_id):
        projection = original_replay(self, session_id)
        projection.model_messages = []
        return projection

    monkeypatch.setattr(SessionStore, "replay", without_model_projection)
    _assert_detected(ReplayRunner(Path(__file__).parents[2]), ReplayCategory.PERSISTENCE)


def test_core_detects_disabled_role_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RoleRegistry, "disable", lambda self, role_id: None)
    report = ReplayRunner(Path(__file__).parents[2]).run(
        tuple(
            case
            for case in _category_cases(ReplayCategory.CONTRACT)
            if case.input["outcome"] == "disabled"
        )
    )
    assert report.passed < report.total


def test_core_detects_missing_tool_request_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Agent, "_emit_tool_requested", lambda self, *args, **kwargs: None)
    _assert_detected(ReplayRunner(Path(__file__).parents[2]), ReplayCategory.TOOL)


def test_core_detects_instruction_history_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Agent, "_full_messages", lambda self: [{"role": "system", "content": self._system}, self.messages[-1]])
    _assert_detected(ReplayRunner(Path(__file__).parents[2]), ReplayCategory.INSTRUCTION)


def test_long_task_detects_completed_effect_reexecution(monkeypatch: pytest.MonkeyPatch) -> None:
    original = LongTaskStore.effect_disposition

    def unsafe_disposition(self, task_id, action_id):
        result = original(self, task_id, action_id)
        return EffectDisposition.EXECUTE if result is EffectDisposition.SKIP_SUCCEEDED else result

    monkeypatch.setattr(LongTaskStore, "effect_disposition", unsafe_disposition)
    _assert_long_task_detected("completed-effect-skip")


def test_long_task_detects_rejected_effect_marked_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(long_task_workflow, "_known_not_executed", lambda result: False)
    _assert_long_task_detected("permission-denied", "policy-blocked", "opaque-tool-exclusive")


def test_long_task_detects_effect_budget_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    original = LongTaskStore.effect_disposition

    def unsafe_disposition(self, task_id, action_id):
        result = original(self, task_id, action_id)
        return EffectDisposition.EXECUTE if result is EffectDisposition.BLOCKED_LIMIT else result

    monkeypatch.setattr(LongTaskStore, "effect_disposition", unsafe_disposition)
    _assert_long_task_detected("effect-budget")


def test_long_task_detects_cancel_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LongTaskStore, "cancel", lambda self, task_id, *, reason: self.replay(task_id))
    _assert_long_task_detected("cancelled-task")


def test_long_task_detects_lease_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LongTaskStore, "acquire_lease", lambda self, task_id, *, owner_id=None: nullcontext(object()))
    monkeypatch.setattr(LongTaskStore, "_require_active_lease", staticmethod(lambda task_id, lease: None))
    _assert_long_task_detected("lease-conflict")


def test_long_task_detects_corrupt_event_log_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    original = LongTaskStore.replay

    def unsafe_replay(self, task_id):
        projection = original(self, task_id)
        if projection.status is LongTaskStatus.RECOVERY_REQUIRED:
            projection.status = LongTaskStatus.RUNNING
            projection.recovery_blockers.clear()
        return projection

    monkeypatch.setattr(LongTaskStore, "replay", unsafe_replay)
    _assert_long_task_detected("corrupt-event-log")
