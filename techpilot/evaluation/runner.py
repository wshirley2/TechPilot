"""Offline, deterministic runner for the frozen Runtime Replay deck."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from techpilot.engine.agent import Agent
from techpilot.engine.context import ContextManager
from techpilot.engine.events import CallbackEventSink, RuntimeEvent, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.tool_execution import ToolConcurrency, ToolEffect, ToolExecutionDescription, ToolExecutionPlan
from techpilot.engine.tools import BashTool, ReadFileTool, WriteFileTool
from techpilot.engine.tools.base import Tool
from techpilot.runtime import (
    LongTaskBudget,
    LongTaskRecoveryRequired,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskWorkflow,
    RuntimeBootstrap,
    RuntimeBootstrapInput,
)
from techpilot.runtime.extensions import (
    PayloadContract,
    RoleHostConfiguration,
    RoleRegistry,
    RoleSkillActivator,
    RoleSpec,
    RuntimeCompatibility,
    SkillRegistry,
    ToolAllowlist,
    ToolRequest,
)
from techpilot.runtime.sessions import SessionEventSink, SessionStore

from .contracts import ReplayCase, ReplayOutcome, ReplayReport, ReplayTrack


class _LongTaskHoldoutAssertionFailure(AssertionError):
    """A private holdout mismatch with deliberately non-sensitive observations."""

    def __init__(self, codes: tuple[str, ...], observed: Mapping[str, Any]) -> None:
        super().__init__("private long-task assertions did not match")
        self.codes = codes
        self.observed = dict(observed)


class ReplayRunner:
    """Run structured cases against the current Runtime implementation."""

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = (repository_root or Path.cwd()).resolve()
        self._handlers: dict[str, Callable[[ReplayCase, Path], Mapping[str, Any]]] = {
            "agent-tool-turn": self._run_agent_tool_turn,
            "tool-execution-plan": self._run_execution_plan,
            "context-snip": self._run_context_snip,
            "context-summary": self._run_context_summary,
            "context-collapse": self._run_context_collapse,
            "session-projection": self._run_session_projection,
            "role-skill-activation": self._run_role_skill_activation,
            "role-runtime-lifecycle": self._run_role_runtime_lifecycle,
            "long-task-runtime": self._run_long_task_runtime,
            "long-task-observability": self._run_long_task_observability,
            "long-task-holdout": self._run_long_task_holdout,
            "instruction-carry": self._run_instruction_carry,
        }

    def run(self, cases: Sequence[ReplayCase]) -> ReplayReport:
        """Run one fixed-suite, Runtime-only deck and retain every outcome."""

        selected = tuple(cases)
        if not selected:
            raise ValueError("replay run requires at least one case")
        if any(case.track is not ReplayTrack.RUNTIME for case in selected):
            raise ValueError("ReplayRunner only runs deterministic runtime cases")
        suites = {case.suite for case in selected}
        if len(suites) != 1:
            raise ValueError("a replay run cannot mix suites")
        outcomes: list[ReplayOutcome] = []
        with tempfile.TemporaryDirectory(prefix="techpilot-replay-") as directory:
            root = Path(directory)
            for case in selected:
                outcomes.append(self._run_case(case, root / case.id))
        return ReplayReport(
            suite=selected[0].suite,
            track=ReplayTrack.RUNTIME,
            cases=selected,
            outcomes=tuple(outcomes),
            git_commit=self._git_commit(),
            git_dirty=self._git_dirty(),
        )

    @staticmethod
    def write_report(report: ReplayReport, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def write_baseline(cls, report: ReplayReport, output: Path) -> Path:
        """Persist a formal baseline only when its source revision is immutable."""

        if report.track is not ReplayTrack.RUNTIME:
            raise ValueError("baseline-v0 only accepts Runtime replay reports")
        if report.suite != "core-v0":
            raise ValueError("baseline-v0 only accepts the frozen core-v0 suite")
        if report.passed != report.total:
            raise ValueError("baseline-v0 requires every core-v0 case to pass")
        if report.git_dirty:
            raise ValueError("baseline-v0 requires a clean Git worktree")
        payload = report.to_dict() | {
            "baseline": {
                "kind": "baseline-v0",
                "frozen_case_set_digest": report.case_set_digest,
            }
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output

    def _run_case(self, case: ReplayCase, root: Path) -> ReplayOutcome:
        handler = self._handlers.get(case.scenario)
        if handler is None:
            return ReplayOutcome(case.id, case.category, False, f"unknown replay scenario: {case.scenario}")
        try:
            root.mkdir(parents=True, exist_ok=True)
            observed = handler(case, root)
        except _LongTaskHoldoutAssertionFailure as error:
            return ReplayOutcome(
                case.id,
                case.category,
                False,
                "long-task-holdout:assertion_mismatch",
                observed={"mismatch_codes": list(error.codes)} | error.observed,
            )
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            if case.scenario == "long-task-holdout":
                return ReplayOutcome(
                    case.id,
                    case.category,
                    False,
                    f"long-task-holdout:{_long_task_holdout_failure_code(error)}",
                )
            return ReplayOutcome(case.id, case.category, False, str(error))
        except Exception as error:  # pragma: no cover - defensive report boundary
            if case.scenario == "long-task-holdout":
                return ReplayOutcome(
                    case.id,
                    case.category,
                    False,
                    f"long-task-holdout:runtime_{type(error).__name__.lower()}",
                )
            return ReplayOutcome(case.id, case.category, False, f"{type(error).__name__}: {error}")
        return ReplayOutcome(case.id, case.category, True, observed=observed)

    def _run_agent_tool_turn(self, case: ReplayCase, root: Path) -> Mapping[str, Any]:
        mode = _string(case.input, "mode")
        value = _string(case.input, "value")
        provider_response = _string(case.input, "provider_response")
        expected_response = _string(case.expected, "response")
        if mode == "success":
            call = ToolCall("tool-1", "echo", {"value": value})
        elif mode == "bad-arguments":
            call = ToolCall("tool-1", "echo", {"unexpected": value})
        elif mode == "unknown-tool":
            call = ToolCall("tool-1", "missing_echo", {"value": value})
        else:
            raise AssertionError(f"unsupported tool replay mode: {mode}")
        provider = _ReplayProvider([LLMResponse(tool_calls=[call]), LLMResponse(content=provider_response)])
        events: list[RuntimeEvent] = []
        store = SessionStore(root / "sessions")
        store.create("replay", repository_root=root, model="fake-replay")
        sink = SessionEventSink(store, CallbackEventSink(events.append))
        agent = Agent(llm=provider, tools=[_EchoTool()], event_sink=sink, session_id="replay")

        response = agent.chat(f"execute {value}")
        projection = store.replay("replay")
        tool_messages = [message for message in projection.messages if message.get("role") == "tool"]
        _expect(response == expected_response, "unexpected final response")
        _expect(len(tool_messages) == 1, "tool replay must produce one durable tool result")
        result = str(tool_messages[0]["content"])
        if "tool_result" in case.expected:
            _expect(result == _string(case.expected, "tool_result"), "Tool result did not preserve the expected argument")
        else:
            _expect(result.startswith(_string(case.expected, "tool_result_prefix")), "Tool rejection result mismatch")
        event_types = [event.event_type.value for event in events]
        _expect(RuntimeEventType.TOOL_REQUESTED.value in event_types, "missing tool_requested event")
        _expect(RuntimeEventType.TOOL_COMPLETED.value in event_types, "missing tool_completed event")
        _expect(event_types[-1] == RuntimeEventType.TURN_COMPLETED.value, "turn did not complete after tool result")
        return {"response": response, "tool_result": result, "event_types": event_types, "projection_size": len(projection.messages)}

    @staticmethod
    def _run_execution_plan(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        del root
        effects = _string(case.input, "effects")
        descriptions = [_description_for(effect, index) for index, effect in enumerate(effects)]
        plan = ToolExecutionPlan.build(descriptions)
        waves = tuple(wave.indexes for wave in plan.waves)
        expected_waves = tuple(tuple(indexes) for indexes in case.expected["waves"])
        _expect(waves == expected_waves, f"unexpected execution waves: {waves}")
        return {"waves": [list(wave) for wave in waves]}

    @staticmethod
    def _run_context_snip(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        del root
        marker = _string(case.input, "marker")
        line_count = _integer(case.input, "line_count")
        position = _string(case.input, "position")
        lines = [f"noise-{index:02d} {'x' * 240}" for index in range(line_count)]
        lines[0 if position == "head" else -1] = f"marker {marker} {'x' * 240}"
        messages = [{"role": "tool", "content": "\n".join(lines)}]
        compressed = ContextManager(max_tokens=200).maybe_compress(messages, None)
        projection = messages[0]["content"]
        _expect(compressed, "tool output did not enter snip compression")
        _expect(marker in projection, "edge evidence was lost during tool snip")
        return {"projection": projection, "marker": marker}

    @staticmethod
    def _run_context_summary(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        del root
        marker = _string(case.input, "marker")
        filler_size = _integer(case.input, "filler_size")
        messages = _context_messages(marker, filler_size)
        compressed = ContextManager(max_tokens=1200).maybe_compress(messages, None)
        projection = str(messages[0].get("content", ""))
        _expect(compressed, "context did not enter summary compression")
        _expect(projection.startswith(_string(case.expected, "prefix")), "expected summary projection was not created")
        _expect(marker in projection, "summary projection lost its file-evidence marker")
        return {"projection": projection, "message_count": len(messages)}

    @staticmethod
    def _run_context_collapse(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        del root
        marker = _string(case.input, "marker")
        filler_size = _integer(case.input, "filler_size")
        messages = _context_messages(marker, filler_size)
        compressed = ContextManager(max_tokens=900).maybe_compress(messages, None)
        projection = str(messages[0].get("content", ""))
        _expect(compressed, "context did not enter hard-collapse compression")
        _expect(projection.startswith(_string(case.expected, "prefix")), "expected hard-collapse projection was not created")
        _expect(marker in projection, "hard-collapse projection lost its file-evidence marker")
        return {"projection": projection, "message_count": len(messages)}

    @staticmethod
    def _run_session_projection(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        mode = _string(case.input, "mode")
        marker = _string(case.input, "marker")
        store = SessionStore(root / "sessions")
        store.create("replay", repository_root=root, model="fake-replay")
        store.append_runtime(_event(RuntimeEventType.TURN_STARTED, "replay", {"user_input": marker}))
        if mode == "tool":
            store.append_runtime(_event(
                RuntimeEventType.TOOL_REQUESTED,
                "replay",
                {"tool_name": "echo", "arguments": {"value": marker}},
                tool_call_id="tool-1",
            ))
            store.append_runtime(_event(
                RuntimeEventType.TOOL_COMPLETED,
                "replay",
                {"tool_name": "echo", "result": f"echo:{marker}", "interrupted": False},
                tool_call_id="tool-1",
            ))
        elif mode == "compressed":
            store.append_runtime(_event(
                RuntimeEventType.CONTEXT_COMPRESSED,
                "replay",
                {"message_projection": [{"role": "user", "content": f"projection:{marker}"}]},
            ))
        elif mode != "turn":
            raise AssertionError(f"unsupported persistence mode: {mode}")
        store.append_runtime(_event(RuntimeEventType.TURN_COMPLETED, "replay", {"content": f"done:{marker}"}))
        projection = store.replay("replay")
        raw_events = [event.to_dict() for event in projection.events]
        model_text = "\n".join(str(message.get("content", "")) for message in projection.model_messages)
        _expect(any(marker in json.dumps(event, ensure_ascii=False) for event in raw_events), "raw Session facts lost marker")
        expected_marker = f"projection:{marker}" if mode == "compressed" else marker
        _expect(expected_marker in model_text, "Session projection lost expected marker")
        return {"event_count": len(raw_events), "model_messages": projection.model_messages}

    @staticmethod
    def _run_role_skill_activation(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        outcome = _string(case.input, "outcome")
        marker = _string(case.input, "marker")
        role = RoleSpec(
            id="replay-role",
            title="Replay role",
            system_prompt="Handle replay evidence.",
            allowed_skill_ids=("replay-skill",),
            input_contract=PayloadContract(schema_id="replay-input-v1", required_keys=("ticket-id",)),
            output_contract=PayloadContract(schema_id="replay-output-v1", required_keys=("summary",)),
        )
        roles = RoleRegistry((role,))
        skills = SkillRegistry(roles)
        skill_path = root / "skills" / "replay" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(
            "---\nname: replay-skill\ndescription: Replay evidence workflow.\ncompatible_role_ids: [replay-role]\n---\n",
            encoding="utf-8",
        )
        skills.discover(root / "skills", approved=outcome not in {"unapproved", "unknown-skill"})
        if outcome == "revoked":
            skills.revoke("replay-skill")
        if outcome == "disabled":
            roles.disable("replay-role")
        runtime = _RoleTarget()
        activator = RoleSkillActivator(roles, skills, (ToolAllowlist(role_id="replay-role", tool_names=("read_log",)),))
        try:
            activation = activator.activate(
                runtime,
                role_id="replay-role",
                role_context=marker,
                skill_names=("missing-skill",) if outcome == "unknown-skill" else ("replay-skill",),
                role_input={} if outcome == "input-invalid" else {"ticket-id": marker},
                tool_requests=(ToolRequest(tool_name="write_file"),) if outcome == "tool-overreach" else (),
            )
            if outcome == "output-overreach":
                activator.validate_outputs(activation, role_output={"unexpected": marker})
            _expect(outcome == "active", f"expected {outcome} to fail closed")
            _expect(runtime.activations == [("replay-role", marker, ("read_log",))], "active Role did not use allowlisted tools")
            return {"outcome": "active", "activations": runtime.activations}
        except ValueError as error:
            _expect(outcome != "active", f"active Role unexpectedly failed: {error}")
            if outcome == "output-overreach":
                _expect(len(runtime.activations) == 1, "output validation must follow a valid Role activation")
            else:
                _expect(runtime.activations == [], "failed Role/Skill activation reached Runtime")
            return {"outcome": outcome, "error": str(error)}

    @staticmethod
    def _run_role_runtime_lifecycle(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        """Exercise Role isolation against the real Runtime, not a recording stub."""

        mode = _string(case.input, "mode")
        marker = _string(case.input, "marker")
        _expect(mode == _string(case.expected, "outcome"), "Role lifecycle case outcome mismatch")
        provider = _ReplayProvider([LLMResponse(content="default-chat-response")])
        runtime = _build_role_runtime(root, provider)
        inspector, reporter = _replay_roles()
        roles = RoleRegistry((inspector, reporter))
        skills = SkillRegistry(roles)
        activator = RoleSkillActivator(
            roles,
            skills,
            (
                ToolAllowlist(role_id=inspector.id, tool_names=("runtime_probe",)),
                ToolAllowlist(role_id=reporter.id),
            ),
        )

        if mode == "registration":
            _expect([item.role.id for item in roles.registrations()] == [inspector.id, reporter.id], "registration order changed")
            _expect(all(item.status == "active" for item in roles.registrations()), "new Role registration is not active")
            _expect(runtime.active_role is None, "registration unexpectedly activated a Role")
        elif mode == "disabled":
            roles.disable(inspector.id)
            try:
                activator.activate(runtime, role_id=inspector.id, role_context=marker)
            except ValueError as error:
                _expect("role is disabled" in str(error), "disabled Role did not fail closed")
            else:
                raise AssertionError("disabled Role reached Runtime")
            _expect(runtime.active_role is None, "disabled Role changed Runtime state")
        elif mode == "incompatible":
            incompatible = inspector.model_copy(
                update={"runtime_compatibility": RuntimeCompatibility(minimum_api_version="2.0.0")}
            )
            try:
                RoleRegistry((incompatible,))
            except ValueError as error:
                _expect("incompatible with Runtime API" in str(error), "incompatible Role was registered")
            else:
                raise AssertionError("incompatible Role was registered")
            _expect(runtime.active_role is None, "incompatible Role changed Runtime state")
        elif mode == "configuration":
            configured = inspector.model_copy(
                update={"host_configuration_contract": PayloadContract(schema_id="runtime-config-v1", required_keys=("region",))}
            )
            configured_roles = RoleRegistry((configured, reporter))
            configured_activator = RoleSkillActivator(
                configured_roles,
                SkillRegistry(configured_roles),
                (ToolAllowlist(role_id=configured.id, tool_names=("runtime_probe",)),),
            )
            try:
                configured_activator.activate(
                    runtime,
                    role_id=configured.id,
                    role_context=marker,
                    host_configuration=RoleHostConfiguration(role_id=configured.id),
                )
            except ValueError as error:
                _expect("missing required fields: region" in str(error), "invalid Role configuration did not fail closed")
            else:
                raise AssertionError("invalid Role configuration reached Runtime")
            _expect(runtime.active_role is None, "invalid Role configuration changed Runtime state")
        elif mode == "activation":
            activation = activator.activate(runtime, role_id=inspector.id, role_context=marker)
            _expect(activation.tool_names == ("runtime_probe",), "Host tool allowlist changed during activation")
            _expect(runtime.active_role is not None and runtime.active_role.role_id == inspector.id, "Role was not activated")
            _expect("runtime_probe" in {tool.name for tool in runtime.tools}, "allowlisted Role tool was not projected")
            _expect(marker in runtime.agent._system_context, "Role context was not projected")
        elif mode == "switch":
            activator.activate(runtime, role_id=inspector.id, role_context=f"{marker}-inspector")
            activator.activate(runtime, role_id=reporter.id, role_context=f"{marker}-reporter")
            _expect(runtime.active_role is not None and runtime.active_role.role_id == reporter.id, "Role switch did not establish target Role")
            _expect("runtime_probe" not in {tool.name for tool in runtime.tools}, "prior Role tool leaked across switch")
            _expect(f"{marker}-reporter" in runtime.agent._system_context, "target Role context missing after switch")
            _expect(f"{marker}-inspector" not in runtime.agent._system_context, "prior Role context leaked across switch")
        elif mode == "scope-exception":
            try:
                with runtime.role_scope(inspector.id, marker, tool_names=("runtime_probe",)):
                    raise RuntimeError("replay cancellation")
            except RuntimeError as error:
                _expect(str(error) == "replay cancellation", "Role scope hid original failure")
            _expect(runtime.active_role is None, "Role scope did not restore default Runtime state")
            _expect("runtime_probe" not in {tool.name for tool in runtime.tools}, "temporary Role tool leaked after exception")
            _expect(marker not in runtime.agent._system_context, "temporary Role context leaked after exception")
        elif mode == "resume":
            activator.activate(runtime, role_id=inspector.id, role_context=marker)
            resumed = _build_role_runtime(root, _ReplayProvider([]), resume_session_id=runtime.agent.session_id)
            _expect(resumed.active_role is None, "Session resume reactivated a historical Role")
            _expect("runtime_probe" not in {tool.name for tool in resumed.tools}, "Session resume restored historical Role tools")
            _expect(any("不会自动恢复 Role、工具或历史授权" in notice for notice in resumed.consume_recovery_notices()), "Session resume omitted Role recovery notice")
        elif mode == "clear-chat":
            activator.activate(runtime, role_id=inspector.id, role_context=marker)
            runtime.clear_role()
            response = runtime.run_turn("continue as default Chat")
            _expect(response == "default-chat-response", "default Chat response mismatch after Role cleanup")
            _expect(runtime.active_role is None, "clear_role did not restore default Runtime state")
            _expect("runtime_probe" not in {tool.name for tool in runtime.tools}, "cleared Role tool leaked into default Chat")
            _expect(marker not in runtime.agent._system_context, "cleared Role context leaked into default Chat")
            _expect(provider.requests and provider.requests[-1]["tools"] == [], "cleared Role tool remained model-visible")
        else:
            raise ValueError(f"unknown role lifecycle mode: {mode}")
        return {"outcome": mode}

    @staticmethod
    def _run_long_task_runtime(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        """Exercise fixed-provider interruption recovery through the real Runtime."""

        mode = _string(case.input, "mode")
        _expect(mode == _string(case.expected, "outcome"), "long-task case outcome mismatch")
        if mode in {"normal-write", "checkpoint-cursor"}:
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([_write_response(), LLMResponse(content="done")]))
            result = LongTaskWorkflow(runtime, task_id="long-task", goal="write deterministic result").run_turn("write result")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "normal long Task did not succeed")
            _expect(counter.calls == 1, "normal write did not execute exactly once")
            _expect((root / "result.txt").read_text(encoding="utf-8") == "durable result\n", "write result mismatch")
            if mode == "checkpoint-cursor":
                session = runtime.session_store.replay(runtime.agent.session_id)
                _expect(result.checkpoint.session_event_cursor == session.events[-1].event_id, "checkpoint cursor is not the latest Session fact")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "normal-read":
            (root / "evidence.txt").write_text("durable evidence\n", encoding="utf-8")
            response = LLMResponse(tool_calls=[ToolCall("read-1", "read_file", {"file_path": "evidence.txt"})])
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([response, LLMResponse(content="read done")]))
            result = LongTaskWorkflow(runtime, task_id="long-task", goal="read deterministic evidence").run_turn("read evidence")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "read-only long Task did not succeed")
            _expect(counter.calls == 1 and result.task.completed_effect_ids == (), "read created an effect ledger entry")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "normal-command":
            response = LLMResponse(tool_calls=[ToolCall("command-1", "bash", {"command": "python -c \"print('ok')\""})])
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([response, LLMResponse(content="command done")]))
            result = LongTaskWorkflow(runtime, task_id="long-task", goal="run deterministic command").run_turn("run command")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "command long Task did not succeed")
            _expect(counter.calls == 1 and len(result.task.completed_effect_ids) == 1, "command effect ledger mismatch")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "permission-denied":
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([_write_response()]), allow=False)
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="write deterministic result")
            try:
                workflow.run_turn("write result")
            except LongTaskStateError:
                pass
            else:
                raise AssertionError("rejected effect unexpectedly completed the Task")
            task = workflow.store.replay("long-task")
            _expect(task.actions["tool-write-1"].status.value == "failed", "rejected effect was not recorded as not executed")
            _expect(not (root / "result.txt").exists(), "rejected write changed the repository")
            _expect(counter.calls == 1, "permission path did not reach the normal executor boundary")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "policy-blocked":
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([_write_response(file_path="../outside.txt")]))
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="blocked deterministic write")
            try:
                workflow.run_turn("write outside")
            except LongTaskStateError:
                pass
            else:
                raise AssertionError("policy-blocked effect unexpectedly completed the Task")
            task = workflow.store.replay("long-task")
            _expect(task.actions["tool-write-1"].status.value == "failed", "policy-blocked effect was not recorded as not executed")
            _expect(not (root.parent / "outside.txt").exists(), "policy-blocked write escaped repository")
            _expect(counter.calls == 1, "policy path did not reach the executor boundary")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "interrupt-before-effect":
            first, first_counter = _build_long_task_runtime(root, _ReplayProvider([_write_response()]))

            def interrupt(phase: str, action_id: str) -> None:
                _expect(action_id == "tool-write-1", "unexpected action id")
                if phase == "before_effect":
                    raise KeyboardInterrupt("replay interruption before effect")

            with _expect_keyboard_interrupt():
                LongTaskWorkflow(first, task_id="long-task", goal="write deterministic result", fault_hook=interrupt).run_turn("write result")
            resumed, resumed_counter = _build_long_task_runtime(root, _ReplayProvider([_write_response(), LLMResponse(content="resumed")]), resume_session_id=first.agent.session_id)
            result = LongTaskWorkflow(resumed, task_id="long-task", goal="write deterministic result").run_turn("continue")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "pre-effect interruption did not recover")
            _expect(first_counter.calls == 0 and resumed_counter.calls == 1, "pre-effect interruption did not execute exactly once after recovery")
            return {"outcome": mode, "initial_tool_calls": first_counter.calls, "resumed_tool_calls": resumed_counter.calls}

        if mode == "read-then-interrupt":
            (root / "evidence.txt").write_text("durable evidence\n", encoding="utf-8")
            calls = [
                ToolCall("read-1", "read_file", {"file_path": "evidence.txt"}),
                ToolCall("write-1", "write_file", {"file_path": "result.txt", "content": "durable result\n"}),
            ]
            first, first_counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=calls)]))

            def interrupt(phase: str, action_id: str) -> None:
                if phase == "before_effect":
                    _expect(action_id == "tool-write-1", "read must complete before the write boundary")
                    raise KeyboardInterrupt("replay interruption after read")

            with _expect_keyboard_interrupt():
                LongTaskWorkflow(first, task_id="long-task", goal="read then write", fault_hook=interrupt).run_turn("read and write")
            task = LongTaskWorkflow(first, task_id="long-task", goal="read then write").store.replay("long-task")
            _expect(task.actions["tool-read-1"].status.value == "succeeded", "read was not durable before interruption")
            _expect(first_counter.calls == 1 and not (root / "result.txt").exists(), "write started before its boundary")
            return {"outcome": mode, "tool_calls": first_counter.calls}

        if mode == "command-completed-skip":
            command = ToolCall("command-1", "bash", {"command": "python -c \"print('ok')\""})
            first, first_counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=[command])]))

            def interrupt(phase: str, action_id: str) -> None:
                if phase == "after_effect_completed":
                    _expect(action_id == "tool-command-1", "unexpected command action")
                    raise KeyboardInterrupt("replay interruption after command")

            with _expect_keyboard_interrupt():
                LongTaskWorkflow(first, task_id="long-task", goal="run command", fault_hook=interrupt).run_turn("run command")
            resumed, resumed_counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=[command]), LLMResponse(content="recovered")]), resume_session_id=first.agent.session_id)
            result = LongTaskWorkflow(resumed, task_id="long-task", goal="run command").run_turn("continue")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "command Task did not recover")
            _expect(first_counter.calls == 1 and resumed_counter.calls == 0, "completed command was executed again")
            return {"outcome": mode, "initial_tool_calls": first_counter.calls, "resumed_tool_calls": resumed_counter.calls}

        if mode == "same-round-read-write":
            (root / "evidence.txt").write_text("durable evidence\n", encoding="utf-8")
            calls = [ToolCall("read-1", "read_file", {"file_path": "evidence.txt"}), _write_response().tool_calls[0]]
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=calls), LLMResponse(content="done")]))
            result = LongTaskWorkflow(runtime, task_id="long-task", goal="read then write").run_turn("read and write")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED and counter.calls == 2, "same-round read/write did not complete")
            _expect((root / "result.txt").exists(), "same-round write was missing")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "same-round-write-write":
            first_calls = [
                ToolCall("write-1", "write_file", {"file_path": "first.txt", "content": "first\n"}),
                ToolCall("write-2", "write_file", {"file_path": "second.txt", "content": "second\n"}),
            ]
            first, first_counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=first_calls)]))

            def interrupt(phase: str, action_id: str) -> None:
                if phase == "after_effect_completed" and action_id == "tool-write-1":
                    raise KeyboardInterrupt("replay interruption after first write")

            with _expect_keyboard_interrupt():
                LongTaskWorkflow(first, task_id="long-task", goal="write twice", fault_hook=interrupt).run_turn("write both")
            _expect((root / "first.txt").exists() and not (root / "second.txt").exists(), "second write started before recovery")
            resumed, resumed_counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=first_calls), LLMResponse(content="recovered")]), resume_session_id=first.agent.session_id)
            result = LongTaskWorkflow(resumed, task_id="long-task", goal="write twice").run_turn("continue")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "double-write Task did not recover")
            _expect(first_counter.calls == 1 and resumed_counter.calls == 1, "first write repeated or second write was skipped")
            _expect((root / "second.txt").read_text(encoding="utf-8") == "second\n", "second write result missing")
            return {"outcome": mode, "initial_tool_calls": first_counter.calls, "resumed_tool_calls": resumed_counter.calls}

        if mode == "same-round-read-read":
            (root / "one.txt").write_text("one\n", encoding="utf-8")
            (root / "two.txt").write_text("two\n", encoding="utf-8")
            calls = [ToolCall("read-1", "read_file", {"file_path": "one.txt"}), ToolCall("read-2", "read_file", {"file_path": "two.txt"})]
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=calls), LLMResponse(content="done")]))
            result = LongTaskWorkflow(runtime, task_id="long-task", goal="read both").run_turn("read both")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED and counter.calls == 2, "same-round reads did not complete")
            _expect(result.task.completed_effect_ids == (), "safe reads created effect entries")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "opaque-tool-exclusive":
            response = LLMResponse(tool_calls=[ToolCall("opaque-1", "opaque_probe", {})])
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([response, LLMResponse(content="opaque done")]))
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="run opaque Tool")
            try:
                workflow.run_turn("run opaque")
            except LongTaskStateError:
                pass
            else:
                raise AssertionError("opaque Tool unexpectedly completed")
            task = workflow.store.replay("long-task")
            _expect(task.actions["tool-opaque-1"].status.value == "failed", "opaque Tool was not fail-closed")
            _expect(task.completed_effect_ids == (), "opaque Tool was incorrectly recorded as completed")
            _expect(counter.calls == 1, "opaque Tool did not reach executor")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode in {"session-mismatch", "repository-mismatch"}:
            first, _ = _build_long_task_runtime(root, _ReplayProvider([]))
            LongTaskWorkflow(first, task_id="long-task", goal="identity check").start_or_resume()
            target = root if mode == "session-mismatch" else root / "other-repository"
            target.mkdir(exist_ok=True)
            second, counter = _build_long_task_runtime(target, _ReplayProvider([LLMResponse(content="must not run")]))
            candidate = LongTaskWorkflow(second, task_id="long-task", goal="identity check")
            if mode == "repository-mismatch":
                candidate.store = LongTaskWorkflow(first, task_id="long-task", goal="identity check").store
            with _expect_state_error():
                candidate.run_turn("continue")
            _expect(counter.calls == 0, "identity mismatch reached Tool executor")
            return {"outcome": mode, "tool_calls": 0}

        if mode == "lease-conflict":
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([_write_response()]))
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="lease check")
            workflow.start_or_resume()
            with workflow.store.acquire_lease("long-task"), _expect_state_error():
                workflow.run_turn("write result")
            _expect(counter.calls == 0, "second lease holder reached Tool executor")
            return {"outcome": mode, "tool_calls": 0}

        if mode == "effect-budget":
            calls = [_write_response(file_path="first.txt").tool_calls[0], _write_response(file_path="second.txt").tool_calls[0]]
            calls[1].id = "write-2"
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(tool_calls=calls)]))
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="budget check", budget=LongTaskBudget(max_effects=1))
            with _expect_state_error():
                workflow.run_turn("write twice")
            _expect(counter.calls == 1 and not (root / "second.txt").exists(), "effect budget allowed second effect")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "cancelled-task":
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([_write_response()]))
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="cancel check")
            workflow.start_or_resume()
            workflow.store.cancel("long-task", reason="replay cancellation")
            with _expect_state_error():
                workflow.run_turn("write result")
            _expect(counter.calls == 0, "cancelled Task reached Tool executor")
            return {"outcome": mode, "tool_calls": 0}

        if mode in {"two-turn-read-write", "two-turn-write-read"}:
            (root / "evidence.txt").write_text("durable evidence\n", encoding="utf-8")
            read = LLMResponse(tool_calls=[ToolCall("read-1", "read_file", {"file_path": "evidence.txt"})])
            write = _write_response()
            first_call, first_done, second_call, second_done = (
                (read, "read done", write, "write done")
                if mode == "two-turn-read-write"
                else (write, "write done", read, "read done")
            )
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([
                first_call, LLMResponse(content=first_done), second_call, LLMResponse(content=second_done),
            ]))
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="two deterministic steps")
            first = workflow.run_step("first step")
            second = workflow.run_step("second step")
            task = workflow.finish(result=second.response)
            _expect(task.status is LongTaskStatus.SUCCEEDED, "two-turn Task did not finish")
            _expect(first.checkpoint.session_event_cursor != second.checkpoint.session_event_cursor, "two turns reused a Session cursor")
            _expect(counter.calls == 2, "two-turn Task did not execute exactly two Tool calls")
            if mode == "two-turn-read-write":
                _expect((root / "result.txt").exists(), "second-turn write was missing")
            else:
                _expect(task.completed_effect_ids == ("effect-write-1",), "first-turn write was not retained")
            return {"outcome": mode, "tool_calls": counter.calls}

        if mode == "corrupt-event-log":
            runtime, counter = _build_long_task_runtime(root, _ReplayProvider([LLMResponse(content="must not run")]))
            workflow = LongTaskWorkflow(runtime, task_id="long-task", goal="corruption check")
            workflow.start_or_resume()
            with workflow.store.path_for("long-task").open("ab") as stream:
                stream.write(b"not-json\n")
            with _expect_recovery_required():
                workflow.run_step("continue")
            _expect(counter.calls == 0, "corrupt Task log reached Tool executor")
            return {"outcome": mode, "tool_calls": 0}

        if mode == "unknown-effect":
            first, first_counter = _build_long_task_runtime(root, _ReplayProvider([_write_response()]))

            def interrupt(phase: str, action_id: str) -> None:
                _expect(action_id == "tool-write-1", "unexpected action id")
                if phase == "after_effect_started":
                    raise KeyboardInterrupt("replay interruption before executor")

            with _expect_keyboard_interrupt():
                LongTaskWorkflow(first, task_id="long-task", goal="write deterministic result", fault_hook=interrupt).run_turn("write result")
            resumed, resumed_counter = _build_long_task_runtime(
                root,
                _ReplayProvider([LLMResponse(content="must not run")]),
                resume_session_id=first.agent.session_id,
            )
            with _expect_recovery_required():
                LongTaskWorkflow(resumed, task_id="long-task", goal="write deterministic result").run_turn("continue")
            _expect(first_counter.calls == 0 and resumed_counter.calls == 0, "unknown effect reached executor during recovery")
            _expect(not (root / "result.txt").exists(), "unknown effect changed the repository")
            return {"outcome": mode, "tool_calls": 0}

        if mode == "completed-effect-skip":
            first, first_counter = _build_long_task_runtime(root, _ReplayProvider([_write_response()]))

            def interrupt(phase: str, action_id: str) -> None:
                _expect(action_id == "tool-write-1", "unexpected action id")
                if phase == "after_effect_completed":
                    raise KeyboardInterrupt("replay interruption after completion")

            with _expect_keyboard_interrupt():
                LongTaskWorkflow(first, task_id="long-task", goal="write deterministic result", fault_hook=interrupt).run_turn("write result")
            resumed, resumed_counter = _build_long_task_runtime(
                root,
                _ReplayProvider([_write_response(), LLMResponse(content="recovered")]),
                resume_session_id=first.agent.session_id,
            )
            result = LongTaskWorkflow(resumed, task_id="long-task", goal="write deterministic result").run_turn("continue")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "recovered Task did not succeed")
            _expect(first_counter.calls == 1 and resumed_counter.calls == 0, "completed effect was executed again")
            return {"outcome": mode, "initial_tool_calls": first_counter.calls, "resumed_tool_calls": resumed_counter.calls}

        if mode == "plain-session-control":
            first, first_counter = _build_long_task_runtime(root, _ReplayProvider([_write_response()]))
            first.agent.tool_executor = _InterruptAfterExecuteExecutor(first.agent.tool_executor)
            with _expect_keyboard_interrupt():
                first.run_turn("write result")
            resumed, resumed_counter = _build_long_task_runtime(
                root,
                _ReplayProvider([_write_response(), LLMResponse(content="ordinary recovery")]),
                resume_session_id=first.agent.session_id,
            )
            _expect(resumed.run_turn("continue") == "ordinary recovery", "ordinary Session did not resume")
            _expect(first_counter.calls == 1 and resumed_counter.calls == 1, "ordinary Session control did not reissue the Tool call")
            return {"outcome": mode, "initial_tool_calls": first_counter.calls, "resumed_tool_calls": resumed_counter.calls}
        raise ValueError(f"unknown long-task runtime mode: {mode}")

    @staticmethod
    def _run_long_task_observability(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        """Observe Tool bodies and effects for public RS-6.2 extension cards.

        These checks intentionally use the ordinary Runtime executor and C5
        scheduler. The probes only record entry/exit around real Tool bodies;
        they do not replace permission checks or Tool execution.
        """

        mode = _string(case.input, "mode")
        _expect(mode == _string(case.expected, "outcome"), "long-task observability outcome mismatch")
        trace = _ToolBodyTrace()

        if mode == "command-effect-recovery":
            command_tool = _ObservedBashTool(trace)
            command = (
                f'"{sys.executable}" -c "open(\'command-marker.txt\', \'w\', '
                "encoding='utf-8').write('actual command')\""
            )
            call = ToolCall("command-1", "bash", {"command": command})
            first, _ = _build_long_task_runtime(
                root,
                _ReplayProvider([LLMResponse(tool_calls=[call])]),
                tools=[command_tool],
            )

            def interrupt(phase: str, action_id: str) -> None:
                if phase == "after_effect_completed":
                    _expect(action_id == "tool-command-1", "unexpected command action")
                    raise KeyboardInterrupt("deterministic interruption after actual command")

            try:
                with _expect_keyboard_interrupt():
                    LongTaskWorkflow(
                        first,
                        task_id="observability-task",
                        goal="write a command marker",
                        fault_hook=interrupt,
                    ).run_turn("run the command")
            except LongTaskStateError as error:
                current = LongTaskWorkflow(first, task_id="observability-task", goal="write a command marker").store.replay("observability-task")
                raise AssertionError(
                    f"initial command interruption did not reach its effect boundary: {error}; "
                    f"actions={[(action_id, action.status.value) for action_id, action in current.actions.items()]}"
                ) from error
            marker = root / "command-marker.txt"
            _expect(command_tool.body_calls == 1, "command Tool body was not executed exactly once before interruption")
            _expect(marker.read_text(encoding="utf-8") == "actual command", "command did not create its marker")
            first_task = LongTaskWorkflow(first, task_id="observability-task", goal="write a command marker").store.replay("observability-task")
            _expect(first_task.actions["tool-command-1"].status.value == "succeeded", "actual command was not durably marked completed")

            resumed_tool = _ObservedBashTool(trace)
            resumed, _ = _build_long_task_runtime(
                root,
                _ReplayProvider([LLMResponse(tool_calls=[call]), LLMResponse(content="recovered")]),
                tools=[resumed_tool],
                resume_session_id=first.agent.session_id,
            )
            resumed_workflow = LongTaskWorkflow(
                resumed,
                task_id="observability-task",
                goal="write a command marker",
            )
            try:
                step = resumed_workflow.run_step("continue")
            except LongTaskStateError as error:
                current = resumed_workflow.store.replay("observability-task")
                raise AssertionError(
                    f"resumed command step did not retain the completed effect: {error}; "
                    f"actions={[(action_id, action.status.value) for action_id, action in current.actions.items()]}"
                ) from error
            recovered = resumed_workflow.store.replay("observability-task")
            _expect(recovered.actions["tool-command-1"].status.value == "succeeded", "skipped command was not retained as completed")
            try:
                task = resumed_workflow.finish(result=step.response)
            except LongTaskStateError as error:
                current = resumed_workflow.store.replay("observability-task")
                raise AssertionError(
                    f"completed command could not finish: {error}; "
                    f"action={current.actions['tool-command-1'].status.value}; "
                    f"actions={sorted(current.actions)}"
                ) from error
            _expect(task.status is LongTaskStatus.SUCCEEDED, "recovered command Task did not succeed")
            _expect(resumed_tool.body_calls == 0, "completed command Tool body executed again during recovery")
            _expect(marker.read_text(encoding="utf-8") == "actual command", "command marker changed during recovery")
            return {"outcome": mode, "initial_tool_body_calls": command_tool.body_calls, "resumed_tool_body_calls": resumed_tool.body_calls}

        if mode == "read-read-overlap":
            (root / "one.txt").write_text("one\n", encoding="utf-8")
            (root / "two.txt").write_text("two\n", encoding="utf-8")
            read_tool = _ObservedReadTool(trace, barrier=threading.Barrier(2))
            calls = [
                ToolCall("read-1", "read_file", {"file_path": "one.txt"}),
                ToolCall("read-2", "read_file", {"file_path": "two.txt"}),
            ]
            runtime, _ = _build_long_task_runtime(
                root,
                _ReplayProvider([LLMResponse(tool_calls=calls), LLMResponse(content="reads complete")]),
                tools=[read_tool],
            )
            result = LongTaskWorkflow(runtime, task_id="observability-task", goal="read two files").run_turn("read both")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "read/read Task did not succeed")
            _expect(read_tool.body_calls == 2, "both read Tool bodies did not execute")
            _expect(trace.max_active == 2, "safe reads did not overlap in actual Tool bodies")
            return {"outcome": mode, "tool_body_calls": read_tool.body_calls, "peak_active_tool_bodies": trace.max_active}

        if mode == "read-write-order":
            (root / "evidence.txt").write_text("evidence\n", encoding="utf-8")
            read_tool = _ObservedReadTool(trace)
            calls = [
                ToolCall("read-1", "read_file", {"file_path": "evidence.txt"}),
                ToolCall("write-1", "write_file", {"file_path": "result.txt", "content": "written after read\n"}),
            ]
            runtime, _ = _build_long_task_runtime(
                root,
                _ReplayProvider([LLMResponse(tool_calls=calls), LLMResponse(content="ordered")]),
                tools=[read_tool, WriteFileTool()],
                effect_trace=trace,
            )
            result = LongTaskWorkflow(runtime, task_id="observability-task", goal="read then write").run_turn("read then write")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "read/write Task did not succeed")
            _expect(
                trace.events == ["read:evidence.txt:start", "read:evidence.txt:end", "write:result.txt:start", "write:result.txt:end"],
                f"read/write Tool body order changed: {trace.events}",
            )
            _expect((root / "result.txt").read_text(encoding="utf-8") == "written after read\n", "write Tool body did not create expected file")
            return {"outcome": mode, "tool_body_events": list(trace.events)}

        if mode == "write-write-exclusive":
            calls = [
                ToolCall("write-1", "write_file", {"file_path": "first.txt", "content": "first\n"}),
                ToolCall("write-2", "write_file", {"file_path": "second.txt", "content": "second\n"}),
            ]
            runtime, _ = _build_long_task_runtime(
                root,
                _ReplayProvider([LLMResponse(tool_calls=calls), LLMResponse(content="writes complete")]),
                tools=[WriteFileTool()],
                effect_trace=trace,
            )
            result = LongTaskWorkflow(runtime, task_id="observability-task", goal="write twice").run_turn("write both")
            _expect(result.task.status is LongTaskStatus.SUCCEEDED, "write/write Task did not succeed")
            _expect(
                trace.completed_write_effects == 2 and trace.max_active == 1,
                f"writes overlapped in actual effects: writes={trace.completed_write_effects}, peak={trace.max_active}, events={trace.events}",
            )
            _expect(
                trace.events == ["write:first.txt:start", "write:first.txt:end", "write:second.txt:start", "write:second.txt:end"],
                f"write/write Tool body order changed: {trace.events}",
            )
            _expect((root / "first.txt").read_text(encoding="utf-8") == "first\n", "first write result missing")
            _expect((root / "second.txt").read_text(encoding="utf-8") == "second\n", "second write result missing")
            return {"outcome": mode, "completed_write_effects": trace.completed_write_effects, "peak_active_effects": trace.max_active}

        raise ValueError(f"unknown long-task observability mode: {mode}")

    @staticmethod
    def _run_long_task_holdout(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        """Run an independently-authored fixed-provider long-task card.

        The card contract intentionally describes only stable Runtime concepts;
        it never selects an internal public-suite mode or bypasses the normal
        Tool executor, Permission gate, C5 scheduling, or Session store.
        """

        initial_state = _mapping_value(case.input, "initial_state")
        provider_script = _mapping_value(case.input, "provider_script")
        interruption = _mapping_value(case.input, "interruption_point")
        recovery = _mapping_value(case.input, "recovery_action")
        assertions = _mapping_value(case.expected, "assertions")

        files = _mapping_value(initial_state, "files")
        for relative_path, content in files.items():
            if not isinstance(relative_path, str) or not isinstance(content, str):
                raise TypeError("long-task holdout initial files must map string paths to string content")
            target = _private_case_path(root, relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        user_input = _string(initial_state, "user_input")
        task_spec = _mapping_value(initial_state, "task")
        task_id = _string(task_spec, "id")
        goal = _string(task_spec, "goal")
        max_effects = _integer(task_spec, "max_effects")
        permission = _string(initial_state, "permission")
        if permission not in {"allow", "deny"}:
            raise ValueError("long-task holdout permission must be allow or deny")

        phase = _string(interruption, "phase")
        if phase not in {"none", "before_effect", "after_effect_started", "after_effect_completed"}:
            raise ValueError("long-task holdout interruption phase is unsupported")
        target_call_id = interruption.get("tool_call_id")
        if target_call_id is not None and not isinstance(target_call_id, str):
            raise TypeError("long-task holdout interruption tool_call_id must be a string when present")

        initial_responses = _long_task_provider_responses(provider_script, "initial")
        resume_responses = _long_task_provider_responses(provider_script, "resume")
        first, first_counter = _build_long_task_runtime(
            root,
            _ReplayProvider(initial_responses),
            allow=permission == "allow",
        )

        def interrupt(actual_phase: str, action_id: str) -> None:
            if phase == "none" or actual_phase != phase:
                return
            if target_call_id is not None and action_id != f"tool-{target_call_id}":
                return
            raise KeyboardInterrupt("private long-task holdout interruption")

        try:
            LongTaskWorkflow(
                first,
                task_id=task_id,
                goal=goal,
                budget=LongTaskBudget(max_effects=max_effects),
                fault_hook=interrupt if phase != "none" else None,
            ).run_turn(user_input)
        except (KeyboardInterrupt, LongTaskStateError):
            pass

        recovery_kind = _string(recovery, "kind")
        resumed_counter_calls = 0
        if recovery_kind == "resume":
            recovery_input = _string(recovery, "user_input")
            resumed, resumed_counter = _build_long_task_runtime(
                root,
                _ReplayProvider(resume_responses),
                allow=permission == "allow",
                resume_session_id=first.agent.session_id,
            )
            resumed_counter_calls = resumed_counter.calls
            try:
                LongTaskWorkflow(
                    resumed,
                    task_id=task_id,
                    goal=goal,
                    budget=LongTaskBudget(max_effects=max_effects),
                ).run_turn(recovery_input)
            except LongTaskStateError:
                pass
            resumed_counter_calls = resumed_counter.calls
        elif recovery_kind == "verify_stop":
            resumed, resumed_counter = _build_long_task_runtime(
                root,
                _ReplayProvider(resume_responses),
                allow=permission == "allow",
                resume_session_id=first.agent.session_id,
            )
            try:
                LongTaskWorkflow(
                    resumed,
                    task_id=task_id,
                    goal=goal,
                    budget=LongTaskBudget(max_effects=max_effects),
                ).run_step(_string(recovery, "user_input"))
            except LongTaskRecoveryRequired:
                pass
            else:
                raise AssertionError("private long-task recovery did not stop for an ambiguous effect")
            resumed_counter_calls = resumed_counter.calls
        elif recovery_kind != "none":
            raise ValueError("long-task holdout recovery kind is unsupported")

        task = LongTaskWorkflow(first, task_id=task_id, goal=goal).store.replay(task_id)
        mismatch_codes: list[str] = []
        if task.status.value != _string(assertions, "task_status"):
            mismatch_codes.append("assertion_task_status_mismatch")
        if first_counter.calls != _integer(assertions, "initial_executor_calls"):
            mismatch_codes.append("assertion_initial_executor_count_mismatch")
        if resumed_counter_calls != _integer(assertions, "resume_executor_calls"):
            mismatch_codes.append("assertion_resume_executor_count_mismatch")
        expected_effects = _string_list(assertions, "completed_effect_ids")
        if task.completed_effect_ids != expected_effects:
            mismatch_codes.append("assertion_completed_effect_ledger_mismatch")
        session = first.session_store.replay(first.agent.session_id)
        session_cursor_advanced = bool(
            task.checkpoints
            and session.events
            and task.checkpoints[-1].session_event_cursor == session.events[-1].event_id
        )
        if session_cursor_advanced is not _boolean(assertions, "session_cursor_advanced"):
            mismatch_codes.append("assertion_session_cursor_mismatch")
        expected_files = _mapping_value(assertions, "files")
        matched_file_assertions = 0
        for relative_path, expectation in expected_files.items():
            if not isinstance(relative_path, str):
                raise TypeError("private long-task file assertion path must be a string")
            expectation_map = _mapping(expectation)
            exists = expectation_map.get("exists")
            if not isinstance(exists, bool):
                raise TypeError("private long-task file assertion requires boolean exists")
            if _private_case_path(root, relative_path).exists() is exists:
                matched_file_assertions += 1
        if matched_file_assertions != len(expected_files):
            mismatch_codes.append("assertion_file_existence_mismatch")
        observed = {
            "task_status": task.status.value,
            "initial_executor_calls": first_counter.calls,
            "resume_executor_calls": resumed_counter_calls,
            "completed_effect_count": len(task.completed_effect_ids),
            "session_cursor_advanced": session_cursor_advanced,
            "file_assertions": {"matched": matched_file_assertions, "total": len(expected_files)},
        }
        if mismatch_codes:
            raise _LongTaskHoldoutAssertionFailure(tuple(mismatch_codes), observed)
        return {
            "task_status": task.status.value,
            "initial_executor_calls": first_counter.calls,
            "resume_executor_calls": resumed_counter_calls,
        }

    @staticmethod
    def _run_instruction_carry(case: ReplayCase, root: Path) -> Mapping[str, Any]:
        del root
        constraint = _string(case.input, "constraint")
        follow_up = _string(case.input, "follow_up")
        provider_response = _string(case.input, "provider_response")
        response = _string(case.expected, "response")
        provider = _ReplayProvider([LLMResponse(content="constraint stored"), LLMResponse(content=provider_response)])
        agent = Agent(llm=provider, tools=[])
        first = agent.chat(constraint, allow_tools=False)
        second = agent.chat(follow_up, allow_tools=False)
        second_request = provider.requests[1]["messages"]
        visible_context = json.dumps(second_request, ensure_ascii=False)
        _expect(first == "constraint stored", "first instruction response mismatch")
        _expect(second == response, "second instruction response mismatch")
        _expect(_string(case.expected, "constraint") in visible_context, "later turn lost the initial user constraint")
        return {"response": second, "second_request": second_request}

    def _git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repository_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    def _git_dirty(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return True
        return bool(result.stdout.strip())


class _ReplayProvider:
    model = "fake-replay"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, on_token=None) -> LLMResponse:
        self.requests.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        response = next(self._responses)
        if on_token is not None and response.content:
            on_token(response.content)
        return response


class _ReplayAllowPrompt:
    def __init__(self, *, allow: bool) -> None:
        self.allow = allow

    def decide(self, request):
        del request
        return PermissionDecision.allow("replay approval") if self.allow else PermissionDecision.deny("replay rejection")


@dataclass
class _CountingExecutor:
    delegate: object
    calls: int = 0

    def begin_turn(self):
        return self.delegate.begin_turn()

    def consume_turn_stop_message(self):
        return self.delegate.consume_turn_stop_message()

    def describe_call(self, tool, arguments):
        return self.delegate.describe_call(tool, arguments)

    def execute_call(self, tool, arguments, *, tool_call_id, execution_context=None):
        self.calls += 1
        return self.delegate.execute_call(
            tool,
            arguments,
            tool_call_id=tool_call_id,
            execution_context=execution_context,
        )


@dataclass
class _ToolBodyTrace:
    """Thread-safe observations made inside public deterministic Tool bodies."""

    events: list[str] = field(default_factory=list)
    active: int = 0
    max_active: int = 0
    completed_write_effects: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def enter(self, label: str) -> None:
        with self._lock:
            self.events.append(f"{label}:start")
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self, label: str) -> None:
        with self._lock:
            self.active -= 1
            self.events.append(f"{label}:end")

    def complete_write(self) -> None:
        with self._lock:
            self.completed_write_effects += 1


class _ObservedReadTool(ReadFileTool):
    """ReadFileTool whose real body records overlap without changing output."""

    def __init__(self, trace: _ToolBodyTrace, *, barrier: threading.Barrier | None = None) -> None:
        self.trace = trace
        self.barrier = barrier
        self.body_calls = 0

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        label = f"read:{Path(file_path).name}"
        self.body_calls += 1
        self.trace.enter(label)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            time.sleep(0.02)
            return super().execute(file_path, offset=offset, limit=limit)
        finally:
            self.trace.leave(label)


class _ObservedBashTool(BashTool):
    """BashTool that records actual ``execute_in`` entry without stubbing it."""

    def __init__(self, trace: _ToolBodyTrace) -> None:
        self.trace = trace
        self.body_calls = 0

    def execute_in(self, command: str, *, cwd: str, timeout: int = 120) -> str:
        label = "bash:command"
        self.body_calls += 1
        self.trace.enter(label)
        try:
            return super().execute_in(command, cwd=cwd, timeout=timeout)
        finally:
            self.trace.leave(label)


@dataclass
class _ObservedEffectExecutor:
    """Observe durable RepositoryToolExecutor writes without replacing them.

    ``write_file`` is intentionally applied by the trusted-diff executor,
    rather than by ``WriteFileTool.execute``. The probe therefore surrounds
    that real executor path and records completion only when its durable file
    result confirms a write.
    """

    delegate: object
    trace: _ToolBodyTrace

    def begin_turn(self):
        return self.delegate.begin_turn()

    def consume_turn_stop_message(self):
        return self.delegate.consume_turn_stop_message()

    def describe_call(self, tool, arguments):
        return self.delegate.describe_call(tool, arguments)

    def execute_call(self, tool, arguments, *, tool_call_id, execution_context=None):
        is_write = tool.name == "write_file"
        label = f"write:{Path(str(arguments.get('file_path', 'unknown'))).name}"
        if is_write:
            self.trace.enter(label)
        try:
            result = self.delegate.execute_call(
                tool,
                arguments,
                tool_call_id=tool_call_id,
                execution_context=execution_context,
            )
            if is_write and result.startswith("Wrote "):
                self.trace.complete_write()
            return result
        finally:
            if is_write:
                self.trace.leave(label)


@dataclass
class _InterruptAfterExecuteExecutor:
    delegate: object

    def begin_turn(self):
        return self.delegate.begin_turn()

    def consume_turn_stop_message(self):
        return self.delegate.consume_turn_stop_message()

    def describe_call(self, tool, arguments):
        return self.delegate.describe_call(tool, arguments)

    def execute_call(self, tool, arguments, *, tool_call_id, execution_context=None):
        self.delegate.execute_call(
            tool,
            arguments,
            tool_call_id=tool_call_id,
            execution_context=execution_context,
        )
        raise KeyboardInterrupt("replay interruption after ordinary tool execution")


def _build_long_task_runtime(
    root: Path,
    provider: _ReplayProvider,
    *,
    allow: bool = True,
    resume_session_id: str | None = None,
    tools: list[Tool] | None = None,
    effect_trace: _ToolBodyTrace | None = None,
):
    runtime = RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=root,
        event_sink=CallbackEventSink(lambda _event: None),
        tools=tools or [WriteFileTool(), ReadFileTool(), BashTool(), _OpaqueProbeTool()],
        model="fake-replay",
        permission_prompt=_ReplayAllowPrompt(allow=allow),
        session_directory=root / "sessions",
        resume_session_id=resume_session_id,
    ))
    executor = runtime.agent.tool_executor
    if effect_trace is not None:
        executor = _ObservedEffectExecutor(executor, effect_trace)
    counter = _CountingExecutor(executor)
    runtime.agent.tool_executor = counter
    return runtime, counter


def _write_response(*, file_path: str = "result.txt") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(
        id="write-1",
        name="write_file",
        arguments={"file_path": file_path, "content": "durable result\n"},
    )])


class _OpaqueProbeTool(Tool):
    """A known Tool without effect metadata; Runtime must fail closed to exclusive."""

    name = "opaque_probe"
    description = "Deterministic opaque replay probe."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self) -> str:
        return "opaque-probe"


@contextmanager
def _expect_keyboard_interrupt():
    try:
        yield
    except KeyboardInterrupt:
        return
    raise AssertionError("expected deterministic interruption")


@contextmanager
def _expect_recovery_required():
    try:
        yield
    except LongTaskRecoveryRequired:
        return
    raise AssertionError("expected long Task recovery requirement")


@contextmanager
def _expect_state_error():
    try:
        yield
    except LongTaskStateError:
        return
    raise AssertionError("expected long Task state rejection")


class _EchoTool(Tool):
    name = "echo"
    description = "Return the supplied value for deterministic replay."
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}

    def execute(self, value: str) -> str:
        return f"echo:{value}"


class _RoleTarget:
    def __init__(self) -> None:
        self.activations: list[tuple[str, str, tuple[str, ...]]] = []

    def activate_role(self, role_id: str, role_context: str, *, tool_names: tuple[str, ...] = ()) -> None:
        self.activations.append((role_id, role_context, tool_names))


class _ReplayRoleTool(Tool):
    name = "runtime_probe"
    description = "A test-only Role tool used to verify Runtime isolation."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self) -> str:
        return "runtime-probe"


def _replay_roles() -> tuple[RoleSpec, RoleSpec]:
    return (
        RoleSpec(
            id="runtime-inspector",
            title="Runtime inspector",
            system_prompt="Inspect supplied Runtime evidence only.",
            task_boundary="Verify Runtime facts without changing repository state.",
        ),
        RoleSpec(
            id="runtime-reporter",
            title="Runtime reporter",
            system_prompt="Summarize supplied Runtime evidence only.",
            task_boundary="Summarize Runtime facts without changing repository state.",
        ),
    )


def _build_role_runtime(
    root: Path,
    provider: _ReplayProvider,
    *,
    resume_session_id: str | None = None,
):
    runtime = RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=root,
        event_sink=CallbackEventSink(lambda _event: None),
        tools=[],
        system_context="Base Runtime replay context.",
        session_directory=root / "sessions",
        resume_session_id=resume_session_id,
        model="fake-replay",
    ))
    runtime.role_tool_catalog["runtime_probe"] = _ReplayRoleTool()
    return runtime


def _description_for(effect: str, index: int) -> ToolExecutionDescription:
    resource = (f"resource-{index}",)
    if effect == "r":
        return ToolExecutionDescription(ToolEffect.READ, ToolConcurrency.SAFE, resource)
    if effect == "w":
        return ToolExecutionDescription(ToolEffect.WRITE, ToolConcurrency.EXCLUSIVE, resource)
    if effect == "x":
        return ToolExecutionDescription(ToolEffect.EXECUTE, ToolConcurrency.EXCLUSIVE, resources_known=False)
    if effect == "u":
        return ToolExecutionDescription.unknown()
    raise AssertionError(f"unknown scheduling effect: {effect}")


def _context_messages(marker: str, filler_size: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index in range(12):
        evidence = marker if index == 0 else f"noise-{index}.txt"
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": f"{evidence} {'x' * filler_size}"})
    return messages


def _event(
    event_type: RuntimeEventType,
    session_id: str,
    payload: dict[str, Any],
    *,
    tool_call_id: str | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type,
        session_id=session_id,
        turn_id="replay-turn",
        round_index=1,
        tool_call_id=tool_call_id,
        payload=payload,
    )


def _long_task_provider_responses(script: Mapping[str, Any], phase: str) -> list[LLMResponse]:
    raw_responses = script.get(phase)
    if not isinstance(raw_responses, list):
        raise TypeError(f"private long-task provider_script requires {phase} responses")
    if phase == "initial" and not raw_responses:
        raise TypeError("private long-task provider_script requires non-empty initial responses")
    responses: list[LLMResponse] = []
    for raw_response in raw_responses:
        response = _mapping(raw_response)
        content = response.get("content", "")
        raw_calls = response.get("tool_calls", [])
        if not isinstance(content, str) or not isinstance(raw_calls, list):
            raise TypeError("private long-task provider response is invalid")
        calls: list[ToolCall] = []
        for raw_call in raw_calls:
            call = _mapping(raw_call)
            call_id = call.get("id")
            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, dict):
                raise TypeError("private long-task Tool call is invalid")
            calls.append(ToolCall(call_id, name, dict(arguments)))
        if not content and not calls:
            raise ValueError("private long-task provider response must include content or Tool calls")
        responses.append(LLMResponse(content=content, tool_calls=calls))
    return responses


def _private_case_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if not relative_path or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("private long-task paths must stay inside the case repository")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("private long-task paths must stay inside the case repository") from error
    if resolved == resolved_root:
        raise ValueError("private long-task paths must name a file")
    return resolved


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("replay case requires a JSON object")
    return value


def _mapping_value(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping(value.get(key))


def _string_list(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(entry, str) for entry in item):
        raise TypeError(f"replay case requires a list of strings {key}")
    return tuple(item)


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise TypeError(f"replay case requires boolean {key}")
    return item


def _long_task_holdout_failure_code(error: Exception) -> str:
    """Return only a stable diagnosis class, never a private case value."""

    if isinstance(error, KeyError):
        return "execution_contract_missing_field"
    if isinstance(error, TypeError):
        return "execution_contract_type_mismatch"
    if isinstance(error, ValueError):
        return "execution_contract_value_mismatch"
    message = str(error)
    if message == "private long-task recovery did not stop for an ambiguous effect":
        return "recovery_safety_mismatch"
    if "status mismatch" in message:
        return "assertion_task_status_mismatch"
    if "initial executor count mismatch" in message:
        return "assertion_initial_executor_count_mismatch"
    if "resumed executor count mismatch" in message:
        return "assertion_resume_executor_count_mismatch"
    if "completed effect ledger mismatch" in message:
        return "assertion_completed_effect_ledger_mismatch"
    if "checkpoint cursor" in message or "missing a completed checkpoint" in message:
        return "assertion_session_cursor_mismatch"
    if "file assertion mismatch" in message:
        return "assertion_file_existence_mismatch"
    return "assertion_mismatch"


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise TypeError(f"replay case requires string {key}")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int):
        raise TypeError(f"replay case requires integer {key}")
    return item


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
