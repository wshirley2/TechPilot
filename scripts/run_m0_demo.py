"""Run three deterministic, no-provider-cost M0 demonstration paths.

The script uses short fixed Provider responses so the actual Runtime, Tool
executor, Permission path, Session persistence, and LongTaskWorkflow can be
shown without presenting a model sample as a Runtime guarantee.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techpilot.chat.session import TerminalEventSink
from techpilot.engine.events import CallbackEventSink
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.permissions import PermissionDecision
from techpilot.engine.tools import WriteFileTool
from techpilot.runtime import LongTaskStatus, LongTaskWorkflow, RuntimeBootstrap, RuntimeBootstrapInput


class FixedProvider:
    """Finite provider transcript used only by this deterministic demo."""

    model = "m0-demo-fixed-provider"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, on_token=None) -> LLMResponse:
        self.requests.append({"messages": messages, "tools": tools})
        response = next(self._responses)
        if on_token and response.content:
            on_token(response.content)
        return response


class AllowPrompt:
    def decide(self, request):
        del request
        return PermissionDecision.allow("M0 deterministic demo approval")


class DenyPrompt:
    def decide(self, request):
        del request
        return PermissionDecision.deny("M0 deterministic demo rejection")


@dataclass
class CountingExecutor:
    """Count underlying Tool execution so recovery can prove skip behavior."""

    delegate: object
    calls: int = 0

    def begin_turn(self):
        return self.delegate.begin_turn()  # type: ignore[attr-defined]

    def consume_turn_stop_message(self):
        return self.delegate.consume_turn_stop_message()  # type: ignore[attr-defined]

    def describe_call(self, tool, arguments):
        return self.delegate.describe_call(tool, arguments)  # type: ignore[attr-defined]

    def execute_call(self, tool, arguments, *, tool_call_id, execution_context=None):
        self.calls += 1
        return self.delegate.execute_call(  # type: ignore[attr-defined]
            tool,
            arguments,
            tool_call_id=tool_call_id,
            execution_context=execution_context,
        )


def _build_runtime(
    repository: Path,
    provider: FixedProvider,
    *,
    permission_prompt,
    session_id: str | None = None,
    write_only: bool = False,
):
    sink = CallbackEventSink(lambda _event: None) if write_only else TerminalEventSink(
        Console(file=StringIO(), force_terminal=False, color_system=None, width=120)
    )
    runtime = RuntimeBootstrap(provider_factory=lambda _config: provider).build(RuntimeBootstrapInput(
        repository=repository,
        event_sink=sink,
        tools=[WriteFileTool()] if write_only else None,
        model=FixedProvider.model,
        permission_prompt=permission_prompt,
        session_directory=repository / "sessions",
        resume_session_id=session_id,
    ))
    if write_only:
        runtime.agent.tool_executor = CountingExecutor(runtime.agent.tool_executor)
    return runtime


def _demo_success(root: Path) -> dict[str, object]:
    repository = root / "success"
    repository.mkdir()
    source = repository / "app.py"
    source.write_text("def greet(name: str) -> str:\n    return f'hi {name}'\n", encoding="utf-8")
    (repository / "test_app.py").write_text(
        "from app import greet\n\n\ndef test_greet_uses_hello_prefix():\n    assert greet('Ada') == 'hello Ada'\n",
        encoding="utf-8",
    )
    old = "return f'hi {name}'"
    new = "return f'hello {name}'"
    validation = subprocess.list2cmdline([
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ])
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("read-app", "read_file", {"file_path": "app.py"})]),
        LLMResponse(tool_calls=[ToolCall("edit-app", "edit_file", {
            "file_path": "app.py", "old_string": old, "new_string": new,
        })]),
        LLMResponse(tool_calls=[ToolCall("validate-app", "bash", {"command": validation})]),
        LLMResponse(content="修改与验证完成。"),
    ])
    runtime = _build_runtime(repository, provider, permission_prompt=AllowPrompt())
    response = runtime.run_turn("将 greet 前缀改为 hello 并验证")
    assert new in source.read_text(encoding="utf-8")
    assert response
    tool_results = {
        message["tool_call_id"]: message["content"]
        for message in runtime.agent.messages
        if message.get("role") == "tool"
    }
    assert set(tool_results) == {"read-app", "edit-app", "validate-app"}
    assert "1 passed" in tool_results["validate-app"], repr(tool_results["validate-app"])
    return {
        "scenario": "success_read_edit_validate",
        "passed": True,
        "provider_requests": len(provider.requests),
        "validated_file": "app.py",
    }


def _demo_rejection(root: Path) -> dict[str, object]:
    repository = root / "rejection"
    repository.mkdir()
    target = repository / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    provider = FixedProvider([
        LLMResponse(tool_calls=[ToolCall("write-denied", "edit_file", {
            "file_path": "notes.txt", "old_string": "original", "new_string": "changed",
        })]),
    ])
    runtime = _build_runtime(repository, provider, permission_prompt=DenyPrompt())
    response = runtime.run_turn("修改 notes.txt")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert len(provider.requests) == 1
    assert "没有执行" in response
    return {
        "scenario": "rejected_write_stops_turn",
        "passed": True,
        "provider_requests": len(provider.requests),
        "target_unchanged": True,
    }


def _write_call(call_id: str = "write-result") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(
        id=call_id,
        name="write_file",
        arguments={"file_path": "result.txt", "content": "durable result\n"},
    )])


def _demo_recovery(root: Path) -> dict[str, object]:
    repository = root / "recovery"
    repository.mkdir()
    first = _build_runtime(repository, FixedProvider([_write_call()]), permission_prompt=AllowPrompt(), write_only=True)
    first_counter = first.agent.tool_executor

    def interrupt_after_completion(phase: str, action_id: str) -> None:
        assert action_id == "tool-write-result"
        if phase == "after_effect_completed":
            raise KeyboardInterrupt("M0 deterministic interruption after durable completion")

    try:
        LongTaskWorkflow(
            first,
            task_id="m0-recovery-demo",
            goal="write durable result",
            fault_hook=interrupt_after_completion,
        ).run_turn("write the result")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("demo fault hook did not interrupt")

    resumed = _build_runtime(
        repository,
        FixedProvider([_write_call(), LLMResponse(content="恢复完成。")]),
        permission_prompt=AllowPrompt(),
        session_id=first.agent.session_id,
        write_only=True,
    )
    resumed_counter = resumed.agent.tool_executor
    result = LongTaskWorkflow(resumed, task_id="m0-recovery-demo", goal="write durable result").run_turn("continue")
    assert result.task.status is LongTaskStatus.SUCCEEDED
    assert result.response == "恢复完成。"
    assert first_counter.calls == 1
    assert resumed_counter.calls == 0
    assert (repository / "result.txt").read_text(encoding="utf-8") == "durable result\n"
    return {
        "scenario": "long_task_recovery_skips_completed_effect",
        "passed": True,
        "initial_tool_calls": first_counter.calls,
        "resumed_tool_calls": resumed_counter.calls,
    }


def run_demo() -> dict[str, object]:
    """Execute all three paths in a disposable repository tree."""

    with tempfile.TemporaryDirectory(prefix="techpilot-m0-demo-") as temporary_directory:
        root = Path(temporary_directory)
        scenarios = [_demo_success(root), _demo_rejection(root), _demo_recovery(root)]
    return {
        "suite": "m0-deterministic-demo-v1",
        "model_calls": 0,
        "scenarios": scenarios,
        "passed": sum(bool(scenario["passed"]) for scenario in scenarios),
        "total": len(scenarios),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TechPilot's deterministic M0 success/rejection/recovery demo.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = parser.parse_args(argv)
    result = run_demo()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"M0 deterministic demo: {result['passed']}/{result['total']} passed; model_calls=0")
        for scenario in result["scenarios"]:
            print(f"- {scenario['scenario']}: passed")
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
