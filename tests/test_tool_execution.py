"""C5 scheduling tests: effects constrain concurrency, never safety controls."""

from __future__ import annotations

import threading
import time

import pytest

from techpilot.chat.executor import RepositoryToolExecutor
from techpilot.engine.agent import Agent
from techpilot.engine.events import CallbackEventSink, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.runtime_control import CancellationToken
from techpilot.engine.tool_execution import (
    ToolConcurrency,
    ToolEffect,
    ToolExecutionDescription,
    ToolExecutionPlan,
    resources_conflict,
)
from techpilot.engine.tools.base import Tool


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, messages, tools=None, on_token=None):
        return next(self.responses)


class Timeline:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.records: list[tuple[str, str, float]] = []
        self.started = threading.Event()

    def run(self, name: str, delay: float) -> str:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.records.append(("start", name, time.monotonic()))
            self.started.set()
        time.sleep(delay)
        with self.lock:
            self.active -= 1
            self.records.append(("end", name, time.monotonic()))
        return name


class TimedRead(Tool):
    name = "read_file"
    description = "timed read"
    parameters = {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}

    def __init__(self, timeline: Timeline, delay: float = 0.12) -> None:
        self.timeline = timeline
        self.delay = delay

    def execute(self, file_path: str) -> str:
        return self.timeline.run(file_path, self.delay)


class TimedWrite(Tool):
    name = "write_file"
    description = "timed write"
    parameters = {
        "type": "object",
        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["file_path", "content"],
    }

    def __init__(self, timeline: Timeline, delay: float = 0.08) -> None:
        self.timeline = timeline
        self.delay = delay

    def execute(self, file_path: str, content: str) -> str:
        del content
        return self.timeline.run(file_path, self.delay)


class TimedBash(Tool):
    name = "bash"
    description = "timed command"
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

    def __init__(self, timeline: Timeline, delay: float = 0.08) -> None:
        self.timeline = timeline
        self.delay = delay

    def execute(self, command: str) -> str:
        return self.timeline.run(command, self.delay)


class UnknownTool(Tool):
    name = "mystery"
    description = "no scheduler metadata"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}

    def __init__(self, timeline: Timeline, delay: float = 0.08) -> None:
        self.timeline = timeline
        self.delay = delay

    def execute(self, value: str) -> str:
        return self.timeline.run(value, self.delay)


class DeclaredSafeTool(UnknownTool):
    name = "declared_safe"

    def describe_call(self, arguments):
        return ToolExecutionDescription(
            ToolEffect.READ,
            ToolConcurrency.SAFE,
            affected_resources=(arguments["value"],),
        )


def _agent(calls, tools, *, executor=None, events=None) -> Agent:
    return Agent(
        llm=FakeLLM([LLMResponse(tool_calls=calls), LLMResponse(content="done")]),
        tools=tools,
        tool_executor=executor,
        event_sink=CallbackEventSink(events.append) if events is not None else None,
    )


def test_multiple_reads_overlap_and_beat_a_serial_baseline():
    timeline = Timeline()
    delay = 0.12
    calls = [
        ToolCall("read-a", "read_file", {"file_path": "a.py"}),
        ToolCall("read-b", "read_file", {"file_path": "b.py"}),
    ]
    started = time.monotonic()
    assert _agent(calls, [TimedRead(timeline, delay=delay)]).chat("read") == "done"
    elapsed = time.monotonic() - started

    assert timeline.max_active == 2
    tool_starts = {name: timestamp for kind, name, timestamp in timeline.records if kind == "start"}
    tool_ends = {name: timestamp for kind, name, timestamp in timeline.records if kind == "end"}
    tool_elapsed = max(tool_ends.values()) - min(tool_starts.values())
    serial_baseline = sum(tool_ends[name] - tool_starts[name] for name in tool_starts)
    assert tool_elapsed < serial_baseline  # Concurrent tool work beats the equivalent serial baseline.
    assert elapsed >= tool_elapsed  # The assertion excludes unrelated Agent/Provider overhead.


def test_execution_plan_uses_safe_read_waves_and_conservative_resource_conflicts():
    read = ToolExecutionDescription(ToolEffect.READ, ToolConcurrency.SAFE, ("a.py",))
    write = ToolExecutionDescription(ToolEffect.WRITE, ToolConcurrency.EXCLUSIVE, ("a.py",))
    unknown = ToolExecutionDescription.unknown()

    plan = ToolExecutionPlan.build([read, read, write, read, unknown, read])

    assert [wave.indexes for wave in plan.waves] == [(0, 1), (2,), (3,), (4,), (5,)]
    assert not resources_conflict(read, read)
    assert resources_conflict(read, write)
    assert resources_conflict(write, write)
    assert resources_conflict(read, unknown)


def test_read_and_write_do_not_overlap():
    timeline = Timeline()
    calls = [
        ToolCall("read", "read_file", {"file_path": "before.py"}),
        ToolCall("write", "write_file", {"file_path": "after.py", "content": "x"}),
    ]

    assert _agent(calls, [TimedRead(timeline), TimedWrite(timeline)]).chat("change") == "done"

    assert timeline.max_active == 1
    assert [record[:2] for record in timeline.records] == [
        ("start", "before.py"), ("end", "before.py"), ("start", "after.py"), ("end", "after.py"),
    ]


def test_writes_and_bash_remain_serial():
    timeline = Timeline()
    calls = [
        ToolCall("write-a", "write_file", {"file_path": "a.py", "content": "a"}),
        ToolCall("bash", "bash", {"command": "pytest"}),
        ToolCall("write-b", "write_file", {"file_path": "b.py", "content": "b"}),
    ]

    assert _agent(calls, [TimedWrite(timeline), TimedBash(timeline)]).chat("change") == "done"

    assert timeline.max_active == 1
    assert [record[:2] for record in timeline.records] == [
        ("start", "a.py"), ("end", "a.py"), ("start", "pytest"), ("end", "pytest"),
        ("start", "b.py"), ("end", "b.py"),
    ]


def test_unknown_tools_default_to_serial_execution():
    timeline = Timeline()
    calls = [
        ToolCall("unknown-a", "mystery", {"value": "a"}),
        ToolCall("unknown-b", "mystery", {"value": "b"}),
    ]

    assert _agent(calls, [UnknownTool(timeline)]).chat("unknown") == "done"

    assert timeline.max_active == 1


def test_new_tool_can_declare_safe_read_without_scheduler_changes():
    timeline = Timeline()
    calls = [
        ToolCall("safe-a", "declared_safe", {"value": "a"}),
        ToolCall("safe-b", "declared_safe", {"value": "b"}),
    ]

    assert _agent(calls, [DeclaredSafeTool(timeline)]).chat("read") == "done"

    assert timeline.max_active == 2


def test_results_and_events_follow_model_order_when_reads_finish_out_of_order():
    timeline = Timeline()
    events = []
    calls = [
        ToolCall("slow", "read_file", {"file_path": "slow.py"}),
        ToolCall("fast", "read_file", {"file_path": "fast.py"}),
    ]
    read = TimedRead(timeline)

    def execute(file_path: str) -> str:
        return timeline.run(file_path, 0.12 if file_path == "slow.py" else 0.01)

    read.execute = execute
    agent = _agent(calls, [read], events=events)
    assert agent.chat("read") == "done"

    completed = [event.tool_call_id for event in events if event.event_type is RuntimeEventType.TOOL_COMPLETED]
    messages = [message["tool_call_id"] for message in agent.messages if message.get("role") == "tool"]
    assert [record[1] for record in timeline.records if record[0] == "end"] == ["fast.py", "slow.py"]
    assert completed == ["slow", "fast"]
    assert messages == ["slow", "fast"]


class EmittingExecutor:
    def describe_call(self, tool, arguments):
        return ToolExecutionDescription(ToolEffect.READ, ToolConcurrency.SAFE, (arguments["value"],))

    def execute_call(self, tool, arguments, *, tool_call_id, execution_context):
        del tool
        value = arguments["value"]
        time.sleep(0.10 if value == "slow" else 0.01)
        execution_context.emit(
            RuntimeEventType.EXECUTION_CONTROL_ASSESSED,
            tool_call_id=tool_call_id,
            payload={"value": value},
        )
        return value


class ExecutorReadTool(Tool):
    name = "executor_read"
    description = "read controlled by the executor"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}

    def execute(self, value: str) -> str:
        return value


def test_executor_events_are_buffered_and_published_in_model_order():
    events = []
    calls = [
        ToolCall("slow", "executor_read", {"value": "slow"}),
        ToolCall("fast", "executor_read", {"value": "fast"}),
    ]
    agent = _agent(calls, [ExecutorReadTool()], executor=EmittingExecutor(), events=events)

    assert agent.chat("read") == "done"

    assessed = [event.tool_call_id for event in events if event.event_type is RuntimeEventType.EXECUTION_CONTROL_ASSESSED]
    completed = [event.tool_call_id for event in events if event.event_type is RuntimeEventType.TOOL_COMPLETED]
    assert assessed == ["slow", "fast"]
    assert completed == ["slow", "fast"]


def test_repository_executor_describes_effect_resources_and_cwd(tmp_path):
    executor = RepositoryToolExecutor(tmp_path)

    read = executor.describe_call(TimedRead(Timeline()), {"file_path": "src/main.py"})
    write = executor.describe_call(TimedWrite(Timeline()), {"file_path": "src/main.py", "content": "x"})
    command = executor.describe_call(TimedBash(Timeline()), {"command": "pytest"})
    unknown = executor.describe_call(UnknownTool(Timeline()), {"value": "x"})

    assert read == ToolExecutionDescription(
        ToolEffect.READ,
        ToolConcurrency.SAFE,
        (str((tmp_path / "src" / "main.py").resolve()),),
        str(tmp_path.resolve()),
    )
    assert write is not None and write.effect is ToolEffect.WRITE and write.concurrency is ToolConcurrency.EXCLUSIVE
    assert command is not None and command.effect is ToolEffect.EXECUTE and command.resources_known is False
    assert unknown is None


class StoppingExecutor:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.calls: list[str] = []
        self.stop_message: str | None = None

    def execute_call(self, tool, arguments, *, tool_call_id):
        del arguments, tool_call_id
        self.calls.append(tool.name)
        self.stop_message = self.reason
        return f"Policy denied {tool.name}: {self.reason}"

    def consume_turn_stop_message(self):
        message = self.stop_message
        self.stop_message = None
        return message


@pytest.mark.parametrize("reason", ["blocked by policy", "user denied the operation"])
def test_block_or_rejection_does_not_start_later_side_effect_calls(reason):
    calls = [
        ToolCall("blocked", "bash", {"command": "danger"}),
        ToolCall("never-write", "write_file", {"file_path": "never.py", "content": "x"}),
    ]
    executor = StoppingExecutor(reason)
    agent = _agent(calls, [TimedBash(Timeline()), TimedWrite(Timeline())], executor=executor)

    assert agent.chat("change") == reason
    assert executor.calls == ["bash"]
    assert [message["content"] for message in agent.messages if message.get("role") == "tool"][-1] == (
        "[not executed: an earlier call stopped this turn]"
    )


def test_cancellation_does_not_start_the_next_side_effect_and_keeps_tool_replies():
    timeline = Timeline()
    calls = [
        ToolCall("read", "read_file", {"file_path": "slow.py"}),
        ToolCall("write", "write_file", {"file_path": "never.py", "content": "x"}),
    ]
    agent = _agent(calls, [TimedRead(timeline, delay=0.15), TimedWrite(timeline)])
    token = CancellationToken()
    result: list[str] = []
    worker = threading.Thread(target=lambda: result.append(agent.chat("change", cancellation_token=token)))
    worker.start()
    assert timeline.started.wait(timeout=1)
    token.cancel("test cancellation")
    worker.join(timeout=2)

    assert result == ["(turn cancelled: test cancellation)"]
    assert all(record[1] != "never.py" for record in timeline.records)
    assert [message["tool_call_id"] for message in agent.messages if message.get("role") == "tool"] == ["read", "write"]
    assert [message["content"] for message in agent.messages if message.get("role") == "tool"][-1] == "[cancelled before execution]"
