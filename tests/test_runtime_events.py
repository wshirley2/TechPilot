"""C1 Runtime Event contract and AgentLoop integration tests."""

from __future__ import annotations

import json

import pytest

from techpilot.engine.agent import Agent
from techpilot.engine.events import CallbackEventSink, RuntimeEvent, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.tools.base import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Return a value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, value: str) -> str:
        return f"echo: {value}"


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, messages, tools=None, on_token=None):
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if on_token and response.content:
            on_token(response.content)
        return response


def collect_events():
    events = []
    return events, CallbackEventSink(events.append)


def event_types(events):
    return [event.event_type for event in events]


def test_runtime_event_is_immutable_and_json_serializable():
    event = RuntimeEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED,
        session_id="session-1",
        turn_id="turn-1",
        round_index=2,
        tool_call_id="call-1",
        payload={"tool_name": "echo", "arguments": {"value": "hi"}},
    )

    encoded = json.dumps(event.to_dict())
    assert '"event_type": "tool_requested"' in encoded
    assert event.to_dict()["tool_call_id"] == "call-1"
    with pytest.raises(AttributeError):
        event.turn_id = "changed"


def test_text_response_emits_one_stable_turn_sequence_and_keeps_on_token():
    events, sink = collect_events()
    tokens = []
    agent = Agent(
        llm=FakeLLM([LLMResponse(content="done", prompt_tokens=3, completion_tokens=1)]),
        tools=[],
        event_sink=sink,
        session_id="session-text",
    )

    assert agent.chat("hello", on_token=tokens.append) == "done"

    assert event_types(events) == [
        RuntimeEventType.TURN_STARTED,
        RuntimeEventType.PROVIDER_STARTED,
        RuntimeEventType.ASSISTANT_TOKEN,
        RuntimeEventType.TURN_COMPLETED,
    ]
    assert {event.turn_id for event in events} == {events[0].turn_id}
    assert {event.session_id for event in events} == {"session-text"}
    assert tokens == ["done"]


@pytest.mark.parametrize("finish_reason,use_config_fallback", [("length", False), (None, True)])
def test_empty_exhausted_output_is_a_limit_not_a_completed_turn(finish_reason, use_config_fallback):
    events, sink = collect_events()
    llm = FakeLLM([LLMResponse(completion_tokens=4, finish_reason=finish_reason)])
    if use_config_fallback:
        llm.extra = {"max_tokens": 4}
    agent = Agent(llm=llm, tools=[], event_sink=sink)

    assert agent.chat("continue") == "(runtime limit reached: output_tokens)"
    assert events[-1].event_type is RuntimeEventType.TURN_LIMIT_REACHED
    assert events[-1].payload["limit"] == "output_tokens"
    assert events[-1].payload["pending_tool_call_ids"] == []
    assert RuntimeEventType.TURN_COMPLETED not in event_types(events)


def test_single_tool_events_keep_provider_tool_call_id_and_old_on_tool_callback():
    events, sink = collect_events()
    tool_callbacks = []
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[ToolCall("call-1", "echo", {"value": "one"})]),
            LLMResponse(content="finished"),
        ]),
        tools=[EchoTool()],
        event_sink=sink,
    )

    assert agent.chat("run", on_tool=lambda name, args: tool_callbacks.append((name, args))) == "finished"

    requested = [event for event in events if event.event_type is RuntimeEventType.TOOL_REQUESTED]
    completed = [event for event in events if event.event_type is RuntimeEventType.TOOL_COMPLETED]
    assert [event.tool_call_id for event in requested] == ["call-1"]
    assert [event.tool_call_id for event in completed] == ["call-1"]
    assert completed[0].payload["result"] == "echo: one"
    assert tool_callbacks == [("echo", {"value": "one"})]


def test_multiple_tool_events_and_history_keep_provider_order():
    events, sink = collect_events()
    calls = [
        ToolCall("call-a", "echo", {"value": "a"}),
        ToolCall("call-b", "echo", {"value": "b"}),
    ]
    agent = Agent(
        llm=FakeLLM([LLMResponse(tool_calls=calls), LLMResponse(content="done")]),
        tools=[EchoTool()],
        event_sink=sink,
    )

    agent.chat("run both")

    requested = [event for event in events if event.event_type is RuntimeEventType.TOOL_REQUESTED]
    completed = [event for event in events if event.event_type is RuntimeEventType.TOOL_COMPLETED]
    tool_messages = [message for message in agent.messages if message.get("role") == "tool"]
    assert [event.tool_call_id for event in requested] == ["call-a", "call-b"]
    assert {event.tool_call_id for event in completed} == {"call-a", "call-b"}
    assert [message["tool_call_id"] for message in tool_messages] == ["call-a", "call-b"]


def test_interrupt_backfills_tool_result_and_emits_interrupted_event():
    events, sink = collect_events()

    class InterruptTool(EchoTool):
        def execute(self, value: str) -> str:
            raise KeyboardInterrupt

    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[ToolCall("call-stop", "echo", {"value": "stop"})]),
        ]),
        tools=[InterruptTool()],
        event_sink=sink,
    )

    with pytest.raises(KeyboardInterrupt):
        agent.chat("interrupt")

    tool_messages = [message for message in agent.messages if message.get("role") == "tool"]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call-stop", "content": "[interrupted]"},
    ]
    assert event_types(events)[-2:] == [
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.TURN_INTERRUPTED,
    ]
    assert events[-1].payload["pending_tool_call_ids"] == ["call-stop"]


def test_provider_failure_and_round_limit_have_terminal_events():
    failed_events, failed_sink = collect_events()
    failing = Agent(llm=FakeLLM([RuntimeError("provider down")]), tools=[], event_sink=failed_sink)
    with pytest.raises(RuntimeError, match="provider down"):
        failing.chat("fail")
    assert event_types(failed_events)[-1] is RuntimeEventType.TURN_FAILED
    assert failed_events[-1].payload["error_type"] == "RuntimeError"

    limit_events, limit_sink = collect_events()
    limited = Agent(
        llm=FakeLLM([LLMResponse(tool_calls=[ToolCall("call-1", "echo", {"value": "x"})])]),
        tools=[EchoTool()],
        max_rounds=1,
        event_sink=limit_sink,
    )
    assert limited.chat("loop") == "(reached maximum tool-call rounds)"
    assert event_types(limit_events)[-1] is RuntimeEventType.TURN_LIMIT_REACHED


def test_context_compression_and_sink_failure_do_not_break_agent(monkeypatch):
    events, sink = collect_events()
    agent = Agent(llm=FakeLLM([LLMResponse(content="ok")]), tools=[], event_sink=sink)
    monkeypatch.setattr(agent.context, "maybe_compress", lambda messages, llm: True)
    assert agent.chat("compress") == "ok"
    assert RuntimeEventType.CONTEXT_COMPRESSED in event_types(events)

    class BrokenSink:
        def emit(self, event):
            raise RuntimeError("observer failed")

    unaffected = Agent(llm=FakeLLM([LLMResponse(content="still works")]), tools=[], event_sink=BrokenSink())
    assert unaffected.chat("hello") == "still works"
