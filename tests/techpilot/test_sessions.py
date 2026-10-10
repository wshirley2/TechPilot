"""C4 event Session storage, replay, recovery, and runtime limit tests."""

from __future__ import annotations

import json

import pytest

from techpilot.engine.agent import Agent
from techpilot.engine.events import CallbackEventSink, RuntimeEvent, RuntimeEventType
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.runtime_control import CancellationToken, RuntimeLimits
from techpilot.engine.tools.base import Tool
from techpilot.runtime.bootstrap import RuntimeBootstrap, RuntimeBootstrapInput
from techpilot.runtime.contracts import RuntimeResultStatus, TaskRuntimeResult
from techpilot.runtime.sessions import SessionEvent, SessionEventSink, SessionStore, SessionStoreError


class EchoTool(Tool):
    name = "echo"
    description = "Return a value."
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

    def execute(self, value: str) -> str:
        return f"echo: {value}"


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.estimated_cost = None

    def chat(self, messages, tools=None, on_token=None):
        self.requests.append({"messages": messages, "tools": tools})
        response = next(self.responses)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        if on_token and response.content:
            on_token(response.content)
        return response


def _event(event_type, session_id="session-1", **kwargs):
    return RuntimeEvent(event_type=event_type, session_id=session_id, turn_id="turn-1", round_index=1, **kwargs)


def test_session_sink_persistence_check_reaches_the_downstream_event_log(tmp_path):
    class DurableDownstream:
        checked = False

        def emit(self, event):
            pass

        def ensure_persisted(self):
            self.checked = True

    downstream = DurableDownstream()
    sink = SessionEventSink(SessionStore(tmp_path / "sessions"), downstream)

    sink.ensure_persisted()

    assert downstream.checked


def test_session_sink_forwards_live_event_but_requires_an_explicit_durability_check(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    observed = []
    sink = SessionEventSink(store, CallbackEventSink(observed.append))

    def fail_to_append(_event) -> None:
        raise OSError("simulated session write failure")

    monkeypatch.setattr(store, "append_runtime", fail_to_append)
    event = _event(RuntimeEventType.TOOL_COMPLETED, tool_call_id="read-1", payload={
        "tool_name": "read_file",
        "result": "README.md",
        "interrupted": False,
    })

    sink.emit(event)

    assert observed == [event]
    with pytest.raises(SessionStoreError, match="latest turn may not be recoverable"):
        sink.ensure_persisted()


def test_session_replay_rebuilds_valid_tool_history_and_ignores_partial_tail(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-1", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(RuntimeEventType.TURN_STARTED, payload={"user_input": "inspect README"}))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_REQUESTED,
        tool_call_id="read-1",
        payload={
            "tool_name": "read_file",
            "arguments": {"file_path": "README.md"},
            "assistant_content": "Reading the README.",
        },
    ))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_COMPLETED,
        tool_call_id="read-1",
        payload={"tool_name": "read_file", "result": "contents", "interrupted": False},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TURN_COMPLETED,
        payload={"content": "README inspected", "prompt_tokens": 3, "completion_tokens": 2},
    ))

    path = store.directory / "session-1.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":1')

    projection = store.replay("session-1")

    assert [message["role"] for message in projection.messages] == ["user", "assistant", "tool", "assistant"]
    assert projection.messages[1]["content"] == "Reading the README."
    assert projection.messages[1]["tool_calls"][0]["id"] == "read-1"
    assert projection.messages[2] == {"role": "tool", "tool_call_id": "read-1", "content": "contents"}
    assert projection.messages[-1]["content"] == "README inspected"
    assert projection.model_messages == projection.messages
    assert projection.total_tokens == 5
    assert projection.last_result == TaskRuntimeResult(
        status=RuntimeResultStatus.SUCCEEDED,
        response="README inspected",
    )
    assert projection.warnings == ["Ignored incomplete trailing Session event"]


def test_session_replay_excludes_orphan_tool_result_from_model_history_after_middle_corruption(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-gap", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(RuntimeEventType.TURN_STARTED, "session-gap", payload={"user_input": "inspect README"}))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_REQUESTED,
        "session-gap",
        tool_call_id="lost-request",
        payload={"tool_name": "read_file", "arguments": {"file_path": "README.md"}, "assistant_content": ""},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_COMPLETED,
        "session-gap",
        tool_call_id="lost-request",
        payload={"tool_name": "read_file", "result": "README contents", "interrupted": False},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TURN_COMPLETED,
        "session-gap",
        payload={"content": "inspection complete"},
    ))

    path = store.path_for("session-gap")
    lines = path.read_bytes().splitlines(keepends=True)
    request_index = next(index for index, line in enumerate(lines) if b'"tool_call_id":"lost-request"' in line)
    lines[request_index] = b'{"event":"corrupt middle record"\n'
    path.write_bytes(b"".join(lines))

    projection = store.replay("session-gap")

    assert len(projection.events) == 4
    assert any("Ignored invalid Session event on line" in warning for warning in projection.warnings)
    assert any("Excluded orphan tool result 'lost-request'" in warning for warning in projection.warnings)
    assert {message["role"] for message in projection.messages} == {"user", "tool", "assistant"}
    assert projection.model_messages == [
        {"role": "user", "content": "inspect README"},
        {"role": "assistant", "content": "inspection complete"},
    ]

    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    llm = FakeLLM([LLMResponse(content="continued safely")])
    runtime = RuntimeBootstrap(provider_factory=lambda _config: llm).build(RuntimeBootstrapInput(
        repository=tmp_path,
        event_sink=CallbackEventSink(lambda _event: None),
        session_directory=store.directory,
        resume_session_id="session-gap",
        tools=[],
    ))
    assert runtime.agent.session_id == "session-gap"
    assert runtime.run_turn("continue") == "continued safely"
    assert all(message["role"] != "tool" for message in llm.requests[0]["messages"])


@pytest.mark.parametrize(("field", "replacement", "warning_fragment"), [
    ("session_id", "another-session", "belongs to"),
    ("event_id", None, "requires a durable event_id"),
    ("event_id", "duplicate", "Duplicate Session event_id"),
    ("round_index", "1", "round_index must be a non-negative integer"),
    ("turn_id", None, "Tool event requires turn_id and tool_call_id"),
    ("tool_call_id", None, "Tool event requires turn_id and tool_call_id"),
    ("tool_call_id", 17, "tool_call_id must be a non-empty string or null"),
    ("payload", {"tool_name": 17, "arguments": {}}, "Tool request requires a tool_name"),
    ("payload", {"tool_name": "read_file", "arguments": "bad"}, "Tool request arguments must be an object"),
])
def test_session_replay_skips_invalid_tool_request_identity_without_rewriting_events(
    tmp_path, field, replacement, warning_fragment,
):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-identity", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(
        RuntimeEventType.TURN_STARTED,
        "session-identity",
        payload={"user_input": "inspect README"},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_REQUESTED,
        "session-identity",
        tool_call_id="read-identity",
        payload={"tool_name": "read_file", "arguments": {"file_path": "README.md"}},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_COMPLETED,
        "session-identity",
        tool_call_id="read-identity",
        payload={"tool_name": "read_file", "result": "README contents"},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TURN_COMPLETED,
        "session-identity",
        payload={"content": "inspection finished"},
    ))
    path = store.path_for("session-identity")
    lines = path.read_bytes().splitlines(keepends=True)
    request = json.loads(lines[2])
    if replacement is None and field == "event_id":
        request.pop(field)
    elif replacement == "duplicate":
        request[field] = json.loads(lines[0])["event_id"]
    else:
        request[field] = replacement
    lines[2] = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
    original = b"".join(lines)
    path.write_bytes(original)

    projection = store.replay("session-identity")

    assert path.read_bytes() == original
    assert len(projection.events) == 4
    assert any(warning_fragment in warning for warning in projection.warnings)
    assert any("Excluded orphan tool result 'read-identity'" in warning for warning in projection.warnings)
    assert projection.model_messages == [
        {"role": "user", "content": "inspect README"},
        {"role": "assistant", "content": "inspection finished"},
    ]


def test_session_replay_accepts_legacy_runtime_event_without_schema_version(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("legacy-session", repository_root=tmp_path, model="fake-model")
    legacy_event = _event(
        RuntimeEventType.TURN_STARTED,
        "legacy-session",
        payload={"user_input": "legacy input"},
    ).to_dict()
    with store.path_for("legacy-session").open("ab") as stream:
        stream.write((json.dumps(legacy_event) + "\n").encode("utf-8"))

    projection = store.replay("legacy-session")

    assert projection.warnings == []
    assert projection.events[-1].schema_version == 1
    assert projection.model_messages == [{"role": "user", "content": "legacy input"}]


@pytest.mark.parametrize(("field", "replacement", "warning_fragment"), [
    ("session_id", "different-session", "belongs to"),
    ("repository_root", "", "requires an absolute repository_root"),
    ("repository_root", ".", "requires an absolute repository_root"),
])
def test_runtime_refuses_to_resume_session_without_valid_creation_identity(
    tmp_path, monkeypatch, field, replacement, warning_fragment,
):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    store = SessionStore(tmp_path / "sessions")
    store.create("missing-anchor", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(
        RuntimeEventType.TURN_STARTED,
        "missing-anchor",
        payload={"user_input": "surviving message"},
    ))
    path = store.path_for("missing-anchor")
    lines = path.read_bytes().splitlines(keepends=True)
    created = json.loads(lines[0])
    if field == "repository_root":
        created["payload"][field] = replacement
    else:
        created[field] = replacement
    lines[0] = (json.dumps(created) + "\n").encode("utf-8")
    path.write_bytes(b"".join(lines))

    projection = store.replay("missing-anchor")
    assert projection.repository_root is None
    assert projection.model_messages == [{"role": "user", "content": "surviving message"}]
    assert any(warning_fragment in warning for warning in projection.warnings)

    bootstrap = RuntimeBootstrap(provider_factory=lambda _config: FakeLLM([]))
    runtime = bootstrap.build(RuntimeBootstrapInput(
        repository=tmp_path,
        event_sink=CallbackEventSink(lambda _event: None),
        session_directory=store.directory,
        tools=[],
    ))
    with pytest.raises(ValueError, match="no valid creation identity"):
        runtime.resume_session("missing-anchor")
    with pytest.raises(ValueError, match="no valid creation identity"):
        bootstrap.build(RuntimeBootstrapInput(
            repository=tmp_path,
            event_sink=CallbackEventSink(lambda _event: None),
            session_directory=store.directory,
            resume_session_id="missing-anchor",
            tools=[],
        ))


def test_session_replay_does_not_accept_later_creation_as_identity(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("late-anchor", repository_root=tmp_path, model="fake-model")
    path = store.path_for("late-anchor")
    original = path.read_bytes()
    path.write_bytes(b"{invalid first record}\n" + original)

    projection = store.replay("late-anchor")

    assert projection.repository_root is None
    assert len(projection.warnings) == 2
    assert any("first record" in warning for warning in projection.warnings)


def test_session_replay_uses_compression_checkpoint_after_a_middle_gap(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-checkpoint", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(RuntimeEventType.TURN_STARTED, "session-checkpoint", payload={"user_input": "first"}))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_REQUESTED,
        "session-checkpoint",
        tool_call_id="lost-request",
        payload={"tool_name": "read_file", "arguments": {"file_path": "README.md"}, "assistant_content": ""},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_COMPLETED,
        "session-checkpoint",
        tool_call_id="lost-request",
        payload={"tool_name": "read_file", "result": "old contents", "interrupted": False},
    ))
    store.append(SessionEvent(
        event_type=RuntimeEventType.CONTEXT_COMPRESSED.value,
        session_id="session-checkpoint",
        payload={"message_projection": [{"role": "user", "content": "[summary] verified history"}]},
    ))
    store.append_runtime(_event(RuntimeEventType.TURN_STARTED, "session-checkpoint", payload={"user_input": "read current README"}))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_REQUESTED,
        "session-checkpoint",
        tool_call_id="read-current",
        payload={"tool_name": "read_file", "arguments": {"file_path": "README.md"}, "assistant_content": ""},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_COMPLETED,
        "session-checkpoint",
        tool_call_id="read-current",
        payload={"tool_name": "read_file", "result": "current contents", "interrupted": False},
    ))

    path = store.path_for("session-checkpoint")
    lines = path.read_bytes().splitlines(keepends=True)
    request_index = next(index for index, line in enumerate(lines) if b'"tool_call_id":"lost-request"' in line)
    lines[request_index] = b'{"event":"corrupt middle record"\n'
    path.write_bytes(b"".join(lines))

    projection = store.replay("session-checkpoint")

    assert any("Excluded orphan tool result 'lost-request'" in warning for warning in projection.warnings)
    assert projection.model_messages == [
        {"role": "user", "content": "[summary] verified history"},
        {"role": "user", "content": "read current README"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "read-current",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"file_path": "README.md"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "read-current", "content": "current contents"},
    ]


def test_session_keeps_raw_facts_when_context_projection_is_compressed(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-2", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(RuntimeEventType.TURN_STARTED, "session-2", payload={"user_input": "first"}))
    store.append_runtime(_event(RuntimeEventType.TURN_COMPLETED, "session-2", payload={"content": "first answer"}))
    store.append(SessionEvent(
        event_type=RuntimeEventType.CONTEXT_COMPRESSED.value,
        session_id="session-2",
        payload={"message_projection": [{"role": "user", "content": "[summary] first"}]},
    ))

    projection = store.replay("session-2")

    assert [message["content"] for message in projection.messages] == ["first", "first answer"]
    assert projection.model_messages == [{"role": "user", "content": "[summary] first"}]


def test_session_replays_messages_after_context_projection_checkpoint(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-2", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(RuntimeEventType.TURN_STARTED, "session-2", payload={"user_input": "first"}))
    store.append(SessionEvent(
        event_type=RuntimeEventType.CONTEXT_COMPRESSED.value,
        session_id="session-2",
        payload={"message_projection": [{"role": "user", "content": "[summary] first"}]},
    ))
    store.append_runtime(_event(RuntimeEventType.TURN_COMPLETED, "session-2", payload={"content": "first answer"}))
    store.append_runtime(_event(RuntimeEventType.TURN_STARTED, "session-2", payload={"user_input": "read README"}))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_REQUESTED,
        "session-2",
        tool_call_id="read-2",
        payload={"tool_name": "read_file", "arguments": {"file_path": "README.md"}, "assistant_content": ""},
    ))
    store.append_runtime(_event(
        RuntimeEventType.TOOL_COMPLETED,
        "session-2",
        tool_call_id="read-2",
        payload={"tool_name": "read_file", "result": "contents", "interrupted": False},
    ))

    projection = store.replay("session-2")

    assert projection.model_messages == [
        {"role": "user", "content": "[summary] first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "read README"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "read-2",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"file_path": "README.md"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "read-2", "content": "contents"},
    ]


def test_session_sink_persists_agent_events_and_limits_leave_recoverable_boundary(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-3", repository_root=tmp_path, model="fake-model")
    observed = []
    sink = SessionEventSink(store, CallbackEventSink(observed.append))
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[ToolCall("echo-1", "echo", {"value": "one"})]),
            LLMResponse(content="this provider call must not happen"),
        ]),
        tools=[EchoTool()],
        event_sink=sink,
        session_id="session-3",
        limits=RuntimeLimits(max_provider_calls=1),
    )

    result = agent.chat("run once")
    projection = store.replay("session-3")

    assert result == "(runtime limit reached: provider_calls)"
    assert projection.status == "limit_reached"
    assert projection.last_result is not None
    assert projection.last_result.status is RuntimeResultStatus.LIMIT_REACHED
    assert projection.last_result.limit == "provider_calls"
    assert projection.messages[-1] == {"role": "tool", "tool_call_id": "echo-1", "content": "echo: one"}
    assert observed[-1].event_type is RuntimeEventType.TURN_LIMIT_REACHED


def test_pre_cancelled_token_records_interruption_without_provider_call():
    events = []
    token = CancellationToken()
    token.cancel("test cancellation")
    agent = Agent(
        llm=FakeLLM([LLMResponse(content="must not run")]),
        tools=[],
        event_sink=CallbackEventSink(events.append),
    )

    assert agent.chat("cancel", cancellation_token=token) == "(turn cancelled: test cancellation)"
    assert [event.event_type for event in events][-1] is RuntimeEventType.TURN_INTERRUPTED


def test_total_token_limit_stops_after_provider_response_with_explicit_reason():
    events = []
    agent = Agent(
        llm=FakeLLM([LLMResponse(content="too expensive", prompt_tokens=2, completion_tokens=2)]),
        tools=[],
        event_sink=CallbackEventSink(events.append),
        limits=RuntimeLimits(max_total_tokens=3),
    )

    assert agent.chat("limit tokens") == "(runtime limit reached: total_tokens)"
    assert events[-1].event_type is RuntimeEventType.TURN_LIMIT_REACHED
    assert events[-1].payload["limit"] == "total_tokens"


def test_tool_round_limit_backfills_requested_calls_for_a_valid_resumable_history(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("session-4", repository_root=tmp_path, model="fake-model")
    sink = SessionEventSink(store, CallbackEventSink(lambda event: None))
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[ToolCall("echo-1", "echo", {"value": "one"})]),
            LLMResponse(tool_calls=[ToolCall("echo-2", "echo", {"value": "two"})]),
        ]),
        tools=[EchoTool()],
        event_sink=sink,
        session_id="session-4",
        limits=RuntimeLimits(max_tool_rounds=1),
    )

    assert agent.chat("run twice") == "(runtime limit reached: tool_rounds)"
    projection = store.replay("session-4")

    assert projection.status == "limit_reached"
    assert projection.last_result is not None
    assert projection.last_result.status is RuntimeResultStatus.LIMIT_REACHED
    assert projection.last_result.limit == "tool_rounds"
    assert projection.messages[-2]["role"] == "assistant"
    assert projection.messages[-2]["tool_calls"][0]["id"] == "echo-2"
    assert projection.messages[-1] == {
        "role": "tool",
        "tool_call_id": "echo-2",
        "content": "[limit reached]",
    }


def test_orchestration_result_overrides_turn_result_without_changing_raw_events(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("managed-session", repository_root=tmp_path, model="fake-model")
    store.append_runtime(_event(
        RuntimeEventType.TURN_COMPLETED,
        "managed-session",
        payload={"content": "agent completed"},
    ))
    turn_result = TaskRuntimeResult(
        status=RuntimeResultStatus.FAILED,
        response="agent completed",
        reason="validation failed",
        error_type="ValidationFailed",
    )
    SessionEventSink(store, CallbackEventSink(lambda event: None)).record_result(
        "managed-session",
        turn_result,
    )

    projection = store.replay("managed-session")

    assert projection.last_result == turn_result
    assert [event.event_type for event in projection.events][-2:] == [
        RuntimeEventType.TURN_COMPLETED.value,
        "runtime_result_recorded",
    ]
