"""Tests for core modules: config, context, session, imports."""

from techpilot.engine import ALL_TOOLS, LLM, Agent, Config, __version__
from techpilot.engine import session as session_module
from techpilot.engine.context import ContextManager, estimate_tokens
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.engine.session import list_sessions, load_session, save_session
from techpilot.engine.tools import get_tool
from techpilot.engine.tools.base import Tool


def test_system_prompt_prefers_native_read_tools_and_simple_shell_calls():
    from techpilot.engine.prompt import system_prompt

    prompt = system_prompt([])

    assert "prefer `glob`, `grep`, and `read_file` over `bash`" in prompt
    assert "Make each call one command" in prompt
    assert "For multiple read-only steps, issue separate tool calls" in prompt
    assert "Do not claim to be OpenAI, ChatGPT" in prompt


def test_version():
    assert __version__ == "0.1.0"


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    assert {tool.name for tool in ALL_TOOLS} == {
        "agent", "bash", "edit_file", "fetch_url", "glob", "grep", "now",
        "read_file", "write_file",
    }


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TECHPILOT_MODEL", "test-model")
    c = Config.from_env()
    assert c.model == "test-model"


def test_config_defaults(monkeypatch):
    # clear relevant env vars without leaking the change into other tests
    monkeypatch.delenv("TECHPILOT_MODEL", raising=False)
    monkeypatch.delenv("TECHPILOT_MAX_TOKENS", raising=False)
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")

    c = Config.from_env()
    assert c.model == "gpt-5.5"
    assert c.max_tokens == 4096
    assert c.temperature == 0.0


def test_config_uses_model_context_default_and_explicit_override(monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    monkeypatch.setenv("TECHPILOT_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("TECHPILOT_MAX_CONTEXT", raising=False)
    assert Config.from_env().max_context_tokens == 1_000_000

    monkeypatch.setenv("TECHPILOT_MAX_CONTEXT", "64000")
    assert Config.from_env().max_context_tokens == 64000


# --- Context ---

def test_estimate_tokens():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 100


def test_context_snip():
    ctx = ContextManager(max_tokens=3000)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": "x\n" * 1000},
    ]
    before = estimate_tokens(msgs)
    ctx._snip_tool_outputs(msgs)
    after = estimate_tokens(msgs)
    assert after < before


def test_context_snip_keeps_unconsumed_tail_tool_result():
    ctx = ContextManager(max_tokens=3000)
    old_output = "old\n" * 1000
    pending_output = "pending\n" * 1000
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": old_output},
        {"role": "assistant", "tool_calls": [{"id": "pending"}]},
        {"role": "tool", "tool_call_id": "pending", "content": pending_output},
    ]

    assert ctx._snip_tool_outputs(messages) is True
    assert messages[1]["content"] != old_output
    assert messages[3]["content"] == pending_output


def test_context_snip_keeps_every_parallel_unconsumed_tool_result():
    ctx = ContextManager(max_tokens=3000)
    first_output = "first\n" * 1000
    second_output = "second\n" * 1000
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "first"}, {"id": "second"}]},
        {"role": "tool", "tool_call_id": "first", "content": first_output},
        {"role": "tool", "tool_call_id": "second", "content": second_output},
    ]

    assert ctx._snip_tool_outputs(messages) is False
    assert messages[1]["content"] == first_output
    assert messages[2]["content"] == second_output


def test_context_snip_releases_tool_result_after_model_reply():
    ctx = ContextManager(max_tokens=3000)
    output = "evidence\n" * 1000
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "evidence"}]},
        {"role": "tool", "tool_call_id": "evidence", "content": output},
        {"role": "assistant", "content": "I reviewed the evidence."},
    ]

    assert ctx._snip_tool_outputs(messages) is True
    assert messages[1]["content"] != output


def test_agent_sends_full_unconsumed_tool_result_to_next_provider_call():
    output = "evidence\n" * 1000

    class LongResultTool(Tool):
        name = "long_result"
        description = "Return controlled long evidence."
        parameters = {"type": "object", "properties": {}}

        def execute(self) -> str:
            return output

    class CapturingProvider:
        def __init__(self) -> None:
            self.requests: list[list[dict]] = []
            self.responses = iter([
                LLMResponse(tool_calls=[ToolCall("long", "long_result", {})]),
                LLMResponse(content="I reviewed the complete evidence."),
            ])

        def chat(self, messages, **kwargs):
            del kwargs
            self.requests.append([dict(message) for message in messages])
            return next(self.responses)

    provider = CapturingProvider()
    agent = Agent(llm=provider, tools=[LongResultTool()], max_context_tokens=1000)

    assert agent.chat("Inspect the evidence.") == "I reviewed the complete evidence."
    tool_messages = [message for message in provider.requests[1] if message.get("role") == "tool"]
    assert tool_messages[0]["content"] == output


def test_context_compress():
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 2000})
    before = estimate_tokens(msgs)
    ctx.maybe_compress(msgs, None)
    after = estimate_tokens(msgs)
    assert after < before
    assert len(msgs) < 40  # should be compressed


def test_safe_split_never_orphans_a_tool_message():
    """The kept tail must not begin with a 'tool' message - it would be severed
    from the assistant tool_calls that produced it, which the API rejects."""
    ctx = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "tool", "tool_call_id": "c2", "content": "result2"},
    ]
    split = ctx._safe_split(messages, keep_recent=1)
    assert messages[split].get("role") != "tool"


def test_compress_never_leaves_an_orphan_tool_reply():
    """After summarisation every tool reply must still follow its tool_calls."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "b" * 800})
    ctx.maybe_compress(msgs, None)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            prev = msgs[i - 1]
            assert prev.get("role") == "tool" or prev.get("tool_calls"), f"orphan tool at {i}"


# --- Session ---

def test_session_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    save_session(msgs, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == msgs
    assert loaded[1] == "test-model"


def test_session_name_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "../Research Notes!")

    assert sid == "Research-Notes"
    assert (tmp_path / "Research-Notes.json").exists()
    assert load_session("../Research Notes!") is not None


def test_session_not_found():
    assert load_session("nonexistent_session_id") is None


def test_list_sessions():
    sessions = list_sessions()
    assert isinstance(sessions, list)


# --- Cost estimation ---

def test_cost_estimation_known_model():
    from techpilot.engine.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "gpt-5.4"
    llm.total_prompt_tokens = 1_000_000
    llm.total_completion_tokens = 500_000
    cost = llm.estimated_cost
    assert cost is not None
    assert cost == 2.5 + 7.5  # $2.5/M in + $15/M out * 0.5M


def test_cost_estimation_deepseek_v4_model():
    from techpilot.engine.llm import LLM

    llm = LLM.__new__(LLM)
    llm.model = "deepseek-v4-pro"
    llm.total_prompt_tokens = 1_000_000
    llm.total_completion_tokens = 1_000_000
    assert llm.estimated_cost == 0.435 + 0.87

def test_cost_estimation_unknown_model():
    from techpilot.engine.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "some-custom-model"
    llm.total_prompt_tokens = 1000
    llm.total_completion_tokens = 500
    assert llm.estimated_cost is None


# --- Changed files tracking ---

def test_edit_tracks_changed_files(tmp_path):
    from techpilot.engine.tools.edit import _changed_files
    _changed_files.clear()
    edit = get_tool("edit_file")
    path = tmp_path / "sample.py"
    path.write_text("aaa\nbbb\n")
    edit.execute(file_path=str(path), old_string="aaa", new_string="zzz")
    assert any(str(path) in p for p in _changed_files)
    _changed_files.clear()


def test_write_tracks_changed_files(tmp_path):
    from techpilot.engine.tools.edit import _changed_files
    _changed_files.clear()
    write = get_tool("write_file")
    path = tmp_path / "tracked.txt"
    write.execute(file_path=str(path), content="tracked\n")
    assert any(path.name in p for p in _changed_files)
    _changed_files.clear()


# --- Agent tool execution ---

def test_agent_tool_scope_is_per_instance():
    """An Agent restricted to a subset of tools must not resolve tools outside it."""
    only_read = [get_tool("read_file")]
    agent = Agent(llm=LLM.__new__(LLM), tools=only_read)
    assert set(agent._tool_by_name) == {"read_file"}

    class _TC:
        name = "bash"  # a real, registered tool - but not in this agent's set
        id = "x"
        arguments = {"command": "echo hi"}

    assert "unknown tool 'bash'" in agent._exec_tool(_TC())


def test_exec_tool_distinguishes_bad_args_from_internal_error():
    """A TypeError raised inside a tool must not be reported as bad arguments."""
    from techpilot.engine.tools.base import Tool

    class _Boom(Tool):
        name = "boom"
        description = "raises TypeError internally"
        parameters = {"type": "object", "properties": {}, "required": []}

        def execute(self):
            raise TypeError("internal explosion")

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Boom()])

    class _BadArgs:
        name, id, arguments = "boom", "1", {"unexpected": 1}

    class _Good:
        name, id, arguments = "boom", "2", {}

    assert "bad arguments" in agent._exec_tool(_BadArgs())
    assert "Error executing boom" in agent._exec_tool(_Good())
    assert "bad arguments" not in agent._exec_tool(_Good())


def test_optional_tool_executor_intercepts_validated_tool_calls():
    """Applications can add policy without changing normal Tool execution."""
    from techpilot.engine.tools.base import Tool

    class _Echo(Tool):
        name = "echo"
        description = "returns the provided value"
        parameters = {"type": "object", "properties": {}, "required": []}

        def execute(self, value: str) -> str:
            return f"direct: {value}"

    class _Executor:
        def __init__(self):
            self.calls = []

        def execute(self, tool, arguments):
            self.calls.append((tool.name, arguments))
            return f"controlled: {arguments['value']}"

    executor = _Executor()
    agent = Agent(llm=LLM.__new__(LLM), tools=[_Echo()], tool_executor=executor)

    class _Good:
        name, id, arguments = "echo", "1", {"value": "hello"}

    assert agent._exec_tool(_Good()) == "controlled: hello"
    assert executor.calls == [("echo", {"value": "hello"})]


def test_agent_without_executor_keeps_direct_tool_execution():
    """The optional extension point must not change the runtime's default path."""
    from techpilot.engine.tools.base import Tool

    class _Echo(Tool):
        name = "echo"
        description = "returns the provided value"
        parameters = {"type": "object", "properties": {}, "required": []}

        def execute(self, value: str) -> str:
            return f"direct: {value}"

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Echo()])

    class _Good:
        name, id, arguments = "echo", "1", {"value": "unchanged"}

    assert agent.tool_executor is None
    assert agent._exec_tool(_Good()) == "direct: unchanged"


def test_interrupt_backfills_missing_tool_replies():
    """A half-finished tool round must be repaired so history stays valid."""
    agent = Agent(llm=LLM.__new__(LLM), tools=[])
    agent.messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
    ]

    class _TC:
        def __init__(self, i):
            self.id = i

    agent._answer_pending_tool_calls([_TC("a"), _TC("b")])
    replies = [m for m in agent.messages if m.get("role") == "tool"]
    ids = [m["tool_call_id"] for m in replies]
    assert sorted(ids) == ["a", "b"]
    assert ids.count("a") == 1  # the already-answered call wasn't duplicated
