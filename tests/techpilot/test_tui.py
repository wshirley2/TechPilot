"""Tests for the optional prompt-toolkit projection over Chat Runtime events."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.application import Application as PromptToolkitApplication
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier
from prompt_toolkit.output import ColorDepth, DummyOutput
from prompt_toolkit.utils import get_cwidth

from techpilot.chat.tui import TechPilotTui, _preserve_shift_enter
from techpilot.engine.events import RuntimeEvent, RuntimeEventType
from techpilot.engine.permissions import PermissionDecision, PermissionEffect, PermissionRequest
from techpilot.runtime.sessions import SessionEvent, SessionStore


def _runtime(tmp_path: Path, store: SessionStore, session_id: str):
    return SimpleNamespace(
        agent=SimpleNamespace(
            session_id=session_id,
            llm=SimpleNamespace(total_prompt_tokens=12, total_completion_tokens=8),
        ),
        config=SimpleNamespace(model="fake-model"),
        session_store=store,
        repository=tmp_path,
        profile=None,
    )


def _event(event_type: RuntimeEventType, *, call_id: str | None = None, payload=None) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type,
        session_id="tui-session",
        turn_id="turn-1",
        round_index=1,
        tool_call_id=call_id,
        payload=payload or {},
    )


def test_tui_projects_events_and_opens_saved_tool_detail(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    for event_type, payload in (
        ("tool_requested", {"tool_name": "bash", "arguments": {"command": "dir /b"}}),
        ("tool_completed", {"tool_name": "bash", "result": "README.md", "interrupted": False}),
    ):
        store.append(SessionEvent(
            event_type=event_type,
            session_id="tui-session",
            turn_id="turn-1",
            round_index=1,
            tool_call_id="call-1",
            payload=payload,
        ))
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))

    tui.enqueue_event(_event(RuntimeEventType.TURN_STARTED, payload={"user_input": "read the repository"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="call-1", payload={"tool_name": "bash"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_COMPLETED, call_id="call-1", payload={"tool_name": "bash", "result": "README.md"}))
    tui._drain_updates()
    tui._handle_details_command("/details call-1")

    assert "❯ you\n  read the repository" in tui.transcript_text
    assert "▸ ✓ bash · completed" in tui.transcript_text
    assert "· details\n  bash · completed" in tui.transcript_text
    assert "Command: dir /b" in tui.transcript_text
    assert "README.md" in tui.transcript_text


def test_tui_shows_block_reason_and_waits_for_the_existing_permission_prompt(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="blocked", payload={"tool_name": "bash"}))
    tui.enqueue_event(_event(
        RuntimeEventType.EXECUTION_CONTROL_ASSESSED,
        call_id="blocked",
        payload={
            "required_control": "block",
            "reasons": [{"message": "complex shell", "evidence": ["redirection"]}],
        },
    ))
    tui._drain_updates()

    assert "BLOCKED" in tui.transcript_text
    assert "complex shell：redirection" in tui.transcript_text

    request = PermissionRequest(
        tool_call_id="write-1",
        tool_name="write_file",
        effect=PermissionEffect.WRITE,
        normalized_arguments={"file_path": "notes.txt", "content": "after\n"},
        reason="File writes require review of a trusted diff",
        scope="notes.txt",
        trusted_preview="--- a/notes.txt\n+++ b/notes.txt\n+after",
    )
    decision: list[PermissionDecision] = []
    started = threading.Event()

    def decide() -> None:
        started.set()
        decision.append(tui.permission_prompt.decide(request))

    # A test runner may be interrupted while the UI is displaying this prompt.
    # Keep the helper daemonized and release it in ``finally`` so an interrupted
    # assertion cannot leave a non-daemon permission waiter blocking pytest exit.
    worker = threading.Thread(target=decide, daemon=True)
    worker.start()
    try:
        assert started.wait(timeout=1)
        tui._drain_updates()
        assert "permission requested" in tui.transcript_text
        assert "Trusted Diff" in tui.transcript_text
        assert tui._input_prompt() == "Permission [1/2/3] > "
        tui._answer_permission_text("allow")
    finally:
        if worker.is_alive():
            tui.answer_permission(PermissionDecision.deny("test cleanup after interruption"))
            worker.join(timeout=1)

    assert decision[0].action.value == "allow"
    assert "permission approved" in tui.transcript_text


def test_tui_renders_cancelled_and_failed_turn_states(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))

    tui.enqueue_event(_event(RuntimeEventType.TURN_INTERRUPTED, payload={}))
    tui._drain_updates()
    assert "Turn cancelled" in tui.transcript_text
    assert "Cancelled" in tui._status_area.text

    tui.enqueue_event(_event(RuntimeEventType.TURN_FAILED, payload={"error": "provider down"}))
    tui._drain_updates()
    assert "Turn failed: provider down" in tui.transcript_text
    assert "Failed" in tui._status_area.text


def test_tui_batches_streaming_tokens_until_the_next_runtime_boundary(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))

    tui.enqueue_event(_event(RuntimeEventType.TURN_STARTED, payload={"user_input": "hello"}))
    tui.enqueue_event(_event(RuntimeEventType.ASSISTANT_TOKEN, payload={"token": "hel"}))
    tui.enqueue_event(_event(RuntimeEventType.ASSISTANT_TOKEN, payload={"token": "lo"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="call-1", payload={"tool_name": "glob"}))
    tui._drain_updates()

    assert "❯ you\n  hello" in tui.transcript_text
    assert "● TechPilot\n  hello" in tui.transcript_text
    assert "▸ ◌ glob · running" in tui.transcript_text


def test_tui_uses_one_transcript_instead_of_side_panels(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))

    class FakeApplication:
        def __init__(self, **kwargs):
            self.layout = kwargs["layout"]
            self.options = kwargs

    monkeypatch.setattr("techpilot.chat.tui.Application", FakeApplication)
    application = tui._build_application()
    children = application.layout.container.children

    assert len(children) == 6
    assert children[1] is tui.conversation
    assert tui._activity_container in children
    assert tui.input.window in children
    assert application.options["full_screen"] is False
    assert application.options["erase_when_done"] is True
    assert application.options["color_depth"] is ColorDepth.DEPTH_24_BIT


def test_tui_clicking_a_tool_arrow_toggles_its_saved_output(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="call-1", payload={"tool_name": "bash", "arguments": {"command": "dir /b"}}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_COMPLETED, call_id="call-1", payload={"tool_name": "bash", "result": "README.md"}))
    tui._drain_updates()

    assert "output:" not in tui.transcript_text
    tui._toggle_tool_handler("call-1")(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))

    assert "▾ ✓ bash · completed" in tui.transcript_text
    assert "output:" in tui.transcript_text
    assert "README.md" in tui.transcript_text

    tui._toggle_tool_handler("call-1")(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))

    assert "▸ ✓ bash · completed" in tui.transcript_text
    assert "output:" not in tui.transcript_text


def test_tui_groups_adjacent_safe_tool_calls_and_keeps_single_call_expansion(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    for call_id, tool_name, result in (
        ("call-1", "glob", "a.py"),
        ("call-2", "read_file", "README.md"),
        ("call-3", "read_file", "pyproject.toml"),
    ):
        tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id=call_id, payload={"tool_name": tool_name}))
        tui.enqueue_event(_event(RuntimeEventType.TOOL_COMPLETED, call_id=call_id, payload={"tool_name": tool_name, "result": result}))
    tui._drain_updates()

    assert "▸ 3 tools · glob ×1, read_file ×2 · completed" in tui.transcript_text
    assert "✓ glob · completed" not in tui.transcript_text

    tui._toggle_tool_group_handler("call-1")(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))

    assert "▾ 3 tools · glob ×1, read_file ×2 · completed" in tui.transcript_text
    assert "  ▸ ✓ glob · completed" in tui.transcript_text
    assert "  ▸ ✓ read_file · completed" in tui.transcript_text

    tui._toggle_tool_handler("call-2")(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))

    assert "  ▾ ✓ read_file · completed" in tui.transcript_text
    assert "README.md" in tui.transcript_text


def test_tui_ctrl_o_opens_the_latest_grouped_tool_detail(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    for call_id, tool_name in (("call-1", "glob"), ("call-2", "read_file")):
        tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id=call_id, payload={"tool_name": tool_name}))
        tui.enqueue_event(_event(RuntimeEventType.TOOL_COMPLETED, call_id=call_id, payload={"tool_name": tool_name, "result": "done"}))
    tui._drain_updates()

    tui._toggle_latest_tool_detail()

    assert "call-1" in tui._expanded_tool_groups
    assert tui._tool_activities["call-2"].expanded
    assert "▾ 2 tools" in tui.transcript_text
    assert "saved detail: /details call-2" in tui.transcript_text


def test_tui_welcome_is_presentation_only_and_markdown_is_rendered_lightly():
    tui = TechPilotTui()
    tui._welcome_visible = True

    assert "Local-first coding agent" in tui.transcript_text
    assert tui._input_prompt() == "❯ "
    welcome_rendered = "".join(fragment[1] for fragment in tui._transcript_fragments())
    assert "┌─ TechPilot · Local-first Chat" in welcome_rendered
    assert "Getting started" in welcome_rendered
    assert "Workspace:" in welcome_rendered
    assert "└" in welcome_rendered

    tui._dismiss_welcome()
    tui._append_transcript(
        "TechPilot",
        "# Heading\n- **important** and `code`\n```python\nprint('hi')\n```",
        kind="assistant",
    )

    assert "# Heading" not in tui.transcript_text
    assert "  Heading" in tui.transcript_text
    assert "  • important and code" in tui.transcript_text
    assert "┌ python" in tui.transcript_text
    assert "└" in tui.transcript_text
    classes = [fragment[0] for fragment in tui._transcript_fragments()]
    assert "class:assistant" in classes
    assert "class:markdown-heading" in classes
    assert "class:markdown-inline-code" in classes
    assert "class:markdown-strong" in classes


def test_tui_run_starts_with_the_welcome_panel_visible(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))

    class FakeApplication:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return 0

    monkeypatch.setattr("techpilot.chat.tui.Application", FakeApplication)
    monkeypatch.setattr("techpilot.chat.tui.print_formatted_text", lambda *_args, **_kwargs: None)

    assert tui.run() == 0
    assert tui._welcome_visible
    rendered = "".join(fragment[1] for fragment in tui._welcome_fragments())
    assert "┌─ TechPilot · Local-first Chat" in rendered
    assert "Type a message to begin" in rendered


def test_tui_exit_emits_complete_colored_transcript_to_scrollback(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui._append_transcript("you", "介绍下你自己", kind="user")
    tui._append_transcript("TechPilot", "我是 TechPilot。", kind="assistant")
    emitted: list[tuple[object, dict[str, object]]] = []

    class FakeApplication:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return 0

    def record_transcript(fragments, **kwargs) -> None:
        emitted.append((fragments, kwargs))

    monkeypatch.setattr("techpilot.chat.tui.Application", FakeApplication)
    monkeypatch.setattr("techpilot.chat.tui.print_formatted_text", record_transcript)

    assert tui.run() == 0
    assert len(emitted) == 1
    fragments, kwargs = emitted[0]
    assert isinstance(fragments, FormattedText)
    rendered = "".join(text for _style, text in fragments)
    assert "┌─ TechPilot · Local-first Chat" in rendered
    assert "介绍下你自己" in rendered
    assert "我是 TechPilot。" in rendered
    assert kwargs["style"] is tui._style
    assert kwargs["color_depth"] is ColorDepth.DEPTH_24_BIT


def test_tui_welcome_accepts_the_first_message_directly(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui._welcome_visible = True
    tui.input.buffer.text = "inspect this project"
    finished = threading.Event()

    def fake_run_turn(text: str) -> None:
        assert text == "inspect this project"
        finished.set()

    tui._run_turn = fake_run_turn
    assert tui._submit(tui.input.buffer) is True
    assert tui._welcome_visible
    assert "Local-first coding agent" in tui.transcript_text
    assert finished.wait(timeout=1)


def test_tui_restores_user_and_assistant_messages_from_a_resumed_session(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    store.append(SessionEvent(
        event_type=RuntimeEventType.TURN_STARTED.value,
        session_id="tui-session",
        turn_id="turn-1",
        payload={"user_input": "介绍下你自己"},
    ))
    store.append(SessionEvent(
        event_type=RuntimeEventType.TURN_COMPLETED.value,
        session_id="tui-session",
        turn_id="turn-1",
        payload={"content": "我是 TechPilot。"},
    ))

    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))

    assert "Local-first coding agent" in tui.transcript_text
    assert "介绍下你自己" in tui.transcript_text
    assert "我是 TechPilot。" in tui.transcript_text




def test_tui_user_marker_is_not_highlighted_but_question_body_is(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(window_width=40)
    entry = tui._append_transcript("you", "Please inspect README", kind="user")

    fragments = tui._entry_fragments(entry)
    assert fragments[0] == ("class:transcript", "❯ you")
    assert ("class:user", "\n  ") in fragments
    assert ("class:user", "Please inspect README") in fragments
    assert ("class:user", " " * (40 - get_cwidth("Please inspect README") - 2)) in fragments


def test_tui_user_background_uses_terminal_width_for_cjk_and_emoji(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(window_width=24)

    for body in ("你现在是什么模型", "模型 🤖"):
        entry = tui._append_transcript("you", body, kind="user")
        fragments = tui._entry_fragments(entry)
        body_fragments = fragments[1:]
        rendered_line = "".join(text for _style, text in body_fragments).split("\n", maxsplit=1)[1]

        assert get_cwidth(rendered_line) == 24


def test_tui_keeps_failed_and_blocked_tool_calls_out_of_folded_groups(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="safe", payload={"tool_name": "glob"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_COMPLETED, call_id="safe", payload={"tool_name": "glob", "result": "a.py"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="blocked", payload={"tool_name": "bash"}))
    tui.enqueue_event(_event(RuntimeEventType.EXECUTION_CONTROL_ASSESSED, call_id="blocked", payload={"required_control": "block"}))
    tui._drain_updates()

    assert "▸ ✓ glob · completed" in tui.transcript_text
    assert "▸ ! bash · block" in tui.transcript_text
    assert "BLOCKED" in tui.transcript_text


def test_tui_shift_enter_inserts_a_newline_and_input_has_bottom_divider(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    assert tui._key_bindings.get_bindings_for_keys(("c-m",))
    assert tui._key_bindings.get_bindings_for_keys(("escape", "c-m"))
    assert tui._key_bindings.get_bindings_for_keys(("escape", "[", "1", "3", ";", "2", "u"))

    with create_pipe_input() as pipe_input:
        def build_application(**kwargs):
            kwargs.pop("input").close()
            return PromptToolkitApplication(input=pipe_input, output=DummyOutput(), **kwargs)

        monkeypatch.setattr("techpilot.chat.tui.Application", build_application)
        application = tui._build_application()
        state: list[str] = []

        def pre_run() -> None:
            for keys in (
                [KeyPress(Keys.Escape, "\x1b"), KeyPress(Keys.ControlM, "\r")],
                [
                    KeyPress(Keys.Escape, "\x1b"),
                    KeyPress("[", "["),
                    KeyPress("1", "1"),
                    KeyPress("3", "3"),
                    KeyPress(";", ";"),
                    KeyPress("2", "2"),
                    KeyPress("u", "u"),
                ],
            ):
                tui.input.buffer.text = "first line"
                tui.input.buffer.cursor_position = len(tui.input.buffer.text)
                application.key_processor.feed_multiple(keys)
                application.key_processor.process_keys()
                tui.input.buffer.insert_text("second line")
                state.append(tui.input.buffer.text)

            monkeypatch.setattr("techpilot.chat.tui._physical_shift_is_pressed", lambda: True)
            tui.input.buffer.text = "first line"
            tui.input.buffer.cursor_position = len(tui.input.buffer.text)
            application.key_processor.feed(KeyPress(Keys.ControlM, "\r"))
            application.key_processor.process_keys()
            tui.input.buffer.insert_text("second line")
            state.append(tui.input.buffer.text)
            application.exit()

        application.run(pre_run=pre_run)
        assert tui.input.window.height.max == 2

        tui.input.buffer.text = "line one\nline two\nline three"
        application.run(pre_run=application.exit)
        assert tui.input.window.render_info is not None
        assert tui.input.window.render_info.window_height >= 3

        tui.input.buffer.text = "line one"
        application.run(pre_run=application.exit)
        assert tui.input.window.render_info is not None
        assert tui.input.window.render_info.window_height == 1

    assert state == ["first line\nsecond line"] * 3
    children = application.layout.container.children
    assert children[-1].char == "─"
    assert not tui.input.window.dont_extend_width()
    assert not children[-1].dont_extend_width()


def test_tui_preserves_windows_shift_enter_as_a_newline_shortcut():
    ordinary_enter = [KeyPress(Keys.ControlM, "\r")]

    assert _preserve_shift_enter(ordinary_enter, shift_pressed=False) == ordinary_enter
    assert _preserve_shift_enter(ordinary_enter, shift_pressed=True) == [
        KeyPress(Keys.Escape, ""),
        KeyPress(Keys.ControlM, "\r"),
    ]


def test_tui_exit_command_closes_the_tui_without_starting_a_turn():
    tui = TechPilotTui()

    class FakeApplication:
        def __init__(self) -> None:
            self.result = None

        def exit(self, *, result: int) -> None:
            self.result = result

    application = FakeApplication()
    tui.application = application
    tui.input.buffer.text = "/exit"

    assert tui._submit(tui.input.buffer) is True
    assert tui.input.buffer.text == ""
    assert application.result == 0
    assert not tui._running


def test_tui_marks_manual_history_browsing_and_uses_numbered_permission_choices(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)
    tui._scroll_transcript(-1, page=True)
    tui._refresh_status()

    assert "Viewing history · Ctrl+End latest" in tui._activity_area.text

    request = PermissionRequest(
        tool_call_id="write-1",
        tool_name="write_file",
        effect=PermissionEffect.WRITE,
        normalized_arguments={},
        reason="review required",
        scope="notes.txt",
    )
    tui._pending_permission = SimpleNamespace(request=request, response=None)
    tui._refresh_permission()

    assert "1 allow once · 2 allow for session · 3 deny" in tui.transcript_text


def test_tui_page_keys_scroll_the_transcript_by_one_page(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)

    tui._scroll_transcript(1, page=True)

    assert tui.conversation.vertical_scroll == 19
    assert not tui._follow_tail

    tui._scroll_transcript(-1, page=True)

    assert tui.conversation.vertical_scroll == 0
    assert not tui._follow_tail


def test_tui_mouse_wheel_explicitly_moves_the_transcript(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)

    tui._transcript_control.mouse_handler(
        MouseEvent(Point(0, 0), MouseEventType.SCROLL_DOWN, MouseButton.LEFT, frozenset())
    )

    assert tui.conversation.vertical_scroll == 3
    assert not tui._follow_tail


def test_tui_tail_cursor_uses_the_actual_fragment_line_count(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.enqueue_event(_event(RuntimeEventType.TURN_STARTED, payload={"user_input": "hello"}))
    tui.enqueue_event(_event(RuntimeEventType.ASSISTANT_TOKEN, payload={"token": "hello"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="call-1", payload={"tool_name": "glob"}))
    tui._drain_updates()

    rendered = "".join(fragment[1] for fragment in tui._transcript_fragments())
    cursor = tui._transcript_cursor()

    assert cursor is not None
    assert cursor.y == rendered.count("\n")


def test_tui_renders_multiple_transcript_entries_without_cursor_overflow(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.enqueue_event(_event(RuntimeEventType.TURN_STARTED, payload={"user_input": "hello"}))
    tui.enqueue_event(_event(RuntimeEventType.ASSISTANT_TOKEN, payload={"token": "hello"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="call-1", payload={"tool_name": "glob"}))

    with create_pipe_input() as pipe_input:
        def build_application(**kwargs):
            kwargs.pop("input").close()
            return PromptToolkitApplication(input=pipe_input, output=DummyOutput(), **kwargs)

        monkeypatch.setattr("techpilot.chat.tui.Application", build_application)
        application = tui._build_application()
        application.run(pre_run=application.exit)


def test_tui_scrolling_to_the_bottom_resumes_tail_follow(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)

    for _ in range(5):
        tui._scroll_transcript(1, page=True)

    assert tui.conversation.vertical_scroll == 80
    assert tui._follow_tail


def test_tui_streaming_and_tool_updates_do_not_drag_a_scrolled_user_back(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)
    tui._scroll_transcript(-1, page=True)
    assert not tui._follow_tail

    tui.enqueue_event(_event(RuntimeEventType.ASSISTANT_TOKEN, payload={"token": "hello"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_REQUESTED, call_id="call-1", payload={"tool_name": "glob"}))
    tui.enqueue_event(_event(RuntimeEventType.TOOL_COMPLETED, call_id="call-1", payload={"tool_name": "glob", "result": "a.py"}))
    tui._drain_updates()

    assert not tui._follow_tail


def test_tui_new_turn_resumes_tail_follow(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)
    tui._scroll_transcript(-1, page=True)
    assert not tui._follow_tail

    tui.enqueue_event(_event(RuntimeEventType.TURN_STARTED, payload={"user_input": "hello"}))
    tui._drain_updates()

    assert tui._follow_tail


def test_tui_scroll_shortcuts_are_registered(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))

    for keys in (("pageup",), ("pagedown",), ("c-home",), ("c-end",), ("c-o",)):
        assert tui._key_bindings.get_bindings_for_keys(keys)


def test_tui_scroll_to_top_and_bottom_shortcuts(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)
    tui.conversation.vertical_scroll = 40
    tui._follow_tail = False

    tui._scroll_to_bottom()
    assert tui._follow_tail
    assert tui.conversation.vertical_scroll == 80

    tui._scroll_to_top()
    assert not tui._follow_tail
    assert tui.conversation.vertical_scroll == 0


def test_tui_cursor_pins_to_the_viewport_when_not_following(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    for i in range(10):
        tui._append_transcript(f"line {i}", "body", kind="assistant")
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)
    tui.conversation.vertical_scroll = 5
    tui._follow_tail = False

    cursor = tui._transcript_cursor()

    rendered = "".join(fragment[1] for fragment in tui._transcript_fragments())
    assert cursor == Point(x=0, y=min(rendered.count("\n"), tui.conversation.vertical_scroll + 19))


def test_tui_scroll_position_survives_rerender_while_not_following(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    for i in range(60):
        tui._append_transcript(f"line {i}", "body", kind="assistant")

    with create_pipe_input() as pipe_input:
        def build_application(**kwargs):
            kwargs.pop("input").close()
            return PromptToolkitApplication(input=pipe_input, output=DummyOutput(), **kwargs)

        monkeypatch.setattr("techpilot.chat.tui.Application", build_application)
        application = tui._build_application()
        state: dict[str, int] = {}

        def pre_run() -> None:
            tui._follow_tail = False
            tui.conversation.vertical_scroll = 30
            application.renderer.render(application, application.layout)
            state["vertical_scroll"] = tui.conversation.render_info.vertical_scroll
            application.exit()

        application.run(pre_run=pre_run)

    assert state["vertical_scroll"] == 30


def test_tui_shift_wheel_scrolls_a_full_page(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("tui-session", repository_root=tmp_path, model="fake-model")
    tui = TechPilotTui()
    tui.bind_runtime(_runtime(tmp_path, store, "tui-session"))
    tui.conversation.render_info = SimpleNamespace(content_height=100, window_height=20)

    tui._transcript_control.mouse_handler(
        MouseEvent(Point(0, 0), MouseEventType.SCROLL_DOWN, MouseButton.LEFT, frozenset({MouseModifier.SHIFT}))
    )

    assert tui.conversation.vertical_scroll == 19
    assert not tui._follow_tail
