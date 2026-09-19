"""Optional prompt-toolkit presentation for the existing Chat Runtime.

The TUI owns no Agent, tool execution, permission policy, or persisted state.
It turns keyboard input into ``TaskRuntime.run_turn`` calls and projects the
same Runtime Events already written to the Session event log.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass

from prompt_toolkit import print_formatted_text
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseEventType, MouseModifier
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

from techpilot.engine.events import RuntimeEvent, RuntimeEventType
from techpilot.engine.permissions import PermissionDecision, PermissionGrantScope, PermissionRequest

from ..runtime import TaskRuntime
from .session import ToolCallDetail, _find_tool_call_detail, _format_tool_call_detail, _tool_call_details, _tool_status


@dataclass
class _TranscriptEntry:
    label: str
    body: str
    kind: str
    call_id: str | None = None


@dataclass
class _ToolActivity:
    call_id: str
    tool_name: str
    status: str = "running"
    started_at: float = 0.0
    finished_at: float | None = None
    arguments: str = ""
    result: str = ""
    expanded: bool = False
    entry: _TranscriptEntry | None = None


@dataclass
class _PendingPermission:
    request: PermissionRequest
    response: queue.Queue[PermissionDecision]


@dataclass(frozen=True)
class _ToolGroup:
    """Ephemeral presentation group for adjacent non-critical Tool Calls."""

    group_id: str
    call_ids: tuple[str, ...]


@dataclass(frozen=True)
class _MarkdownLine:
    """A small, terminal-safe projection of one assistant Markdown line."""

    raw: str
    plain: str
    style: str = "markdown-body"


class _TranscriptControl(FormattedTextControl):
    """Render the transcript and scroll it directly on mouse-wheel input."""

    def __init__(self, tui: TechPilotTui) -> None:
        super().__init__(
            tui._transcript_fragments,
            focusable=True,
            show_cursor=False,
            get_cursor_position=tui._transcript_cursor,
        )
        self._tui = tui

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type in {MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN}:
            direction = -1 if mouse_event.event_type is MouseEventType.SCROLL_UP else 1
            page = bool({MouseModifier.SHIFT, MouseModifier.CONTROL} & mouse_event.modifiers)
            self._tui._scroll_transcript(direction, page=page)
            return
        return super().mouse_handler(mouse_event)


def _preserve_shift_enter(key_presses: list[KeyPress], *, shift_pressed: bool) -> list[KeyPress]:
    """Represent the modifier Windows supplies for Shift+Enter.

    ``ConsoleInputReader`` receives the Shift flag but prompt-toolkit leaves
    Enter as an ordinary ``ControlM`` key. Prefixing it with Escape matches the
    portable Shift+Enter sequence handled by the TUI key bindings below.
    """
    if (
        shift_pressed
        and len(key_presses) == 1
        and key_presses[0].key in {Keys.ControlJ, Keys.ControlM}
    ):
        key_press = key_presses[0]
        return [KeyPress(Keys.Escape, ""), KeyPress(key_press.key, key_press.data)]
    return key_presses


def _physical_shift_is_pressed() -> bool:
    """Best-effort Shift check for Windows terminals that lose modifiers.

    ConPTY-backed terminals can turn Shift+Enter into an unmodified carriage
    return before prompt-toolkit sees it. ``GetAsyncKeyState`` still reflects
    the physical key while that return event is being handled.
    """
    if os.name != "nt":
        return False
    from ctypes import windll

    return bool(windll.user32.GetAsyncKeyState(0x10) & 0x8000)


class TuiPermissionPrompt:
    """PermissionPrompt adapter that waits for a decision made in the TUI."""

    def __init__(self, tui: TechPilotTui) -> None:
        self._tui = tui

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        return self._tui.request_permission(request)


class TuiEventSink:
    """Forward the existing Runtime Event stream into an ephemeral UI queue."""

    last_turn_streamed = True

    def __init__(self, tui: TechPilotTui) -> None:
        self._tui = tui

    def emit(self, event: RuntimeEvent) -> None:
        self._tui.enqueue_event(event)

    def ensure_persisted(self) -> None:
        """SessionEventSink already persists before calling this observer."""


class TechPilotTui:
    """A small keyboard-first TUI over the normal TaskRuntime."""

    _MAX_CONVERSATION_CHARS = 80_000

    def __init__(self) -> None:
        self.runtime: TaskRuntime | None = None
        self.event_sink = TuiEventSink(self)
        self.permission_prompt = TuiPermissionPrompt(self)
        self._updates: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._entries: list[_TranscriptEntry] = []
        self._tool_activities: dict[str, _ToolActivity] = {}
        self._expanded_tool_groups: set[str] = set()
        self._pending_permission: _PendingPermission | None = None
        self._restored_session_id: str | None = None
        self._running = False
        self._streamed_turn = False
        self._stream_entry: _TranscriptEntry | None = None
        self._turn_started_at = 0.0
        self._turn_tool_count = 0
        self._stream_buffer = ""
        self._follow_tail = True
        self._status = "Ready"
        self._style = _tui_style()
        # The welcome panel is the first transcript block, not a separate
        # page. It remains part of the history while newer entries scroll it
        # out of the viewport naturally.
        self._welcome_visible = True

        self._topbar = TextArea(read_only=True, height=1, style="class:topbar", focusable=False)
        self._transcript_control = _TranscriptControl(self)
        self.conversation = Window(
            content=self._transcript_control,
            wrap_lines=True,
            style="class:transcript",
        )
        self.input = TextArea(
            multiline=True,
            # The text-change hook below keeps this Dimension in sync with
            # the current buffer, including shrinking after deletion.
            height=Dimension(min=1, preferred=1, max=1),
            prompt=self._input_prompt,
            style="class:input",
            accept_handler=self._submit,
        )
        self.input.buffer.on_text_changed += self._on_input_text_changed
        self._status_area = TextArea(read_only=True, height=1, style="class:status", focusable=False)
        self._activity_area = TextArea(read_only=True, height=1, style="class:activity", focusable=False)
        self._activity_container = ConditionalContainer(
            content=self._activity_area,
            filter=Condition(lambda: bool(self._activity_text())),
        )

        bindings = KeyBindings()

        @bindings.add("c-c")
        def _cancel_or_exit(_event) -> None:
            self.cancel_or_exit()

        @bindings.add("c-d")
        def _exit(_event) -> None:
            self.exit()

        @bindings.add("c-o")
        def _toggle_latest_tool(_event) -> None:
            self._toggle_latest_tool_detail()

        @bindings.add("enter", eager=True)
        def _submit_input(event) -> None:
            if _physical_shift_is_pressed():
                self._insert_input_newline(event.current_buffer)
                return
            self._submit(event.current_buffer)

        # Common terminals encode Shift+Enter as Escape + Enter. Windows
        # Terminal and Kitty can instead report the CSI 13;2u modified key.
        for shift_enter_keys in (
            ("escape", "enter"),
            ("escape", "[", "1", "3", ";", "2", "u"),
        ):
            @bindings.add(*shift_enter_keys, eager=True)
            def _insert_input_newline(event) -> None:
                self._insert_input_newline(event.current_buffer)

        @bindings.add("pageup")
        def _page_up(_event) -> None:
            self._scroll_transcript(-1, page=True)

        @bindings.add("pagedown")
        def _page_down(_event) -> None:
            self._scroll_transcript(1, page=True)

        @bindings.add("c-home")
        def _scroll_top(_event) -> None:
            self._scroll_to_top()

        @bindings.add("c-end")
        def _scroll_bottom(_event) -> None:
            self._scroll_to_bottom()

        self._key_bindings = bindings
        self.application: Application[int] | None = None

    def _build_application(self) -> Application[int]:
        return Application(
            layout=Layout(
                HSplit([
                    self._topbar,
                    self.conversation,
                    self._activity_container,
                    Window(height=1, char="─", style="class:divider", dont_extend_height=True),
                    self.input,
                    Window(height=1, char="─", style="class:divider", dont_extend_height=True),
                ]),
                focused_element=self.input,
            ),
            input=self._create_input(),
            key_bindings=self._key_bindings,
            # Some embedded terminals set NO_COLOR=1 and TERM=dumb for child
            # processes even while they render ANSI truecolor correctly. A
            # TUI is only built after the interactive TTY gate, so use the
            # palette it explicitly defines instead of degrading tool and
            # safety distinctions to monochrome.
            color_depth=ColorDepth.DEPTH_24_BIT,
            style=self._style,
            # The dynamic viewport only contains the latest screenful. Clear
            # it on exit, then emit the complete styled transcript below.
            full_screen=False,
            erase_when_done=True,
            mouse_support=True,
            before_render=self._drain_updates,
        )

    @staticmethod
    def _create_input():
        """Return the normal input source, with a Windows Shift+Enter repair."""
        source = create_input()
        if os.name != "nt":
            return source

        # Import lazily: prompt_toolkit's Win32 module is unavailable on
        # non-Windows systems, where the normal VT input already reports the
        # portable Escape+Enter sequence handled above.
        from prompt_toolkit.input.win32 import ConsoleInputReader, Win32Input

        if not isinstance(source, Win32Input) or not isinstance(source.console_input_reader, ConsoleInputReader):
            return source

        class _ShiftEnterConsoleInputReader(ConsoleInputReader):
            def _event_to_key_presses(self, event) -> list[KeyPress]:
                key_presses = super()._event_to_key_presses(event)
                return _preserve_shift_enter(
                    key_presses,
                    shift_pressed=bool(event.ControlKeyState & self.SHIFT_PRESSED),
                )

        source.console_input_reader.close()
        source.console_input_reader = _ShiftEnterConsoleInputReader()
        return source

    def bind_runtime(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime
        self._restore_session_transcript()
        self._status = "Ready"
        self._refresh_status()

    def run(self) -> int:
        if self.runtime is None:
            raise RuntimeError("Bind a TaskRuntime before running the TUI")
        self._welcome_visible = True
        self._refresh_status()
        self.application = self._build_application()
        result = self.application.run()
        self._emit_exit_transcript()
        return result

    def _emit_exit_transcript(self) -> None:
        """Write the complete styled history to normal terminal scrollback."""

        fragments = FormattedText((style, text) for style, text, *_ in self._transcript_fragments())
        print_formatted_text(
            fragments,
            style=self._style,
            color_depth=ColorDepth.DEPTH_24_BIT,
        )

    def _restore_session_transcript(self) -> None:
        """Project durable chat messages back into the TUI after ``--resume``."""

        if self.runtime is None:
            return
        session_store = getattr(self.runtime, "session_store", None)
        if session_store is None:
            return
        session_id = self.runtime.agent.session_id
        if self._restored_session_id == session_id:
            return
        try:
            projection = session_store.replay(session_id)
        except (OSError, ValueError):
            return
        for message in projection.messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "user":
                self._entries.append(_TranscriptEntry("you", content, "user"))
            elif role == "assistant":
                self._entries.append(_TranscriptEntry("TechPilot", content, "assistant"))
        self._restored_session_id = session_id
        self._trim_transcript()


    def enqueue_event(self, event: RuntimeEvent) -> None:
        self._updates.put(("event", event))
        self._invalidate()

    def request_permission(self, request: PermissionRequest) -> PermissionDecision:
        response: queue.Queue[PermissionDecision] = queue.Queue(maxsize=1)
        self._updates.put(("permission", _PendingPermission(request, response)))
        self._invalidate()
        return response.get()

    def answer_permission(self, decision: PermissionDecision) -> None:
        pending = self._pending_permission
        if pending is None:
            return
        self._pending_permission = None
        pending.response.put(decision)
        self._status = "Permission approved" if decision.action.value == "allow" else "Permission denied"
        outcome = "approved" if decision.action.value == "allow" else "denied"
        self._append_transcript("permission", f"permission {outcome}\n  {decision.reason}", kind="permission")
        self._refresh_status()

    def cancel_or_exit(self) -> None:
        if self._pending_permission is not None:
            self.answer_permission(PermissionDecision.deny("用户取消了权限确认"))
            return
        if self._running and self.runtime is not None:
            self.runtime.cancel_current_turn("cancelled from TUI")
            self._status = "Cancelling current turn…"
            self._refresh_status()
            return
        self.exit()

    def exit(self) -> None:
        if self._pending_permission is not None:
            self.answer_permission(PermissionDecision.deny("TUI closed while awaiting permission"))
        if self._running and self.runtime is not None:
            self.runtime.cancel_current_turn("TUI closed")
        if self.application is not None:
            self.application.exit(result=0)

    def _handle_details_command(self, text: str) -> bool:
        if not text.startswith("/details"):
            return False
        parts = text.split(maxsplit=1)
        call_id = parts[1].strip() if len(parts) > 1 else ""
        if call_id in {"on", "off"}:
            self._append_transcript("details", "Tool details are available on demand. Use /details <call-id>.", kind="details")
            return True
        if not call_id:
            if not self._tool_activities:
                self._append_transcript("details", "No Tool Calls have been recorded in this TUI yet.", kind="details")
                return True
            calls = "\n".join(
                f"  /details {call.call_id}  {call.tool_name} · {call.status}"
                for call in self._tool_activities.values()
            )
            self._append_transcript("details", f"Tool details:\n{calls}", kind="details")
            return True
        self._open_tool_detail(call_id)
        return True

    def _open_tool_detail(self, call_id: str) -> None:
        if self.runtime is None or self.runtime.session_store is None:
            self._append_transcript("error", "Could not load saved Tool Call details: Session storage is unavailable.", kind="error")
            return
        try:
            projection = self.runtime.session_store.replay(self.runtime.agent.session_id)
        except (OSError, ValueError) as error:
            self._append_transcript("error", f"Could not load saved Tool Call details: {error}", kind="error")
            return
        detail = _find_tool_call_detail(_tool_call_details(projection.events), call_id)
        if detail is None:
            self._append_transcript("details", f"No saved detail is available for Tool Call {call_id} yet.", kind="details")
            return
        self._append_transcript("details", self._detail_text(detail), kind="details")

    def _detail_text(self, detail: ToolCallDetail) -> str:
        assert self.runtime is not None
        body = _format_tool_call_detail(
            detail,
            repository=self.runtime.repository,
            validation_commands=(self.runtime.profile.validation_commands if self.runtime.profile else []),
        )
        return f"{detail.tool_name} · {detail.status}\n\n{body}"

    def _submit(self, buffer) -> bool:
        text = buffer.text.strip()
        if not text:
            return False
        if text.casefold() in {"exit", "quit", "/exit", "/quit"}:
            buffer.text = ""
            self.exit()
            return True
        if self._pending_permission is not None:
            buffer.text = ""
            self._answer_permission_text(text)
            return True
        if self._handle_details_command(text):
            buffer.text = ""
            return True
        if self._running:
            self._status = "A turn is already running. Press Ctrl+C to cancel it."
            self._refresh_status()
            return False
        if self.runtime is None:
            return False
        buffer.text = ""
        self._start_runtime_turn(text)
        return True

    def _start_runtime_turn(
        self,
        text: str,
        *,
        status: str = "正在思考…",
    ) -> None:
        self._follow_tail = True
        self._running = True
        self._streamed_turn = False
        self._stream_entry = None
        self._stream_buffer = ""
        self._status = status
        self._refresh_status()
        threading.Thread(target=self._run_turn, args=(text,), daemon=True).start()

    def _dismiss_welcome(self, buffer=None) -> None:
        """Keep the welcome block and clear only an optional input buffer."""
        if buffer is not None:
            buffer.text = ""
        self._refresh_status()
        self._invalidate()

    def _insert_input_newline(self, buffer=None) -> None:
        (buffer or self.input.buffer).insert_text("\n")

    def _on_input_text_changed(self, buffer) -> None:
        """Resize the input window immediately as lines are added or removed."""
        text = buffer.text
        logical_lines = max(1, text.count("\n") + 1)
        width = 0
        info = self.input.window.render_info
        if info is not None:
            width = info.window_width
        output = getattr(self.application, "output", None)
        if not width and output is not None:
            width = output.get_size().columns
        if width:
            prompt_width = len(self._input_prompt())
            visual_lines = 0
            for index, line in enumerate(text.split("\n")):
                line_width = len(line) + (prompt_width if index == 0 else 0)
                visual_lines += max(1, (line_width + width - 1) // width)
            logical_lines = max(logical_lines, visual_lines)
        self.input.window.height = Dimension(min=1, preferred=logical_lines, max=logical_lines)
        self._invalidate()

    def _input_prompt(self) -> str:
        if self._pending_permission is not None:
            return "Permission [1/2/3] > "
        return "❯ "

    def _run_turn(self, text: str) -> None:
        assert self.runtime is not None
        try:
            self.runtime.run_turn(text)
        except Exception as error:
            self._updates.put(("worker_error", str(error)))
            self._invalidate()
        finally:
            self._updates.put(("turn_finished", None))
            self._invalidate()

    def _invalidate(self) -> None:
        invalidate = getattr(self.application, "invalidate", None)
        if invalidate is not None:
            invalidate()

    def _drain_updates(self, _sender=None) -> None:
        while True:
            try:
                kind, value = self._updates.get_nowait()
            except queue.Empty:
                break
            if kind == "event":
                self._apply_event(value)  # type: ignore[arg-type]
            elif kind == "permission":
                self._pending_permission = value  # type: ignore[assignment]
                self._status = "Waiting for permission"
                self._refresh_permission()
            elif kind == "worker_error":
                self._status = f"Turn failed: {value}"
                self._append_transcript("error", f"Runtime error: {value}", kind="error")
            elif kind == "turn_finished":
                self._running = False
                self._refresh_status()
        self._flush_stream_buffer()
        self._refresh_status()

    def _apply_event(self, event: RuntimeEvent) -> None:
        event_type = event.event_type
        payload = event.payload
        if event_type is not RuntimeEventType.ASSISTANT_TOKEN:
            self._flush_stream_buffer()
        if event_type is RuntimeEventType.TURN_STARTED:
            self._turn_started_at = time.monotonic()
            self._turn_tool_count = 0
            self._stream_entry = None
            self._follow_tail = True
            user_input = str(payload.get("user_input", ""))
            self._append_transcript("you", user_input, kind="user")
            self._stream_buffer = ""
            self._status = "Thinking…"
        elif event_type is RuntimeEventType.PROVIDER_STARTED:
            self._status = "Requesting model…"
        elif event_type is RuntimeEventType.ASSISTANT_TOKEN:
            self._streamed_turn = True
            self._stream_buffer += str(payload.get("token", ""))
            self._status = "Streaming answer…"
        elif event_type is RuntimeEventType.TOOL_REQUESTED:
            call_id = event.tool_call_id or event.event_id
            activity = _ToolActivity(
                call_id,
                str(payload.get("tool_name", "tool")),
                started_at=time.monotonic(),
                arguments=str(payload.get("arguments", "")),
            )
            self._tool_activities[call_id] = activity
            self._turn_tool_count += 1
            self._stream_entry = None
            self._status = f"Running {payload.get('tool_name', 'tool')}…"
            activity.entry = self._append_transcript(
                f"tool {activity.tool_name} running",
                "",
                kind="tool",
                call_id=call_id,
            )
        elif event_type is RuntimeEventType.EXECUTION_CONTROL_ASSESSED:
            self._apply_control(event)
        elif event_type is RuntimeEventType.TOOL_COMPLETED:
            call_id = event.tool_call_id or event.event_id
            activity = self._tool_activities.get(call_id)
            if activity is None:
                activity = self._tool_activities[call_id] = _ToolActivity(
                    call_id,
                    str(payload.get("tool_name", "tool")),
                    started_at=time.monotonic(),
                )
                activity.entry = self._append_transcript(
                    f"tool {activity.tool_name} running",
                    "",
                    kind="tool",
                    call_id=call_id,
                )
            result = str(payload.get("result", ""))
            activity.result = result
            activity.status = "cancelled" if payload.get("interrupted") else _tool_status(result, payload.get("tool_status"))
            activity.finished_at = time.monotonic()
            self._status = f"{activity.tool_name}: {activity.status}"
            self._refresh_tool_entry(activity)
        elif event_type is RuntimeEventType.TURN_COMPLETED:
            if not self._streamed_turn and payload.get("content"):
                self._append_transcript("TechPilot", str(payload["content"]), kind="assistant")
            self._status = "Completed"
        elif event_type is RuntimeEventType.TURN_INTERRUPTED:
            self._append_transcript("error", "Turn cancelled", kind="error")
            self._status = "Cancelled"
        elif event_type is RuntimeEventType.TURN_FAILED:
            self._append_transcript("error", f"Turn failed: {payload.get('error', 'unknown error')}", kind="error")
            self._status = "Failed"
        elif event_type is RuntimeEventType.TURN_LIMIT_REACHED:
            self._append_transcript("error", f"Limit reached: {payload.get('limit', 'runtime limit')}", kind="error")
            self._status = "Limit reached"

    def _apply_control(self, event: RuntimeEvent) -> None:
        required = str(event.payload.get("required_control", ""))
        call_id = event.tool_call_id or event.event_id
        activity = self._tool_activities.get(call_id)
        if activity is not None and required:
            activity.status = required
            self._refresh_tool_entry(activity)
        if required == "block":
            reasons = event.payload.get("reasons", [])
            rendered = "\n".join(
                f"- {item.get('message', 'control rule')}：{'；'.join(str(value) for value in item.get('evidence', []))}"
                for item in reasons if isinstance(item, dict)
            ) or "- No additional evidence"
            self._append_transcript(
                "BLOCKED",
                f"The Tool Call was not executed.\nReasons and evidence:\n{rendered}",
                kind="blocked",
            )
            self._status = "Blocked by execution control"

    def _refresh_permission(self) -> None:
        pending = self._pending_permission
        if pending is None:
            return
        request = pending.request
        choices = "1 allow once · 2 allow for session · 3 deny"
        if request.command_prefix:
            choices = f"{choices} · prefix: {' '.join(request.command_prefix)}"
        preview = f"\n\nTrusted Diff:\n{request.trusted_preview}" if request.trusted_preview else ""
        self._append_transcript(
            "permission requested",
            f"Tool: {request.tool_name}\nEffect: {request.effect.value}\nScope: {request.scope}\n"
            f"Reason: {request.reason}{preview}\n\nReply: {choices}",
            kind="permission",
        )

    def _answer_permission_text(self, text: str) -> None:
        pending = self._pending_permission
        if pending is None:
            return
        normalized = text.strip().lower().replace("_", " ")
        if normalized in {"a", "allow", "allow once", "once", "1"}:
            self.answer_permission(PermissionDecision.allow("user allowed this operation"))
            return
        if normalized in {"s", "session", "allow session", "2"}:
            self.answer_permission(PermissionDecision.allow("user allowed this operation for the session", PermissionGrantScope.SESSION))
            return
        if normalized in {"p", "prefix", "allow prefix"} and pending.request.command_prefix:
            self.answer_permission(PermissionDecision.allow("user allowed this command prefix for the session", PermissionGrantScope.PREFIX))
            return
        if normalized in {"d", "n", "deny", "no", "3"}:
            self.answer_permission(PermissionDecision.deny("user denied this operation"))
            return
        self._append_transcript("permission", "Reply with: allow / session / prefix / deny", kind="permission")

    def _refresh_status(self) -> None:
        if self.runtime is None:
            self._status_area.text = self._status
            self._topbar.text = "TechPilot"
            self._activity_area.text = self._activity_text()
            return
        self._status_area.text = self._status
        self._topbar.text = f"TechPilot   ·   {self.runtime.config.model}   ·   {self.runtime.agent.session_id[:8]}"
        self._activity_area.text = self._activity_text()

    def _activity_text(self) -> str:
        if not self._follow_tail:
            return "↑ Viewing history · Ctrl+End latest"
        if self._pending_permission is not None:
            return "! Waiting for permission"
        if self._running:
            elapsed = time.monotonic() - self._turn_started_at if self._turn_started_at else 0.0
            tool_count = f" · {self._turn_tool_count} tool" if self._turn_tool_count else ""
            return f"✻ {self._status} · {elapsed:.1f}s{tool_count} · Ctrl+O latest tool"
        return ""

    def _append_transcript(
        self,
        label: str,
        body: str,
        *,
        kind: str = "system",
        call_id: str | None = None,
    ) -> _TranscriptEntry:
        entry = _TranscriptEntry(label, body, kind, call_id)
        self._entries.append(entry)
        self._trim_transcript()
        self._invalidate()
        return entry

    def _trim_transcript(self) -> None:
        while len(self.transcript_text) > self._MAX_CONVERSATION_CHARS and len(self._entries) > 1:
            self._entries.pop(0)

    @property
    def transcript_text(self) -> str:
        entries = [self._display_item_plain_text(item) for item in self._display_items()]
        return "\n\n".join([_WELCOME_TEXT, *entries])

    def _transcript_cursor(self) -> Point | None:
        rendered = "".join(fragment[1] for fragment in self._transcript_fragments())
        last_line = max(0, rendered.count("\n"))
        if self._follow_tail:
            return Point(x=0, y=last_line)
        # When not following, keep the cursor on the last visible line so the
        # Window's scroll clamping preserves vertical_scroll instead of
        # resetting it to the top (the default cursor position) on re-render.
        info = self.conversation.render_info
        height = info.window_height if info is not None else 1
        return Point(x=0, y=min(last_line, self.conversation.vertical_scroll + max(0, height - 1)))

    def _entry_plain_text(self, entry: _TranscriptEntry) -> str:
        if entry.kind == "tool":
            activity = self._tool_activities.get(entry.call_id or "")
            if activity is None:
                return f"▸ {entry.label}"
            return "\n".join(self._tool_plain_lines(activity, prefix=""))
        lines = [f"{_message_marker(entry.kind)} {entry.label}"]
        if entry.body:
            lines.extend(f"  {line.plain}" for line in _markdown_lines(entry.body))
        return "\n".join(lines)

    def _transcript_fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = list(self._welcome_fragments())
        items = self._display_items()
        if items:
            fragments.append(("class:transcript", "\n\n"))
        for index, item in enumerate(items):
            fragments.extend(self._display_item_fragments(item))
            if index < len(items) - 1:
                fragments.append(("class:transcript", "\n\n"))
        return fragments

    def _welcome_fragments(self) -> StyleAndTextTuples:
        """Render the startup panel without adding a second UI framework."""
        info = self.conversation.render_info
        available_width = getattr(info, "window_width", 0) if info is not None else 0
        width = min(112, max(52, available_width - 4 if available_width else 88))
        inner = width - 2
        title = "─ TechPilot · Local-first Chat "
        title_line = f"┌{title}{'─' * max(1, inner - len(title))}┐\n"
        fragments: StyleAndTextTuples = [("class:welcome-border", title_line)]

        def row(
            left: str = "",
            right: str = "",
            *,
            left_style: str = "welcome-copy",
            right_style: str = "welcome-tips",
        ) -> None:
            column_width = max(22, (inner - 4) // 2)
            left_text = _fit_text(left, column_width - 2)
            right_text = _fit_text(right, inner - column_width - 2)
            gap = inner - column_width - 2
            fragments.extend([
                ("class:welcome-border", "│"),
                (f"class:{left_style}", f"  {left_text:<{column_width - 2}}"),
                ("class:welcome-copy", "  "),
                (f"class:{right_style}", f"{right_text:<{gap}}"),
                ("class:welcome-border", "│\n"),
            ])

        model = str(getattr(self.runtime.config, "model", "default")) if self.runtime is not None else "default"
        repository = str(getattr(self.runtime, "repository", "current directory")) if self.runtime is not None else "current directory"
        session_id = str(getattr(getattr(self.runtime, "agent", None), "session_id", "not started")) if self.runtime is not None else "not started"

        row("", "")
        row("TechPilot", "Getting started", left_style="welcome-brand", right_style="welcome-brand")
        row("Local-first coding agent", "Enter             send message")
        row("Chat · tools · safeguards", "Shift+Enter       new line")
        row("", "Ctrl+O            latest tool")
        row(f"Model: {model}", "/details <id>     saved detail", left_style="welcome-meta")
        row(f"Workspace: {repository}", "Ctrl+C            cancel / exit", left_style="welcome-meta")
        row(f"Session: {session_id[:8]}", "", left_style="welcome-meta")
        row("", "")
        row("Ready when you are", "Type a message to begin", left_style="welcome-hint", right_style="welcome-hint")
        fragments.append(("class:welcome-border", f"└{'─' * inner}┘"))
        return fragments

    def _display_items(self) -> list[_TranscriptEntry | _ToolGroup]:
        items: list[_TranscriptEntry | _ToolGroup] = []
        index = 0
        while index < len(self._entries):
            entry = self._entries[index]
            if not self._is_groupable_tool_entry(entry):
                items.append(entry)
                index += 1
                continue
            group_entries = [entry]
            index += 1
            while index < len(self._entries) and self._is_groupable_tool_entry(self._entries[index]):
                group_entries.append(self._entries[index])
                index += 1
            if len(group_entries) == 1:
                items.append(group_entries[0])
                continue
            call_ids = tuple(entry.call_id for entry in group_entries if entry.call_id)
            items.append(_ToolGroup(group_id=call_ids[0], call_ids=call_ids))
        return items

    def _is_groupable_tool_entry(self, entry: _TranscriptEntry) -> bool:
        if entry.kind != "tool" or not entry.call_id:
            return False
        activity = self._tool_activities.get(entry.call_id)
        return activity is not None and activity.status in {"running", "completed"}

    def _display_item_plain_text(self, item: _TranscriptEntry | _ToolGroup) -> str:
        if isinstance(item, _ToolGroup):
            return self._tool_group_plain_text(item)
        return self._entry_plain_text(item)

    def _display_item_fragments(self, item: _TranscriptEntry | _ToolGroup) -> StyleAndTextTuples:
        if isinstance(item, _ToolGroup):
            return self._tool_group_fragments(item)
        return self._entry_fragments(item)

    def _entry_fragments(self, entry: _TranscriptEntry) -> StyleAndTextTuples:
        style = f"class:{entry.kind}"
        if entry.kind == "tool":
            activity = self._tool_activities.get(entry.call_id or "")
            if activity is None:
                return [("class:tool", f"▸ {entry.label}")]
            return self._tool_fragments(activity, prefix="")
        # Keep the user marker/label on the terminal background. Only the
        # actual question body receives the subtle user-message highlight.
        header_style = "class:transcript" if entry.kind == "user" else style
        fragments: StyleAndTextTuples = [(header_style, f"{_message_marker(entry.kind)} {entry.label}")]
        for line in _markdown_lines(entry.body):
            fragments.append((style, "\n  "))
            fragments.extend(_markdown_fragments(line, fallback_style=style))
            if entry.kind == "user":
                info = self.conversation.render_info
                width = getattr(info, "window_width", 0) if info is not None else 0
                if width:
                    fragments.append((style, " " * max(0, width - get_cwidth(line.plain) - 2)))
        return fragments

    def _tool_plain_lines(self, activity: _ToolActivity, *, prefix: str) -> list[str]:
        arrow = "▾" if activity.expanded else "▸"
        lines = [f"{prefix}{arrow} {self._tool_label(activity)}"]
        if activity.expanded:
            lines.extend(f"{prefix}  {line}" for line in self._tool_output_lines(activity))
        return lines

    def _tool_fragments(self, activity: _ToolActivity, *, prefix: str) -> StyleAndTextTuples:
        arrow = "▾" if activity.expanded else "▸"
        style = self._tool_style(activity)
        fragments: StyleAndTextTuples = [
            (style, prefix),
            ("class:tool-toggle", f"{arrow} ", self._toggle_tool_handler(activity.call_id)),
            (style, self._tool_label(activity)),
        ]
        if activity.expanded:
            for line in self._tool_output_lines(activity):
                fragments.append((style, f"\n{prefix}  {line}"))
        return fragments

    def _tool_group_plain_text(self, group: _ToolGroup) -> str:
        expanded = group.group_id in self._expanded_tool_groups
        arrow = "▾" if expanded else "▸"
        lines = [f"{arrow} {self._tool_group_label(group)}"]
        if expanded:
            for call_id in group.call_ids:
                lines.extend(self._tool_plain_lines(self._tool_activities[call_id], prefix="  "))
        return "\n".join(lines)

    def _tool_group_fragments(self, group: _ToolGroup) -> StyleAndTextTuples:
        expanded = group.group_id in self._expanded_tool_groups
        arrow = "▾" if expanded else "▸"
        fragments: StyleAndTextTuples = [
            ("class:tool-group-toggle", f"{arrow} ", self._toggle_tool_group_handler(group.group_id)),
            ("class:tool-group", self._tool_group_label(group)),
        ]
        if expanded:
            for call_id in group.call_ids:
                fragments.append(("class:tool-group", "\n"))
                fragments.extend(self._tool_fragments(self._tool_activities[call_id], prefix="  "))
        return fragments

    def _tool_group_label(self, group: _ToolGroup) -> str:
        activities = [self._tool_activities[call_id] for call_id in group.call_ids]
        counts: dict[str, int] = {}
        for activity in activities:
            counts[activity.tool_name] = counts.get(activity.tool_name, 0) + 1
        names = ", ".join(f"{name} ×{count}" for name, count in counts.items())
        completed = sum(activity.status == "completed" for activity in activities)
        running = len(activities) - completed
        if running:
            outcome = f"{completed} completed · {running} running"
            duration = ""
        else:
            outcome = "completed"
            started = min(activity.started_at for activity in activities)
            finished = max(activity.finished_at or time.monotonic() for activity in activities)
            duration = f" · {finished - started:.2f}s" if started else ""
        return f"{len(activities)} tools · {names} · {outcome}{duration}"

    def _tool_label(self, activity: _ToolActivity) -> str:
        duration = ""
        if activity.finished_at is not None and activity.started_at:
            duration = f" · {activity.finished_at - activity.started_at:.2f}s"
        marker = {
            "running": "◌",
            "completed": "✓",
            "cancelled": "×",
        }.get(activity.status, "!")
        return f"{marker} {activity.tool_name} · {activity.status}{duration}"

    def _tool_style(self, activity: _ToolActivity) -> str:
        if activity.status == "running":
            return "class:tool-running"
        if activity.status == "completed":
            return "class:tool-completed"
        return "class:tool-problem"

    def _tool_output_lines(self, activity: _ToolActivity) -> list[str]:
        lines: list[str] = []
        if activity.arguments:
            lines.append(f"arguments: {_compact_text(activity.arguments, 600)}")
        if activity.result:
            lines.append("output:")
            lines.extend(_display_output_lines(activity.result, call_id=activity.call_id))
        if not lines:
            lines.append("No tool output has been recorded yet.")
        lines.append(f"saved detail: /details {activity.call_id}")
        return lines

    def _toggle_tool_handler(self, call_id: str):
        def toggle(mouse_event):
            if mouse_event.event_type is MouseEventType.MOUSE_UP:
                activity = self._tool_activities.get(call_id)
                if activity is not None:
                    activity.expanded = not activity.expanded
                    self._follow_tail = False
                    self._invalidate()

        return toggle

    def _toggle_tool_group_handler(self, group_id: str):
        def toggle(mouse_event):
            if mouse_event.event_type is MouseEventType.MOUSE_UP:
                if group_id in self._expanded_tool_groups:
                    self._expanded_tool_groups.remove(group_id)
                else:
                    self._expanded_tool_groups.add(group_id)
                self._follow_tail = False
                self._invalidate()

        return toggle

    def _toggle_latest_tool_detail(self) -> None:
        """Keyboard equivalent of opening the newest visible Tool Call detail."""
        if not self._tool_activities:
            return
        latest = self._tool_activities[next(reversed(self._tool_activities))]
        for group in self._display_items():
            if isinstance(group, _ToolGroup) and latest.call_id in group.call_ids:
                self._expanded_tool_groups.add(group.group_id)
                break
        latest.expanded = not latest.expanded
        self._follow_tail = False
        self._invalidate()

    def _refresh_tool_entry(self, activity: _ToolActivity) -> None:
        self._invalidate()

    def _scroll_transcript(self, direction: int, *, page: bool = False) -> None:
        info = self.conversation.render_info
        if info is None:
            return
        max_scroll = max(0, info.content_height - info.window_height)
        if not max_scroll:
            return
        # A wheel notch moves a few lines; a page key moves almost a full window.
        step = max(1, info.window_height - 1) if page else 3
        self.conversation.vertical_scroll = min(max_scroll, max(0, self.conversation.vertical_scroll + direction * step))
        self._follow_tail = direction > 0 and self.conversation.vertical_scroll >= max_scroll
        self._invalidate()

    def _scroll_to_top(self) -> None:
        self._follow_tail = False
        self.conversation.vertical_scroll = 0
        self._invalidate()

    def _scroll_to_bottom(self) -> None:
        self._follow_tail = True
        info = self.conversation.render_info
        if info is not None:
            self.conversation.vertical_scroll = max(0, info.content_height - info.window_height)
        self._invalidate()

    def _flush_stream_buffer(self) -> None:
        if not self._stream_buffer:
            return
        text = self._stream_buffer
        self._stream_buffer = ""
        if self._stream_entry is None:
            self._stream_entry = self._append_transcript("TechPilot", text, kind="assistant")
        else:
            self._stream_entry.body += text
            self._invalidate()

def tui_supported() -> bool:
    """Return whether this process has an interactive terminal for the TUI."""

    import sys

    return bool(sys.stdin.isatty() and sys.stdout.isatty())


_WELCOME_TEXT = """✦ TechPilot
  Local-first coding agent

  Chat with your repository, inspect tool activity, and keep
  every write or command behind explicit safeguards.

  Type a message to begin · Ctrl+C to exit"""
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")
_QUOTE_RE = re.compile(r"^(\s*)>\s?(.*)$")
_INLINE_MARKDOWN_RE = re.compile(r"(`[^`]+`|\*\*.+?\*\*|__.+?__)")


def _tui_style() -> Style:
    """Return the shared palette for the live viewport and exit transcript."""

    return Style.from_dict({
        "topbar": "bg:#0f1014 #7bba55",
        "status": "bg:#0f1014 #6e6d72",
        "activity": "bg:#0f1014 #527c3b",
        "divider": "bg:#0f1014 #626263",
        "input": "bg:#0f1014 #cfd1d6",
        "transcript": "bg:#0f1014 #cfd1d6",
        "user": "bg:#34363b #f0f1f3",
        "assistant": "#eee9df",
        "tool": "#85898f",
        "tool-toggle": "bold #a2a5aa",
        "tool-group": "#7f858d",
        "tool-group-toggle": "bold #b0b6bf",
        "tool-running": "#88c5d1",
        "tool-completed": "#8fbe83",
        "tool-problem": "#e07c7c",
        "permission": "#d7a65b",
        "blocked": "#e06d6d",
        "error": "#e06d6d",
        "details": "#9298a1",
        "choice": "#d7a65b",
        "welcome-border": "#7bba55",
        "welcome-brand": "bold #f0c674",
        "welcome-subtitle": "#9aa0a6",
        "welcome-copy": "#e5e7eb",
        "welcome-meta": "#9fc8ff",
        "welcome-tips": "#cfd1d6",
        "welcome-hint": "bold #7bba55",
        "markdown-heading": "bold #f0c674",
        "markdown-list": "#e8eaed",
        "markdown-quote": "#a9b2bd",
        "markdown-code": "bg:#20242a #b9ccec",
        "markdown-code-fence": "#707780",
        "markdown-inline-code": "bg:#252b34 #9fc8ff",
        "markdown-strong": "bold #ffffff",
        "markdown-rule": "#626263",
    })


def _message_marker(kind: str) -> str:
    return {
        "user": "❯",
        "assistant": "●",
        "permission": "!",
        "blocked": "!",
        "error": "×",
        "details": "·",
    }.get(kind, "·")


def _fit_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return "…"
    return text[: width - 1] + "…"


def _markdown_lines(text: str) -> list[_MarkdownLine]:
    """Render the useful Markdown subset without adding a UI dependency."""
    lines: list[_MarkdownLine] = []
    in_code_block = False
    for source in text.splitlines():
        if source.lstrip().startswith("```"):
            language = source.lstrip()[3:].strip()
            if in_code_block:
                lines.append(_MarkdownLine("└", "└", "markdown-code-fence"))
            else:
                marker = f"┌ {language}" if language else "┌"
                lines.append(_MarkdownLine(marker, marker, "markdown-code-fence"))
            in_code_block = not in_code_block
            continue
        if in_code_block:
            lines.append(_MarkdownLine(source, source, "markdown-code"))
            continue
        if heading := _HEADING_RE.match(source):
            raw = heading.group(2)
            lines.append(_MarkdownLine(raw, _strip_inline_markdown(raw), "markdown-heading"))
            continue
        if bullet := _BULLET_RE.match(source):
            raw = f"{bullet.group(1)}• {bullet.group(2)}"
            lines.append(_MarkdownLine(raw, _strip_inline_markdown(raw), "markdown-list"))
            continue
        if quote := _QUOTE_RE.match(source):
            raw = f"{quote.group(1)}▎ {quote.group(2)}"
            lines.append(_MarkdownLine(raw, _strip_inline_markdown(raw), "markdown-quote"))
            continue
        if source.strip() in {"---", "***", "___"}:
            lines.append(_MarkdownLine("─", "─", "markdown-rule"))
            continue
        lines.append(_MarkdownLine(source, _strip_inline_markdown(source)))
    return lines


def _strip_inline_markdown(text: str) -> str:
    return _INLINE_MARKDOWN_RE.sub(
        lambda match: match.group(0)[1:-1] if match.group(0).startswith("`") else match.group(0)[2:-2],
        text,
    )


def _markdown_fragments(line: _MarkdownLine, *, fallback_style: str) -> StyleAndTextTuples:
    style = fallback_style if line.style == "markdown-body" else f"class:{line.style}"
    fragments: StyleAndTextTuples = []
    for part in _INLINE_MARKDOWN_RE.split(line.raw):
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            fragments.append(("class:markdown-inline-code", part[1:-1]))
        elif (part.startswith("**") and part.endswith("**")) or (part.startswith("__") and part.endswith("__")):
            fragments.append(("class:markdown-strong", part[2:-2]))
        else:
            fragments.append((style, part))
    return fragments


def _compact_text(text: str, max_chars: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _display_output_lines(text: str, *, call_id: str, max_chars: int = 4_000) -> list[str]:
    if len(text) > max_chars:
        text = (
            text[:max_chars]
            + f"\n… output truncated in the TUI; use /details {call_id} for saved full detail."
        )
    return text.splitlines() or ["(empty)"]
