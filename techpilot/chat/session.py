"""Minimal event-driven terminal Chat for TechPilot C2."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from techpilot.engine.context import estimate_tokens
from techpilot.engine.events import RuntimeEvent, RuntimeEventType
from techpilot.engine.tool_results import ToolResult, ToolStatus, display_tool_status
from techpilot.engine.tools.edit import _changed_files

from ..runtime import (
    CompletionCriterionKind,
    EffectJournalState,
    EffectJournalStore,
    GoalStageAssessmentStatus,
    LongTaskStateError,
    LongTaskStatus,
    LongTaskStore,
    LongTaskWorkflow,
    TaskRuntime,
)

InputFunction = Callable[[str], str]
_DETAIL_PAGE_SIZE = 8_000


@dataclass
class ToolCallDetail:
    """A display-only projection assembled from existing Runtime Events."""

    call_id: str
    tool_name: str = "unknown"
    arguments: object = None
    requested_at: str | datetime | None = None
    result: str | None = None
    completed_at: str | datetime | None = None
    interrupted: bool = False
    execution_control: dict[str, object] | None = None
    tool_status: str | None = None
    result_facts: dict[str, object] | None = None

    @property
    def status(self) -> str:
        if self.result is None:
            return "running"
        return "interrupted" if self.interrupted else _tool_status(self.result, self.tool_status)

    @property
    def duration_seconds(self) -> float | None:
        started = _as_datetime(self.requested_at)
        completed = _as_datetime(self.completed_at)
        if started is None or completed is None:
            return None
        return max(0.0, (completed - started).total_seconds())


@dataclass
class _CollapsedToolCall:
    """Ephemeral terminal state for a compact, per-turn tool-call summary."""

    call_id: str
    tool_name: str
    started_at: float
    status: str | None = None
    completed_at: float | None = None


class SlashCommandCompleter(Completer):
    """Offer local Chat commands while the user is typing a slash command."""

    _COMMANDS = (
        ("/help", "显示命令帮助"),
        ("/status", "显示会话与上下文摘要"),
        ("/tools", "查看可用工具"),
        ("/files", "查看本次进程修改的文件"),
        ("/diff", "查看当前 Git 差异"),
        ("/tokens", "查看 Token 用量与估算费用"),
        ("/compact", "压缩模型上下文"),
        ("/save", "查看自动保存的 Session"),
        ("/sessions", "列出可恢复的 Session"),
        ("/session", "查看指定 Session"),
        ("/resume", "恢复指定 Session"),
        ("/details", "切换工具结果详情显示"),
        ("/model", "查看或切换模型"),
        ("/goal", "创建、继续或管理可恢复的长程目标"),
        ("/clear", "清空当前对话消息"),
        ("/exit", "退出 Chat"),
        ("/quit", "退出 Chat"),
    )

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text or "\t" in text:
            return
        for command, description in self._COMMANDS:
            if command.startswith(text):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )


class TerminalEventSink:
    """Render Runtime Events without understanding or duplicating AgentLoop."""

    def __init__(self, console: Console | None = None, *, result_limit: int = 800):
        self.console = console or Console()
        self.result_limit = result_limit
        self._tool_started: dict[str, tuple[float, str]] = {}
        self._collapsed_tool_calls: dict[str, _CollapsedToolCall] = {}
        self._blocked_tool_calls: set[str] = set()
        self._streamed = False
        self.last_turn_streamed = False
        self.show_full_results = False

    def emit(self, event: RuntimeEvent) -> None:
        if event.event_type is RuntimeEventType.TURN_STARTED:
            self._streamed = False
            self.last_turn_streamed = False
            self._tool_started.clear()
            self._collapsed_tool_calls.clear()
            self._blocked_tool_calls.clear()
        elif event.event_type is RuntimeEventType.ASSISTANT_TOKEN:
            self._render_collapsed_tool_calls()
            self.console.print(event.payload.get("token", ""), end="", markup=False, highlight=False)
            self._streamed = True
            self.last_turn_streamed = True
        elif event.event_type is RuntimeEventType.TOOL_REQUESTED:
            call_id = event.tool_call_id or event.event_id
            name = event.payload.get("tool_name", "unknown")
            started_at = time.monotonic()
            self._tool_started[call_id] = (started_at, str(name))
            if self.show_full_results:
                arguments = _brief_arguments(event.payload.get("arguments", {}))
                prefix = "\n" if self._streamed else ""
                self.console.print(
                    f"{prefix}[dim]→ {name}({arguments}) · running · details: /details {call_id}[/dim]"
                )
                self._streamed = False
            else:
                self._collapsed_tool_calls[call_id] = _CollapsedToolCall(
                    call_id=call_id,
                    tool_name=str(name),
                    started_at=started_at,
                )
        elif event.event_type is RuntimeEventType.EXECUTION_CONTROL_ASSESSED:
            required = event.payload.get("required_control")
            if required == "block":
                self._blocked_tool_calls.add(event.tool_call_id or event.event_id)
                reasons = event.payload.get("reasons", [])
                details = []
                for reason in reasons:
                    if not isinstance(reason, dict):
                        continue
                    evidence = reason.get("evidence", [])
                    evidence_text = "；".join(str(item) for item in evidence)
                    details.append(f"- {reason.get('message', '控制规则')}：{evidence_text}")
                self.console.print(Panel(
                    "\n".join(["该 Tool Call 未执行。", "原因与证据：", *(details or ["- 无额外详情"])]),
                    title="操作已阻断",
                    border_style="red",
                ))
        elif event.event_type is RuntimeEventType.TOOL_COMPLETED:
            call_id = event.tool_call_id or event.event_id
            started = self._tool_started.pop(call_id, None)
            elapsed = f" in {time.monotonic() - started[0]:.2f}s" if started is not None else ""
            tool_name = str(event.payload.get("tool_name") or (started[1] if started else "tool"))
            result = str(event.payload.get("result", ""))
            status = "interrupted" if event.payload.get("interrupted") else _tool_status(result, event.payload.get("tool_status"))
            block_was_rendered = call_id in self._blocked_tool_calls
            self._blocked_tool_calls.discard(call_id)
            style = "yellow" if status in {"interrupted", "not executed", "effect unknown"} else ("red" if status in {"error", "denied"} else "green")
            if not self.show_full_results and status == "completed":
                call = self._collapsed_tool_calls.get(call_id)
                if call is not None:
                    call.status = status
                    call.completed_at = time.monotonic()
                    return
            self._render_collapsed_tool_calls()
            self._collapsed_tool_calls.pop(call_id, None)
            self.console.print(
                f"[{style}]← {tool_name}: {status}{elapsed} · details: /details {call_id}[/{style}]"
            )
            if result:
                if status == "not executed":
                    self.console.print("[dim]未执行：达到运行限制；详情保留在 Session 中。[/dim]")
                elif (status in {"error", "denied", "interrupted", "effect unknown"} and not block_was_rendered) or self.show_full_results:
                    self.console.print(f"[dim]{_summarize(result, self.result_limit)}[/dim]")
        elif event.event_type is RuntimeEventType.CONTEXT_COMPRESSED:
            self.console.print("[dim]Context compressed.[/dim]")
        elif event.event_type is RuntimeEventType.TURN_COMPLETED:
            self._render_collapsed_tool_calls()
            if self._streamed:
                self.console.print()
                self._streamed = False
        elif event.event_type is RuntimeEventType.TURN_FAILED:
            self._render_collapsed_tool_calls()
            self.console.print(f"[red]Turn failed: {event.payload.get('error', 'unknown error')}[/red]")
        elif event.event_type is RuntimeEventType.TURN_INTERRUPTED:
            self._render_collapsed_tool_calls()
            self.console.print("[yellow]Current turn interrupted.[/yellow]")
        elif event.event_type is RuntimeEventType.TURN_LIMIT_REACHED:
            self._render_collapsed_tool_calls()
            limit = event.payload.get("limit") or "tool-call rounds"
            self.console.print(f"[yellow]Runtime limit reached: {limit}.[/yellow]")

    def _render_collapsed_tool_calls(self) -> None:
        completed = [
            call for call in self._collapsed_tool_calls.values()
            if call.status == "completed" and call.completed_at is not None
        ]
        if not completed:
            return
        names: list[str] = []
        for call in completed:
            if call.tool_name not in names:
                names.append(call.tool_name)
        name_summary = "、".join(
            f"{name} × {sum(call.tool_name == name for call in completed)}" for name in names
        )
        started_at = min(call.started_at for call in completed)
        completed_at = max(call.completed_at for call in completed if call.completed_at is not None)
        prefix = "\n" if self._streamed else ""
        self.console.print(
            f"{prefix}[dim]工具调用已折叠：{name_summary}（{len(completed)} 项完成，"
            f"{completed_at - started_at:.2f}s）· /details 查看并选择详情[/dim]"
        )
        self._streamed = False
        for call in completed:
            self._collapsed_tool_calls.pop(call.call_id, None)


class ChatSession:
    """Own terminal input and commands while Agent owns the conversation loop."""

    def __init__(
        self,
        runtime: TaskRuntime,
        *,
        console: Console | None = None,
        input_fn: InputFunction | None = None,
    ):
        self.runtime = runtime
        self.console = console or Console()
        self.input_fn = input_fn
        self._prompt_session: PromptSession[str] | None = None
        self._goal_workflow: LongTaskWorkflow | None = None

    def run(self) -> int:
        self.show_startup()
        while True:
            try:
                user_input = self._read_input("You > ").strip()
            except EOFError:
                self._pause_active_goal("Chat input closed before Goal completion")
                self.console.print("\nBye!")
                return 0
            except KeyboardInterrupt:
                self._pause_active_goal("Chat interrupted before Goal completion")
                self.console.print("\nBye!")
                return 0

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
                self._pause_active_goal("Chat exited before Goal completion")
                self.console.print("Bye!")
                return 0
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            try:
                response = self._run_user_turn(user_input)
                # Providers used by embedders may not stream token callbacks.
                sink = self.runtime.agent.event_sink
                if not getattr(sink, "last_turn_streamed", False) and response:
                    self.console.print(Markdown(response))
            except KeyboardInterrupt:
                if self._goal_workflow is not None:
                    self.console.print("[yellow]Goal step cancelled. Use /goal status before continuing.[/yellow]")
                else:
                    self.console.print("[yellow]Turn cancelled. The chat session is still active.[/yellow]")
            except Exception as error:
                self.console.print(f"[red]Error: {error}[/red]")
            finally:
                self._ensure_session_persisted()

    def _read_input(self, prompt: str) -> str:
        if self.input_fn is not None:
            return self.input_fn(prompt)
        if self._prompt_session is None:
            self._prompt_session = PromptSession()
        return self._prompt_session.prompt(
            prompt,
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
        )

    def show_startup(self) -> None:
        self._show_recovery_notices()
        profile = self.runtime.profile
        profile_lines = [f"Language: {profile.language}"] if profile else []
        if profile:
            profile_lines.extend([
                f"Frameworks: {', '.join(profile.frameworks) or '(none detected)'}",
                f"Entrypoints: {', '.join(profile.entrypoints[:5]) or '(none detected)'}",
                f"Tests: {', '.join(profile.test_files[:5]) or '(none detected)'}",
                "Validation: " + (", ".join(" ".join(c) for c in profile.validation_commands) or "(none detected)"),
            ])
        if self.runtime.profile_warning:
            profile_lines.append(f"Profile warning: {self.runtime.profile_warning}")
        body = "\n".join([
            "[bold]TechPilot Chat[/bold]",
            f"Repository: [cyan]{self.runtime.repository}[/cyan]",
            f"Model: [cyan]{self.runtime.config.model}[/cyan]",
            "操作保护：写入和命令会在执行前按实际影响进行确认或阻断。",
            f"Session: [dim]{self.runtime.agent.session_id[:8]}[/dim]",
            f"Tools: {', '.join(tool.name for tool in self.runtime.tools)}",
            *profile_lines,
            "Type /help for commands; Ctrl+C cancels a turn; /exit exits Chat.",
        ])
        self.console.print(Panel(body, border_style="blue"))

    def _handle_command(self, command_line: str) -> None:
        command, _, argument = command_line.partition(" ")
        handlers = {
            "/help": self._show_help,
            "/status": self._show_status,
            "/tools": self._show_tools,
            "/files": self._show_files,
            "/diff": self._show_diff,
            "/tokens": self._show_tokens,
            "/compact": self._compact,
            "/save": self._show_save,
            "/sessions": self._show_sessions,
            "/session": lambda: self._session_command(argument.strip()),
            "/resume": lambda: self._resume(argument.strip()),
            "/details": lambda: self._details(argument.strip()),
            "/model": lambda: self._model(argument.strip()),
            "/goal": lambda: self._goal(argument.strip()),
            "/clear": self._clear,
        }
        handler = handlers.get(command.lower())
        if handler is None:
            self.console.print(f"[yellow]Unknown command: {command} (try /help)[/yellow]")
        else:
            handler()

    def _show_help(self) -> None:
        lines = [
            "/status  Runtime and repository summary",
            "/tools   Available tools",
            "/files   Files changed in this process",
            "/diff    Current Git working-tree diff",
            "/tokens  Provider token usage and estimated cost",
            "/compact Compress model context",
            "/model [name]  Show or switch model",
            "/goal <outcome>  Create a durable Goal setup",
            "/goal [start|status|pause|resume <id>|complete|clear|cancel]  Manage the active Goal; start requires acceptance checks",
            "/goal constraint <rule>  Add a confirmed hard constraint to the active Goal",
            "/goal constraint remove <id>  Retire one confirmed hard constraint",
            "/goal constraints  List confirmed hard constraints",
            "/goal criterion <file-exists|file-contains|test-command> ...  Add a Runtime completion check",
            "/goal criterion remove <id>  Retire one completion check",
            "/goal criteria  List Runtime completion checks",
            "/goal plan [show|load <repo-relative-json>|select <step-id>]  Record and inspect a bounded stage plan",
            "/goal evaluate  Assess the selected checkpoint using durable evidence; does not rerun tools",
            "/goal review <verified|blocked> <reason>  Resolve a stage awaiting human review",
            "/goal recover <tool-call-id> <completed|not-executed> <reason>  Record an operator-verified effect outcome",
            "/clear   Clear conversation messages",
        ]
        lines.extend([
            "/save    Show the active auto-saved event Session",
            "/sessions  List recoverable Sessions",
            "/session show [id]  Show Session facts and recovery warnings",
            "/resume <id>  Resume an event Session for this repository",
            "/details  List saved Tool Calls and choose one to expand",
            "/details [on|off]  Toggle full future Tool Result output",
            "/details <tool-call-id> [page]  Show one saved Tool Call detail",
            "/exit    Exit Chat",
        ])
        self.console.print(Panel("\n".join(lines), title="TechPilot Commands"))

    def _show_status(self) -> None:
        """Show the small set of live facts most useful during a Chat turn."""

        agent = self.runtime.agent
        llm = agent.llm
        prompt = getattr(llm, "total_prompt_tokens", 0)
        completion = getattr(llm, "total_completion_tokens", 0)
        cost = getattr(llm, "estimated_cost", None)
        if callable(cost):
            cost = cost()
        cost_text = f"${cost:.6f}" if isinstance(cost, (int, float)) else "unknown"
        current_tokens = estimate_tokens(agent.messages)
        max_context = getattr(agent.context, "max_tokens", None)
        context_text = str(current_tokens)
        if max_context:
            remaining = max(0, max_context - current_tokens)
            context_text = f"{current_tokens:,} used / {remaining:,} remaining (max {max_context:,})"

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Session ID", agent.session_id)
        table.add_row("Model", getattr(self.runtime.config, "model", getattr(llm, "model", "unknown")))
        table.add_row("Context", context_text)
        table.add_row("Usage", f"{prompt} prompt + {completion} completion = {prompt + completion} total")
        table.add_row("Estimated cost", cost_text)
        self.console.print(table)

    def _run_user_turn(self, user_input: str) -> str:
        """Run ordinary Chat, or the currently selected durable Goal."""

        if self._goal_workflow is None:
            return self.runtime.run_turn(user_input)
        result = self._goal_workflow.run_step(user_input)
        return result.response

    def _goal(self, value: str) -> None:
        """Expose an explicit, Codex-like Goal lifecycle inside ordinary Chat."""

        command, separator, argument = value.partition(" ")
        normalized = command.lower()
        argument = argument.strip()
        if not value:
            self._show_goal_status()
            return
        if normalized == "status" and not separator:
            self._show_goal_status()
            return
        if normalized == "pause" and not separator:
            self._pause_active_goal("Paused by user")
            return
        if normalized == "clear" and not separator:
            if not self._pause_active_goal("Goal cleared from this Chat"):
                return
            self._goal_workflow = None
            self.console.print(
                "[yellow]Goal 已从当前 Chat 解绑；恢复原 Session 后可用 /goal resume <id> 查看或继续配置。[/yellow]"
            )
            return
        if normalized == "cancel" and not separator:
            workflow = self._require_goal()
            if workflow is None:
                return
            try:
                task = workflow.store.cancel(workflow.task_id, reason="Cancelled by user")
            except (OSError, LongTaskStateError, ValueError) as error:
                self.console.print(f"[red]Could not cancel Goal: {error}[/red]")
                return
            self._goal_workflow = None
            self.console.print(f"[yellow]Goal cancelled: {task.task_id}[/yellow]")
            return
        if normalized == "complete" and not separator:
            workflow = self._require_goal()
            if workflow is None:
                return
            try:
                task = workflow.finish(result="Completed by user after verified completion claim")
            except (OSError, LongTaskStateError, ValueError) as error:
                self.console.print(f"[red]Could not complete Goal: {error}[/red]")
                return
            self._goal_workflow = None
            self.console.print(f"[green]Goal completed: {task.task_id}[/green]")
            return
        if normalized == "start" and not separator:
            self._start_goal_execution()
            return
        if normalized == "resume":
            self._resume_goal(argument)
            return
        if normalized == "constraint":
            self._record_goal_constraint(argument)
            return
        if normalized == "constraints" and not separator:
            self._show_goal_constraints()
            return
        if normalized == "criterion":
            self._record_goal_criterion(argument)
            return
        if normalized == "criteria" and not separator:
            self._show_goal_criteria()
            return
        if normalized == "plan":
            self._goal_plan(argument)
            return
        if normalized == "evaluate" and not separator:
            self._evaluate_goal_stage()
            return
        if normalized == "review":
            self._review_goal_stage(argument)
            return
        if normalized == "recover":
            self._recover_goal_effect(argument)
            return
        if self._goal_workflow is not None:
            task = self._goal_workflow.store.replay(self._goal_workflow.task_id)
            if task.status is LongTaskStatus.SETUP:
                self.console.print(
                    "[yellow]Goal is in setup. Add /goal criterion ... then run /goal start before sending work input.[/yellow]"
                )
                return
            self.console.print("[yellow]已有运行中的 Goal；请先用 /goal status、/goal complete、/goal pause 或 /goal clear。[/yellow]")
            return
        self._start_goal(value)

    def _start_goal(self, goal: str) -> None:
        task_id = f"goal-{uuid4().hex[:12]}"
        self._goal_workflow = LongTaskWorkflow(self.runtime, task_id=task_id, goal=goal)
        try:
            task = self._goal_workflow.create_for_setup()
        except (OSError, LongTaskStateError, ValueError) as error:
            self._goal_workflow = None
            self.console.print(f"[red]Could not create Goal setup: {error}[/red]")
            return
        self.console.print(f"[green]Goal setup created: {task.task_id}[/green]")
        self.console.print("[dim]Add /goal criterion ... and optional /goal constraint ..., then run /goal start.[/dim]")
        self._show_goal_status()

    def _start_goal_execution(self) -> None:
        workflow = self._require_goal()
        if workflow is None:
            return
        try:
            task = workflow.start()
            self.console.print(f"[green]Goal started: {task.task_id}[/green]")
            response = self._run_user_turn(task.goal)
        except KeyboardInterrupt:
            self.console.print("[yellow]Goal step cancelled. Use /goal status before continuing.[/yellow]")
            return
        except (OSError, LongTaskStateError, ValueError) as error:
            self.console.print(f"[red]Goal step failed: {error}[/red]")
            return
        finally:
            self._ensure_session_persisted()
        sink = self.runtime.agent.event_sink
        if not getattr(sink, "last_turn_streamed", False) and response:
            self.console.print(Markdown(response))

    def _resume_goal(self, task_id: str) -> None:
        if not task_id:
            self.console.print("[yellow]Usage: /goal resume <goal-id>[/yellow]")
            return
        if self._goal_workflow is not None:
            self.console.print("[yellow]当前 Chat 已绑定一个 Goal；请先 /goal clear 或完成它。[/yellow]")
            return
        store = LongTaskStore.for_repository(
            self.runtime.repository,
            session_directory=self.runtime.session_store.directory,
        )
        try:
            task = store.replay(task_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not resume Goal: {error}[/red]")
            return
        if task.session_id != self.runtime.agent.session_id:
            self.console.print(
                "[yellow]该 Goal 属于另一个 Session；请先 /resume <session-id>，再执行 /goal resume <goal-id>。[/yellow]"
            )
            return
        workflow = LongTaskWorkflow(
            self.runtime,
            task_id=task.task_id,
            goal=task.goal,
            budget=task.budget,
        )
        try:
            current = workflow.start_or_resume(reason="Resumed from Chat Goal command")
        except LongTaskStateError as error:
            self.console.print(f"[red]Could not resume Goal: {error}[/red]")
            return
        self._goal_workflow = workflow
        self.console.print(f"[green]Goal resumed: {current.task_id} ({current.status.value})[/green]")
        self._show_goal_status()

    def _show_goal_status(self) -> None:
        workflow = self._goal_workflow
        if workflow is None:
            self.console.print("[dim]No active Goal. Start one with /goal <outcome>.[/dim]")
            return
        try:
            task = workflow.store.replay(workflow.task_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not read Goal status: {error}[/red]")
            return
        shared_effect_issue = None
        if task.session_id and workflow.runtime.session_store is not None:
            try:
                journal = EffectJournalStore.for_session_directory(workflow.runtime.session_store.directory).replay(task.session_id)
                if journal.warnings:
                    shared_effect_issue = "Effect Journal has invalid or incomplete events"
                elif journal.ambiguous_entries:
                    shared_effect_issue = "Unresolved Tool effects: " + ", ".join(item.tool_call_id for item in journal.ambiguous_entries)
            except OSError as error:
                shared_effect_issue = f"Effect Journal could not be read: {error}"
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Goal ID", task.task_id)
        table.add_row("Status", task.status.value)
        table.add_row("Outcome", task.goal)
        table.add_row("Hard constraints", "; ".join(item.text for item in task.active_constraints) or "none")
        table.add_row("Completion criteria", str(len(task.active_completion_criteria)))
        claim = task.active_completion_claim
        table.add_row("Completion claim", f"ready ({claim.claim_id})" if claim else "not ready")
        verification = task.last_completion_verification
        if verification is not None:
            table.add_row("Last verification", "passed" if verification.passed else "failed")
        table.add_row("Pending actions", str(len(task.pending_action_ids)))
        table.add_row("Completed effects", str(len(task.completed_effect_ids)))
        table.add_row("Checkpoints", str(len(task.checkpoints)))
        if task.checkpoints:
            latest_checkpoint = task.checkpoints[-1]
            table.add_row("Latest checkpoint", latest_checkpoint.checkpoint_id)
            table.add_row("Session cursor", latest_checkpoint.session_event_cursor or "missing")
        if task.recovery_blockers:
            table.add_row("Recovery blockers", ", ".join(task.recovery_blockers))
        if task.ambiguous_effect_ids:
            table.add_row("Unknown effects", ", ".join(task.ambiguous_effect_ids))
        if shared_effect_issue:
            table.add_row("Shared effect blocker", shared_effect_issue)
        plan = task.plan
        if plan is None:
            table.add_row("Stage plan", "unplanned")
            next_action = (
                "Reconcile Goal recovery before another turn"
                if task.status is LongTaskStatus.RECOVERY_REQUIRED or task.recovery_blockers or task.ambiguous_effect_ids or shared_effect_issue
                else "Record a stage plan with /goal plan load <json-file>, or continue the legacy Goal manually"
            )
            table.add_row("Next action", next_action)
        else:
            table.add_row("Stage plan", f"v{plan.version} ({len(plan.steps)} stages)")
            table.add_row("Verified stages", ", ".join(task.verified_step_ids) or "none")
            selected = plan.step(task.selected_step_id) if task.selected_step_id else None
            table.add_row("Pending stages", str(len(plan.steps) - len(task.verified_step_ids) - (1 if selected else 0)))
            if selected is None:
                current = "none selected"
            elif task.selected_step_needs_evaluation:
                current = f"{selected.step_id} (needs_evaluation)"
            elif task.selected_step_assessment is not None:
                current = f"{selected.step_id} ({task.selected_step_assessment.status.value})"
            else:
                current = f"{selected.step_id} (active)"
            table.add_row("Current stage", current)
            table.add_row("Ready stages", ", ".join(task.ready_step_ids) or "none")
            blocked = [
                f"{step.step_id} ← {', '.join(dependency for dependency in step.dependencies if dependency not in task.verified_step_ids)}"
                for step in plan.steps if any(dependency not in task.verified_step_ids for dependency in step.dependencies)
            ]
            table.add_row("Blocked dependencies", "; ".join(blocked) or "none")
            latest_assessment = task.selected_step_assessment or (task.stage_assessments[-1] if task.stage_assessments else None)
            if latest_assessment is not None:
                label = "Stage assessment" if task.selected_step_assessment is latest_assessment else "Latest recorded assessment"
                table.add_row(label, f"{latest_assessment.step_id}: {latest_assessment.status.value} — {latest_assessment.detail}")
                table.add_row("Evidence events", ", ".join(latest_assessment.evidence_event_ids))
                table.add_row("Evidence Tool Calls", ", ".join(latest_assessment.tool_call_ids) or "none")
            if task.recovery_blockers or task.ambiguous_effect_ids or shared_effect_issue or task.status is LongTaskStatus.RECOVERY_REQUIRED:
                next_action = "Inspect recovery blockers and reconcile unknown effects before continuing"
            elif task.selected_step_needs_evaluation and selected is not None:
                next_action = "Run /goal evaluate to assess the saved checkpoint without rerunning the turn"
            elif task.selected_step_assessment is not None:
                next_action = task.selected_step_assessment.next_action
            elif selected is not None:
                next_action = selected.next_action
            elif task.ready_step_ids:
                next_action = f"Select a stage with /goal plan select {task.ready_step_ids[0]}"
            elif len(task.verified_step_ids) == len(plan.steps):
                next_action = "All stages verified; review final Goal criteria and completion claim"
            else:
                next_action = "No stage is ready; inspect the dependency contract"
            table.add_row("Next action", next_action)
        self.console.print(table)

    def _goal_plan(self, value: str) -> None:
        workflow = self._require_goal()
        if workflow is None:
            return
        command, separator, argument = value.partition(" ")
        normalized = command.lower()
        argument = argument.strip()
        if not value or (normalized == "show" and not separator):
            self._show_goal_plan()
            return
        if normalized == "select" and argument:
            try:
                task = workflow.store.select_goal_step(workflow.task_id, argument)
            except (OSError, LongTaskStateError, TypeError, ValueError) as error:
                self.console.print(f"[red]Could not select Goal stage: {error}[/red]")
                return
            self.console.print(f"[green]Selected Goal stage: {task.selected_step_id}[/green]")
            self._show_goal_status()
            return
        if normalized == "load" and argument:
            try:
                repository = self.runtime.repository.resolve()
                source = (repository / argument).resolve()
                if not source.is_relative_to(repository) or not source.is_file():
                    raise ValueError("Goal plan file must be a file inside the repository")
                if source.stat().st_size > 64_000:
                    raise ValueError("Goal plan file is too large")
                document = json.loads(source.read_text(encoding="utf-8"))
                task = workflow.store.record_goal_plan(workflow.task_id, document)
            except (OSError, LongTaskStateError, TypeError, ValueError) as error:
                self.console.print(f"[red]Could not load Goal plan: {error}[/red]")
                return
            self.console.print(f"[green]Recorded Goal stage plan v{task.plan.version}[/green]")
            self._show_goal_status()
            return
        self.console.print("[yellow]Usage: /goal plan [show|load <repo-relative-json>|select <step-id>][/yellow]")

    def _show_goal_plan(self) -> None:
        workflow = self._require_goal()
        if workflow is None:
            return
        try:
            task = workflow.store.replay(workflow.task_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not read Goal plan: {error}[/red]")
            return
        plan = task.plan
        if plan is None:
            self.console.print("[dim]Goal is unplanned. Use /goal plan load <repo-relative-json>.[/dim]")
            return
        self.console.print(f"[cyan]Goal stage plan v{plan.version}[/cyan] — {plan.reason}")
        if len(task.plans) > 1:
            self.console.print("History: " + "; ".join(f"v{item.version}: {item.reason}" for item in task.plans))
        for step in plan.steps:
            if step.step_id in task.verified_step_ids:
                state = "verified"
            elif step.step_id == task.selected_step_id:
                state = "needs_evaluation" if task.selected_step_needs_evaluation else "active"
                if task.selected_step_assessment is not None:
                    state = task.selected_step_assessment.status.value
            else:
                state = "blocked" if any(dependency not in task.verified_step_ids for dependency in step.dependencies) else "pending (awaiting evaluation)" if task.selected_step_needs_evaluation else "pending (ready)"
            self.console.print(f"[bold]{step.step_id}[/bold] ({state}): {step.goal}")
            self.console.print(f"  Depends on: {', '.join(step.dependencies) or 'none'}")
            self.console.print(f"  Output: {step.expected_output}")
            self.console.print(f"  Scope: {', '.join(step.work_scope)}")
            self.console.print(f"  Evidence: {step.evidence}; check: {step.check}")
            self.console.print(f"  Next: {step.next_action}")

    def _evaluate_goal_stage(self) -> None:
        workflow = self._require_goal()
        if workflow is None:
            return
        try:
            _task, assessment = workflow.evaluate_selected_step()
        except (OSError, LongTaskStateError, ValueError) as error:
            self.console.print(f"[red]Could not evaluate Goal stage: {error}[/red]")
            self._show_goal_status()
            return
        self.console.print(f"[green]Goal stage assessed: {assessment.step_id} → {assessment.status.value}[/green]")
        self._show_goal_status()

    def _review_goal_stage(self, value: str) -> None:
        status_text, separator, reason = value.partition(" ")
        if not separator or not reason.strip():
            self.console.print("[yellow]Usage: /goal review <verified|blocked> <reason>[/yellow]")
            return
        workflow = self._require_goal()
        if workflow is None:
            return
        try:
            status = GoalStageAssessmentStatus(status_text)
            workflow.review_selected_step(status, reason=reason)
        except (OSError, LongTaskStateError, ValueError) as error:
            self.console.print(f"[red]Could not review Goal stage: {error}[/red]")
            return
        self.console.print(f"[green]Goal stage review recorded: {status.value}[/green]")
        self._show_goal_status()

    def _record_goal_constraint(self, text: str) -> None:
        if not text:
            self.console.print("[yellow]Usage: /goal constraint <rule>[/yellow]")
            return
        workflow = self._require_goal()
        if workflow is None:
            return
        action, separator, argument = text.partition(" ")
        if action.lower() == "remove":
            if not separator or not argument.strip():
                self.console.print("[yellow]Usage: /goal constraint remove <id>[/yellow]")
                return
            try:
                workflow.store.remove_constraint(
                    workflow.task_id,
                    argument.strip(),
                    reason="Removed by user from Chat Goal command",
                )
            except (OSError, LongTaskStateError, ValueError) as error:
                self.console.print(f"[red]Could not remove Goal constraint: {error}[/red]")
                return
            self.console.print(f"[green]Goal constraint removed: {argument.strip()}[/green]")
            return
        try:
            task = workflow.record_constraint(text)
        except (OSError, LongTaskStateError, ValueError) as error:
            self.console.print(f"[red]Could not record Goal constraint: {error}[/red]")
            return
        constraint = next((item for item in task.active_constraints if item.text == text.strip()), None)
        if constraint is None:
            self.console.print("[yellow]Goal constraint was already recorded.[/yellow]")
            return
        self.console.print(f"[green]Goal constraint recorded: {constraint.constraint_id}[/green]")

    def _show_goal_constraints(self) -> None:
        workflow = self._require_goal()
        if workflow is None:
            return
        try:
            task = workflow.store.replay(workflow.task_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not read Goal constraints: {error}[/red]")
            return
        if not task.active_constraints:
            self.console.print("[dim]No confirmed hard constraints for this Goal.[/dim]")
            return
        table = Table(title="Goal hard constraints", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Source")
        table.add_column("Rule")
        for constraint in task.active_constraints:
            table.add_row(constraint.constraint_id, constraint.source, constraint.text)
        self.console.print(table)

    def _record_goal_criterion(self, text: str) -> None:
        workflow = self._require_goal()
        if workflow is None:
            return
        before_ids = {criterion.criterion_id for criterion in workflow.store.replay(workflow.task_id).active_completion_criteria}
        action, separator, remainder = text.partition(" ")
        normalized = action.lower()
        remainder = remainder.strip()
        if normalized == "remove":
            if not remainder:
                self.console.print("[yellow]Usage: /goal criterion remove <id>[/yellow]")
                return
            try:
                workflow.store.remove_completion_criterion(
                    workflow.task_id,
                    remainder,
                    reason="Removed by user from Chat Goal command",
                )
            except (OSError, LongTaskStateError, ValueError) as error:
                self.console.print(f"[red]Could not remove completion criterion: {error}[/red]")
                return
            self.console.print(f"[green]Goal completion criterion removed: {remainder}[/green]")
            return
        try:
            if normalized == "file-exists" and remainder:
                task = workflow.record_completion_criterion(CompletionCriterionKind.FILE_EXISTS, path=remainder)
            elif normalized == "file-contains":
                path, separator, expected = remainder.partition(" ")
                if not path or not separator or not expected.strip():
                    raise ValueError("file-contains requires a path and expected text")
                task = workflow.record_completion_criterion(
                    CompletionCriterionKind.FILE_CONTAINS,
                    path=path,
                    expected_text=expected.strip(),
                )
            elif normalized == "test-command" and remainder:
                task = workflow.record_completion_criterion(CompletionCriterionKind.TEST_COMMAND, command=remainder)
            else:
                self.console.print(
                    "[yellow]Usage: /goal criterion file-exists <path> | "
                    "file-contains <path> <text> | test-command <command>[/yellow]"
                )
                return
        except (OSError, LongTaskStateError, ValueError) as error:
            self.console.print(f"[red]Could not record completion criterion: {error}[/red]")
            return
        criterion = next(
            (item for item in task.active_completion_criteria if item.criterion_id not in before_ids),
            None,
        )
        if criterion is None:
            self.console.print("[yellow]Goal completion criterion was already recorded.[/yellow]")
            return
        self.console.print(f"[green]Goal completion criterion recorded: {criterion.criterion_id}[/green]")

    def _show_goal_criteria(self) -> None:
        workflow = self._require_goal()
        if workflow is None:
            return
        try:
            task = workflow.store.replay(workflow.task_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not read Goal completion criteria: {error}[/red]")
            return
        if not task.active_completion_criteria:
            self.console.print("[dim]No Runtime completion criteria. Add one before /goal complete.[/dim]")
            return
        table = Table(title="Goal completion criteria", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Kind")
        table.add_column("Check")
        for criterion in task.active_completion_criteria:
            if criterion.kind is CompletionCriterionKind.FILE_EXISTS:
                check = criterion.path or ""
            elif criterion.kind is CompletionCriterionKind.FILE_CONTAINS:
                check = f"{criterion.path}: contains {criterion.expected_text!r}"
            else:
                check = f"exact command exits 0: {criterion.command}"
            table.add_row(criterion.criterion_id, criterion.kind.value, check)
        self.console.print(table)

    def _recover_goal_effect(self, value: str) -> None:
        """Write a user-confirmed effect fact, then reopen the blocked Goal."""

        workflow = self._require_goal()
        if workflow is None:
            return
        parts = value.split(maxsplit=2)
        if len(parts) != 3:
            self.console.print(
                "[yellow]Usage: /goal recover <tool-call-id> <completed|not-executed> <reason>[/yellow]"
            )
            return
        tool_call_id, disposition, reason = parts
        normalized = disposition.lower()
        if normalized == "completed":
            state = EffectJournalState.FINISHED
            outcome = ToolResult(f"[reconciled completed] {reason}", ToolStatus.COMPLETED)
        elif normalized == "not-executed":
            state = EffectJournalState.NOT_EXECUTED
            outcome = ToolResult(f"[reconciled not executed] {reason}", ToolStatus.NOT_EXECUTED)
        else:
            self.console.print(
                "[yellow]Effect outcome must be completed or not-executed; it is never inferred from a model reply.[/yellow]"
            )
            return
        journal = EffectJournalStore.for_session_directory(self.runtime.session_store.directory)
        try:
            entry = journal.reconcile(
                self.runtime.agent.session_id,
                tool_call_id,
                state=state,
                result=outcome,
                reason=reason,
            )
            task = workflow.store.replay(workflow.task_id)
            if task.status is LongTaskStatus.RECOVERY_REQUIRED:
                task = workflow.store.resume(
                    workflow.task_id,
                    reason="Shared Effect Journal reconciled by user from Chat Goal command",
                )
        except (OSError, LongTaskStateError, ValueError) as error:
            self.console.print(f"[red]Could not reconcile Goal effect: {error}[/red]")
            return
        self.console.print(
            f"[green]Goal effect reconciled: {entry.tool_call_id} ({entry.state.value}); Goal {task.status.value}.[/green]"
        )

    def _require_goal(self) -> LongTaskWorkflow | None:
        if self._goal_workflow is None:
            self.console.print("[yellow]No active Goal. Start one with /goal <outcome>.[/yellow]")
        return self._goal_workflow

    def _pause_active_goal(self, reason: str) -> bool:
        workflow = self._goal_workflow
        if workflow is None:
            return False
        try:
            task = workflow.store.replay(workflow.task_id)
            if task.status is LongTaskStatus.RUNNING:
                task = workflow.store.pause(workflow.task_id, reason=reason)
            label = "Goal setup saved" if task.status is LongTaskStatus.SETUP else "Goal paused"
            self.console.print(f"[yellow]{label}: {task.task_id}[/yellow]")
            return True
        except (OSError, LongTaskStateError, ValueError) as error:
            self.console.print(f"[red]Could not pause Goal: {error}[/red]")
            return False

    def _show_tools(self) -> None:
        for tool in self.runtime.tools:
            self.console.print(f"[cyan]{tool.name}[/cyan]  {tool.description}")

    def _show_files(self) -> None:
        files = self._repository_changed_files()
        if not files:
            self.console.print("[dim]No files changed in this process.[/dim]")
            return
        for file_path in files:
            self.console.print(file_path)

    def _show_diff(self) -> None:
        try:
            root_result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.runtime.repository,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self.console.print(f"[red]Could not read Git diff: {error}[/red]")
            return
        repository = self.runtime.repository.resolve()
        git_root = Path(root_result.stdout.strip()).resolve() if root_result.returncode == 0 else None
        if git_root != repository:
            self.console.print(
                "[yellow]当前目录不是独立 Git 仓库，已避免展示父级仓库的变更。"
                "可使用 /files 查看本次会话修改的文件。[/yellow]"
            )
            return
        try:
            result = subprocess.run(
                ["git", "diff", "--no-ext-diff"],
                cwd=repository,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self.console.print(f"[red]Could not read Git diff: {error}[/red]")
            return
        output = result.stdout or result.stderr
        if output.strip():
            self.console.print(output, markup=False)
        else:
            self.console.print("[dim]No Git diff.[/dim]")

    def _show_tokens(self) -> None:
        llm = self.runtime.agent.llm
        prompt = getattr(llm, "total_prompt_tokens", 0)
        completion = getattr(llm, "total_completion_tokens", 0)
        line = f"Tokens: {prompt} prompt + {completion} completion = {prompt + completion} total"
        cost = getattr(llm, "estimated_cost", None)
        if cost is not None:
            line += f" (~${cost:.4f})"
        self.console.print(line)

    def _compact(self) -> None:
        compact = getattr(self.runtime.agent, "compact_context", None)
        if compact is not None:
            compressed, before, after = compact()
        else:
            before = estimate_tokens(self.runtime.agent.messages)
            compressed = self.runtime.agent.context.maybe_compress(
                self.runtime.agent.messages,
                self.runtime.agent.llm,
            )
            after = estimate_tokens(self.runtime.agent.messages)
        message = f"Compressed: {before} → {after} tokens" if compressed else f"Nothing to compress ({before} tokens)"
        self.console.print(message)

    def _show_save(self) -> None:
        store = getattr(self.runtime, "session_store", None)
        if store is None:
            self.console.print("[yellow]Event Session storage is unavailable for this runtime.[/yellow]")
            return
        self._ensure_session_persisted()
        self.console.print(
            f"[green]自动保存已开启。[/green] Session: {self.runtime.agent.session_id}\n"
            f"Events: {store.directory / (self.runtime.agent.session_id + '.jsonl')}"
        )

    def _show_sessions(self) -> None:
        store = getattr(self.runtime, "session_store", None)
        if store is None:
            self.console.print("[yellow]Event Session storage is unavailable for this runtime.[/yellow]")
            return
        items = store.list()
        if not items:
            self.console.print("[dim]No saved Sessions.[/dim]")
            return
        table = Table(title="TechPilot Sessions")
        table.add_column("Session")
        table.add_column("Status")
        table.add_column("Model")
        table.add_column("Events", justify="right")
        table.add_column("Warnings", justify="right")
        for item in items:
            table.add_row(
                item.session_id,
                item.status,
                item.model or "?",
                str(len(item.events)),
                str(len(item.warnings)),
            )
        self.console.print(table)

    def _session_command(self, argument: str) -> None:
        values = argument.split()
        if not values or values[0].lower() != "show":
            self.console.print("[yellow]Usage: /session show [session-id][/yellow]")
            return
        store = getattr(self.runtime, "session_store", None)
        if store is None:
            self.console.print("[yellow]Event Session storage is unavailable for this runtime.[/yellow]")
            return
        session_id = values[1] if len(values) > 1 else self.runtime.agent.session_id
        try:
            item = store.replay(session_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not load Session: {error}[/red]")
            return
        warnings = "\n".join(f"- {warning}" for warning in item.warnings) or "- none"
        if item.last_result is not None:
            result_line = _session_result_line(item)
        else:
            result_line = "Last result: -"
        self.console.print(Panel(
            "\n".join([
                f"Session: {item.session_id}",
                f"Repository: {item.repository_root or '?'}",
                f"Model: {item.model or '?'}",
                f"Status: {item.status}",
                result_line,
                f"Events: {len(item.events)}",
                f"Messages: {len(item.messages)} (model projection: {len(item.model_messages)})",
                f"Usage: {item.prompt_tokens} prompt + {item.completion_tokens} completion",
                "Recovery warnings:",
                warnings,
            ]),
            title="Session details",
        ))

    def _resume(self, session_id: str) -> None:
        if not session_id:
            self.console.print("[yellow]Usage: /resume <session-id>[/yellow]")
            return
        resume = getattr(self.runtime, "resume_session", None)
        if resume is None:
            self.console.print("[yellow]Event Session storage is unavailable for this runtime.[/yellow]")
            return
        try:
            item = resume(session_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not resume Session: {error}[/red]")
            return
        warning = f" Warnings: {len(item.warnings)}." if item.warnings else ""
        self.console.print(
            f"[green]Resumed Session {item.session_id} ({len(item.model_messages)} messages).[/green]{warning}"
        )
        if self._goal_workflow is not None:
            self._goal_workflow = None
            self.console.print("[yellow]已解绑旧 Session 的 Goal；如需继续，请使用 /goal resume <goal-id>。[/yellow]")
        self._show_recovery_notices()

    def _details(self, value: str) -> None:
        sink = self.runtime.agent.event_sink
        terminal_sink = getattr(sink, "downstream", sink)
        if not isinstance(terminal_sink, TerminalEventSink):
            self.console.print("[yellow]Tool detail mode is unavailable for this output target.[/yellow]")
            return
        values = value.split()
        normalized = value.lower()
        if normalized in {"on", "off"}:
            if normalized:
                terminal_sink.show_full_results = normalized == "on"
            state = "on" if terminal_sink.show_full_results else "off"
            self.console.print(f"Tool detail mode: [cyan]{state}[/cyan]")
            return
        if not normalized:
            self._show_tool_call_list()
            return
        if len(values) not in {1, 2}:
            self.console.print("[yellow]Usage: /details [on|off|<tool-call-id> [page]][/yellow]")
            return
        page = 1
        if len(values) == 2:
            try:
                page = int(values[1])
            except ValueError:
                self.console.print("[yellow]Page must be a positive integer.[/yellow]")
                return
        if page < 1:
            self.console.print("[yellow]Page must be a positive integer.[/yellow]")
            return
        store = getattr(self.runtime, "session_store", None)
        if store is None:
            self.console.print("[yellow]Saved Tool Call details are unavailable for this runtime.[/yellow]")
            return
        try:
            projection = store.replay(self.runtime.agent.session_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not load Tool Call details: {error}[/red]")
            return
        detail = _find_tool_call_detail(_tool_call_details(projection.events), values[0])
        if detail is None:
            self.console.print(
                f"[yellow]No saved Tool Call matches {values[0]!r}. Use /details to choose a saved Tool Call.[/yellow]"
            )
            return
        self._show_tool_call_detail(detail, page)

    def _show_tool_call_list(self) -> None:
        store = getattr(self.runtime, "session_store", None)
        if store is None:
            self.console.print("[yellow]Saved Tool Call details are unavailable for this runtime.[/yellow]")
            return
        try:
            projection = store.replay(self.runtime.agent.session_id)
        except (OSError, ValueError) as error:
            self.console.print(f"[red]Could not load Tool Call details: {error}[/red]")
            return
        details = _tool_call_details(projection.events)
        if not details:
            self.console.print("[dim]This Session has no saved Tool Calls.[/dim]")
            return
        table = Table(title="折叠的工具调用（/details <id> 展开）")
        table.add_column("ID", style="cyan")
        table.add_column("Tool")
        table.add_column("Status")
        table.add_column("Duration", justify="right")
        for detail in details:
            duration = f"{detail.duration_seconds:.2f}s" if detail.duration_seconds is not None else "-"
            table.add_row(detail.call_id, detail.tool_name, detail.status, duration)
        self.console.print(table)

    def _show_tool_call_detail(self, detail: ToolCallDetail, page: int) -> None:
        body = _format_tool_call_detail(
            detail,
            repository=self.runtime.repository,
            validation_commands=(self.runtime.profile.validation_commands if self.runtime.profile else []),
        )
        page_count = max(1, (len(body) + _DETAIL_PAGE_SIZE - 1) // _DETAIL_PAGE_SIZE)
        if page > page_count:
            self.console.print(f"[yellow]Page {page} is unavailable; this detail has {page_count} page(s).[/yellow]")
            return
        start = (page - 1) * _DETAIL_PAGE_SIZE
        end = start + _DETAIL_PAGE_SIZE
        footer = ""
        if page_count > 1:
            footer = (
                f"\n\n[输出已分页：第 {page}/{page_count} 页；"
                f"完整 Tool Result 已保留在 Session。使用 /details {detail.call_id} <page> 查看其他页。]"
            )
        duration = f"{detail.duration_seconds:.2f}s" if detail.duration_seconds is not None else "unknown"
        self.console.print(Panel(
            Text(body[start:end] + footer),
            title=f"Tool details · {detail.tool_name} · {detail.status} · {duration}",
            border_style=_detail_border_style(detail.status),
        ))

    def _ensure_session_persisted(self) -> None:
        ensure = getattr(self.runtime, "ensure_persisted", None)
        if ensure is None:
            return
        try:
            ensure()
        except OSError as error:
            self.console.print(f"[red]Session autosave error: {error}[/red]")

    def _model(self, name: str) -> None:
        if name:
            change_model = getattr(self.runtime, "set_model", None)
            if change_model is not None:
                change_model(name)
            else:
                self.runtime.agent.llm.model = name
                self.runtime.config.model = name
            self.console.print(f"Model switched to [cyan]{name}[/cyan]")
        else:
            self.console.print(f"Current model: [cyan]{self.runtime.config.model}[/cyan]")

    def _clear(self) -> None:
        if self._goal_workflow is not None:
            self.console.print("[yellow]当前 Goal 不能直接 /clear；请用 /goal clear 暂停并解绑，避免破坏可恢复上下文。[/yellow]")
            return
        self.runtime.agent.reset()
        self.console.print("[yellow]Conversation messages cleared.[/yellow]")

    def _show_recovery_notices(self) -> None:
        consume = getattr(self.runtime, "consume_recovery_notices", None)
        notices = consume() if callable(consume) else []
        for notice in notices:
            self.console.print(Panel(notice, title="旧版隔离请求已冻结", border_style="yellow"))

    def _repository_changed_files(self) -> list[str]:
        files = []
        for value in _changed_files:
            path = Path(value).resolve()
            try:
                files.append(path.relative_to(self.runtime.repository).as_posix())
            except ValueError:
                continue
        return sorted(files)


def _brief_arguments(arguments: object, max_length: int = 120) -> str:
    if not isinstance(arguments, dict):
        return _summarize(str(arguments), max_length)
    value = ", ".join(f"{key}={item!r}" for key, item in arguments.items())
    return _summarize(value, max_length)


def _tool_call_details(events: list[object]) -> list[ToolCallDetail]:
    """Rebuild display-only Tool Call details from persisted Runtime Events."""

    details: dict[str, ToolCallDetail] = {}
    for event in events:
        event_type = getattr(event, "event_type", "")
        event_type = getattr(event_type, "value", event_type)
        call_id = getattr(event, "tool_call_id", None)
        payload = getattr(event, "payload", {})
        if not call_id or not isinstance(payload, dict):
            continue
        detail = details.get(call_id)
        if event_type == RuntimeEventType.TOOL_REQUESTED.value:
            detail = detail or ToolCallDetail(call_id=call_id)
            detail.tool_name = str(payload.get("tool_name", detail.tool_name))
            detail.arguments = payload.get("arguments", {})
            detail.requested_at = getattr(event, "created_at", None)
            details[call_id] = detail
        elif event_type == RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value:
            detail = detail or ToolCallDetail(call_id=call_id)
            detail.tool_name = str(payload.get("tool_name", detail.tool_name))
            detail.execution_control = dict(payload)
            details[call_id] = detail
        elif event_type == RuntimeEventType.TOOL_COMPLETED.value:
            detail = detail or ToolCallDetail(call_id=call_id)
            detail.tool_name = str(payload.get("tool_name", detail.tool_name))
            detail.result = str(payload.get("result", ""))
            detail.tool_status = payload.get("tool_status")
            raw_facts = payload.get("result_facts")
            detail.result_facts = dict(raw_facts) if isinstance(raw_facts, dict) else None
            detail.interrupted = bool(payload.get("interrupted"))
            detail.completed_at = getattr(event, "created_at", None)
            details[call_id] = detail
    return list(details.values())


def _find_tool_call_detail(details: list[ToolCallDetail], value: str) -> ToolCallDetail | None:
    exact = [detail for detail in details if detail.call_id == value]
    if exact:
        return exact[-1]
    matches = [detail for detail in details if detail.call_id.startswith(value)]
    return matches[-1] if len(matches) == 1 else None


def _format_tool_call_detail(
    detail: ToolCallDetail,
    *,
    repository: Path,
    validation_commands: list[list[str]],
) -> str:
    """Format stored facts without rerunning, re-diffing, or mutating a Tool Call."""

    result = detail.result if detail.result is not None else "(Tool Call is still running; no saved result yet.)"
    duration = f"{detail.duration_seconds:.2f}s" if detail.duration_seconds is not None else "unknown"
    lines = [
        f"Tool Call ID: {detail.call_id}",
        f"Tool: {detail.tool_name}",
        f"Status: {detail.status}",
        f"Duration: {duration}",
    ]
    arguments = detail.arguments if isinstance(detail.arguments, dict) else {}
    if detail.tool_name == "bash":
        command = str(arguments.get("command", ""))
        stdout, stderr, exit_code = _shell_result_parts(result)
        facts = detail.result_facts or {}
        if isinstance(facts.get("stdout"), str):
            stdout = facts["stdout"]
        if isinstance(facts.get("stderr"), str):
            stderr = facts["stderr"]
        stored_exit_code = facts.get("exit_code")
        if isinstance(stored_exit_code, int) and not isinstance(stored_exit_code, bool):
            exit_code = stored_exit_code
        stored_cwd = facts.get("cwd")
        cwd = stored_cwd if isinstance(stored_cwd, str) else str(repository)
        lines.extend([
            "",
            "Shell:",
            f"Command: {command or '(missing from saved arguments)'}",
            f"cwd: {cwd}",
            f"Exit code: {exit_code if exit_code is not None else ('cancelled' if detail.interrupted else '0 or unavailable')}",
            "stdout:",
            stdout if not facts.get("truncated") else "(not retained here; output was truncated)",
            "stderr:",
            (stderr or "(none)") if not facts.get("truncated") else "(not retained here; output was truncated)",
        ])
        if facts.get("truncated"):
            lines.append(f"Output: truncated after {facts.get('output_chars', 'unknown')} stream chars")
            artifact = facts.get("output_artifact")
            lines.append(f"Full output artifact: {artifact if isinstance(artifact, str) else '(not persisted)'}")
        if facts.get("timed_out"):
            lines.append(f"Timeout: {facts.get('timeout_seconds', 'unknown')}s")
        if _is_validation_command(command, validation_commands):
            validation_status = (
                "passed" if detail.status == "completed"
                else "unknown" if detail.status == "effect unknown" else "failed"
            )
            lines.extend(["", f"Validation: {validation_status}"])
    elif detail.tool_name in {"edit_file", "write_file"}:
        summary, trusted_diff = _split_result_summary(result)
        lines.extend([
            "",
            "File operation:",
            f"Path: {arguments.get('file_path', '(missing from saved arguments)')}",
            f"Change summary: {summary}",
            "Permission / Trusted Diff:",
            _permission_detail(detail),
            "Diff:",
            trusted_diff or "(No trusted diff was returned; the operation may have been denied, failed, or made no change.)",
        ])
    else:
        lines.extend([
            "",
            "Arguments:",
            _format_arguments(arguments),
            "Result:",
            result,
        ])

    if detail.execution_control is not None:
        lines.extend(["", "Execution Control:", _format_execution_control(detail.execution_control)])
    elif detail.status in {"denied", "interrupted", "error", "not executed", "effect unknown"}:
        lines.extend(["", "Result detail:", result])
    return "\n".join(lines)


def _format_execution_control(payload: dict[str, object]) -> str:
    required = payload.get("required_control", "unknown")
    summary = payload.get("normalized_summary")
    lines = [f"Control: {required}"]
    if summary is not None:
        lines.extend(["Normalized request:", _format_arguments(summary)])
    reasons = payload.get("reasons", [])
    if isinstance(reasons, list) and reasons:
        lines.append("Reasons and evidence:")
        for reason in reasons:
            if not isinstance(reason, dict):
                continue
            evidence = reason.get("evidence", [])
            evidence_text = "；".join(str(item) for item in evidence) if isinstance(evidence, list) else str(evidence)
            lines.append(f"- {reason.get('message', 'control rule')}：{evidence_text or '(none)'}")
    return "\n".join(lines)


def _permission_detail(detail: ToolCallDetail) -> str:
    result = detail.result or ""
    if detail.status == "denied":
        return f"Denied: {result}"
    if detail.status in {"error", "interrupted", "not executed", "effect unknown"}:
        return f"Not completed: {detail.status}. {result}"
    return "Completed. The saved Tool Result below contains the Trusted Diff reviewed for this write."


def _format_arguments(arguments: object) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return repr(arguments)


def _shell_result_parts(result: str) -> tuple[str, str, int | None]:
    exit_match = re.search(r"(?:\n|^)\[exit code: (-?\d+)\]\s*$", result)
    exit_code = int(exit_match.group(1)) if exit_match else None
    output = result[:exit_match.start()] if exit_match else result
    stdout, separator, stderr = output.partition("\n[stderr]\n")
    return stdout or "(no output)", stderr if separator else "", exit_code


def _split_result_summary(result: str) -> tuple[str, str]:
    summary, separator, remainder = result.partition("\n")
    return summary or "(no summary)", remainder if separator else ""


def _is_validation_command(command: str, validation_commands: list[list[str]]) -> bool:
    return command in {" ".join(parts) for parts in validation_commands}


def _as_datetime(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _detail_border_style(status: str) -> str:
    if status in {"error", "denied"}:
        return "red"
    if status in {"interrupted", "not executed", "running", "effect unknown"}:
        return "yellow"
    return "green"


def _summarize(value: str, limit: int) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head - 24)
    return f"{normalized[:head]}\n... output folded ...\n{normalized[-tail:]}"


def _tool_status(result: str, status: object = None) -> str:
    return display_tool_status(result, status)


def _session_result_line(session) -> str:
    """Make turn success explicit when a tool call was blocked inside it."""

    result = session.last_result
    if result is None:
        return "Last result: -"
    latest_turn_id = next(
        (
            event.turn_id
            for event in reversed(session.events)
            if event.event_type in {
                RuntimeEventType.TURN_COMPLETED.value,
                RuntimeEventType.TURN_STARTED.value,
            }
        ),
        None,
    )
    blocked = latest_turn_id is not None and any(
        event.turn_id == latest_turn_id
        and event.event_type == RuntimeEventType.EXECUTION_CONTROL_ASSESSED.value
        and event.payload.get("required_control") == "block"
        for event in session.events
    )
    if blocked:
        return (
            f"Last result: {result.status.value} ("
            "Tool Call blocked, not executed)"
        )
    return f"Last result: {result.status.value}"
