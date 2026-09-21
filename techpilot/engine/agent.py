"""Core agent loop.

This is the heart of TechPilot's runtime. The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import copy
import inspect
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from .context import ContextManager, estimate_tokens
from .events import EventSink, NullEventSink, RuntimeEvent, RuntimeEventType
from .llm import LLM
from .prompt import system_prompt
from .runtime_control import CancellationToken, RuntimeCancelled, RuntimeLimitExceeded, RuntimeLimits
from .tool_execution import (
    PreparedToolCall,
    ToolExecutionDescription,
    ToolExecutionPlan,
    declared_tool_description,
    default_tool_description,
)
from .tool_results import ToolResult, ToolStatus, tool_result
from .tool_validation import validate_tool_arguments
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool


class ToolExecutor(Protocol):
    """Optional interception point for applications that need controlled Tool execution."""

    def execute(self, tool: Tool, arguments: dict[str, Any]) -> str:
        """Execute one validated Tool request and return its text result."""


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Correlation facts an application executor may use before a Tool effect."""

    session_id: str
    turn_id: str
    round_index: int
    event_sink: EventSink

    def emit(self, event_type: RuntimeEventType, *, tool_call_id: str, payload: dict[str, Any]) -> None:
        """Emit an executor fact without allowing an observer failure to alter execution."""

        try:
            self.event_sink.emit(RuntimeEvent(
                event_type=event_type,
                session_id=self.session_id,
                turn_id=self.turn_id,
                round_index=self.round_index,
                tool_call_id=tool_call_id,
                payload=payload,
            ))
        except Exception:
            pass


class _BufferedEventSink:
    """Collect executor events until Agent can publish them in Tool Call order."""

    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []
        self._lock = threading.Lock()

    def emit(self, event: RuntimeEvent) -> None:
        with self._lock:
            self._events.append(event)

    def flush_to(self, sink: EventSink) -> None:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        for event in events:
            try:
                sink.emit(event)
            except Exception:
                pass


@dataclass(frozen=True, slots=True)
class _PlannedToolResults:
    results: list[str]
    stop_message: str | None = None
    cancelled: bool = False


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        tool_executor: ToolExecutor | None = None,
        event_sink: EventSink | None = None,
        session_id: str | None = None,
        working_directory: str | None = None,
        assistant_name: str = "TechPilot",
        system_context: str | None = None,
        limits: RuntimeLimits | None = None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self.tool_executor = tool_executor
        self.event_sink = event_sink if event_sink is not None else NullEventSink()
        self.session_id = session_id or uuid4().hex
        self.limits = limits or RuntimeLimits()
        self._working_directory = working_directory
        self._assistant_name = assistant_name
        self._system_context = system_context
        self._active_cancellation_token: CancellationToken | None = None
        self._system = self._build_system_prompt()

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _build_system_prompt(self) -> str:
        return system_prompt(
            self.tools,
            working_directory=self._working_directory,
            assistant_name=self._assistant_name,
            additional_context=self._system_context,
        )

    def update_system_context(self, system_context: str | None) -> None:
        """Refresh runtime facts without disturbing the conversation history."""

        self._system_context = system_context
        self._system = self._build_system_prompt()

    def update_tools(self, tools: list[Tool]) -> None:
        """Replace the model-visible tool set without losing conversation state."""

        self.tools = list(tools)
        self._tool_by_name = {tool.name: tool for tool in self.tools}
        for tool in self.tools:
            if isinstance(tool, AgentTool):
                tool._parent_agent = self
        self._system = self._build_system_prompt()

    def cancel_current_turn(self, reason: str = "cancelled by user") -> bool:
        """Request cooperative cancellation at the next provider/tool boundary."""

        if self._active_cancellation_token is None:
            return False
        self._active_cancellation_token.cancel(reason)
        return True

    def compact_context(self) -> tuple[bool, int, int]:
        """Create a durable projection checkpoint without deleting Session facts."""

        before = estimate_tokens(self.messages)
        compressed = self.context.maybe_compress(self.messages, self.llm)
        after = estimate_tokens(self.messages)
        if compressed:
            self._emit_context_compressed(uuid4().hex, 0, phase="manual")
        return compressed, before, after

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(
        self,
        user_input: str,
        on_token=None,
        on_tool=None,
        cancellation_token: CancellationToken | None = None,
        *,
        allow_tools: bool = True,
    ) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        turn_id = uuid4().hex
        round_index = 0
        pending_tool_calls = []
        token = cancellation_token or CancellationToken()
        started_at = time.monotonic()
        provider_calls = 0
        tool_rounds = 0
        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        cost_before = self._estimated_cost()
        self._active_cancellation_token = token
        self._begin_tool_turn()
        self._emit_event(
            RuntimeEventType.TURN_STARTED,
            turn_id,
            round_index,
            payload={"user_input": user_input},
        )
        self.messages.append({"role": "user", "content": user_input})
        try:
            self._check_runtime_controls(
                token,
                started_at,
                provider_calls,
                tool_rounds,
                turn_prompt_tokens + turn_completion_tokens,
                cost_before,
            )
            if self.context.maybe_compress(self.messages, self.llm):
                self._emit_context_compressed(turn_id, round_index, phase="before_provider")

            for round_index in range(1, self.max_rounds + 1):
                self._check_runtime_controls(
                    token,
                    started_at,
                    provider_calls,
                    tool_rounds,
                    turn_prompt_tokens + turn_completion_tokens,
                    cost_before,
                )
                self._emit_event(
                    RuntimeEventType.PROVIDER_STARTED,
                    turn_id,
                    round_index,
                    payload={"message_count": len(self.messages)},
                )
                provider_calls += 1

                def handle_token(token: str, event_round: int = round_index) -> None:
                    self._emit_event(
                        RuntimeEventType.ASSISTANT_TOKEN,
                        turn_id,
                        event_round,
                        payload={"token": token},
                    )
                    if on_token:
                        on_token(token)

                resp = self.llm.chat(
                    messages=self._full_messages(),
                    tools=self._tool_schemas() if allow_tools else [],
                    on_token=handle_token,
                )
                turn_prompt_tokens += resp.prompt_tokens
                turn_completion_tokens += resp.completion_tokens
                # no tool calls -> LLM is done, return text
                if not resp.tool_calls:
                    self._check_runtime_controls(
                        token,
                        started_at,
                        provider_calls,
                        tool_rounds,
                        turn_prompt_tokens + turn_completion_tokens,
                        cost_before,
                        check_provider_limit=False,
                    )
                    self.messages.append(resp.message)
                    self._emit_event(
                        RuntimeEventType.TURN_COMPLETED,
                        turn_id,
                        round_index,
                        payload={
                            "content": resp.content,
                            "prompt_tokens": resp.prompt_tokens,
                            "completion_tokens": resp.completion_tokens,
                        },
                    )
                    return resp.content

                # tool calls -> execute (parallel when multiple, like Claude Code's
                # StreamingToolExecutor which runs independent tools concurrently)
                self.messages.append(resp.message)
                pending_tool_calls = list(resp.tool_calls)

                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    self._emit_tool_requested(
                        tc,
                        turn_id,
                        round_index,
                        on_tool,
                        assistant_content=resp.content,
                    )
                    self._check_runtime_controls(
                        token,
                        started_at,
                        provider_calls,
                        tool_rounds,
                        turn_prompt_tokens + turn_completion_tokens,
                        cost_before,
                        check_provider_limit=False,
                    )
                    self._check_tool_round_limit(tool_rounds)
                    tool_rounds += 1
                    token.raise_if_cancelled()
                    result = self._exec_tool_with_event(tc, turn_id, round_index)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    stop_message = self._consume_tool_turn_stop()
                    if stop_message is not None:
                        return self._finish_stopped_turn(turn_id, round_index, stop_message)
                else:
                    for tc in resp.tool_calls:
                        self._emit_tool_requested(
                            tc,
                            turn_id,
                            round_index,
                            on_tool,
                            assistant_content=resp.content,
                        )
                    self._check_runtime_controls(
                        token,
                        started_at,
                        provider_calls,
                        tool_rounds,
                        turn_prompt_tokens + turn_completion_tokens,
                        cost_before,
                        check_provider_limit=False,
                    )
                    self._check_tool_round_limit(tool_rounds)
                    tool_rounds += 1
                    planned = self._exec_tools_in_plan(
                        resp.tool_calls,
                        turn_id=turn_id,
                        round_index=round_index,
                        cancellation_token=token,
                    )
                    for tc, result in zip(resp.tool_calls, planned.results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                    if planned.cancelled:
                        token.raise_if_cancelled()
                    stop_message = planned.stop_message
                    if stop_message is not None:
                        return self._finish_stopped_turn(turn_id, round_index, stop_message)
                pending_tool_calls = []

                # compress if tool outputs are big
                if self.context.maybe_compress(self.messages, self.llm):
                    self._emit_context_compressed(turn_id, round_index, phase="after_tools")

            result = "(reached maximum tool-call rounds)"
            self._emit_event(
                RuntimeEventType.TURN_LIMIT_REACHED,
                turn_id,
                self.max_rounds,
                payload={"max_rounds": self.max_rounds, "result": result},
            )
            return result
        except RuntimeCancelled as error:
            backfilled = self._answer_pending_tool_calls(pending_tool_calls)
            self._emit_backfilled_tools(backfilled, turn_id, round_index, "[cancelled]", interrupted=True)
            result = f"(turn cancelled: {error})"
            self._emit_event(
                RuntimeEventType.TURN_INTERRUPTED,
                turn_id,
                round_index,
                payload={
                    "reason": str(error),
                    "pending_tool_call_ids": [tc.id for tc in backfilled],
                },
            )
            return result
        except RuntimeLimitExceeded as error:
            backfilled = self._answer_pending_tool_calls(pending_tool_calls)
            self._emit_backfilled_tools(backfilled, turn_id, round_index, "[limit reached]", interrupted=False)
            result = f"(runtime limit reached: {error.limit})"
            self._emit_event(
                RuntimeEventType.TURN_LIMIT_REACHED,
                turn_id,
                round_index,
                payload={
                    "limit": error.limit,
                    "actual": error.actual,
                    "maximum": error.maximum,
                    "result": result,
                    "pending_tool_call_ids": [tc.id for tc in backfilled],
                },
            )
            return result
        except KeyboardInterrupt:
            # Ctrl+C mid-execution would leave the assistant tool_calls message
            # without replies, poisoning the next request; backfill it first.
            backfilled = self._answer_pending_tool_calls(pending_tool_calls)
            token.cancel("keyboard interrupt")
            self._emit_backfilled_tools(backfilled, turn_id, round_index, "[interrupted]", interrupted=True)
            self._emit_event(
                RuntimeEventType.TURN_INTERRUPTED,
                turn_id,
                round_index,
                payload={"pending_tool_call_ids": [tc.id for tc in backfilled]},
            )
            raise
        except Exception as exc:
            self._emit_event(
                RuntimeEventType.TURN_FAILED,
                turn_id,
                round_index,
                payload={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        finally:
            self._active_cancellation_token = None

    def _emit_event(
        self,
        event_type: RuntimeEventType,
        turn_id: str,
        round_index: int,
        *,
        payload: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Emit an observation without allowing consumer failures to stop the Agent."""
        if event_type is RuntimeEventType.TOOL_COMPLETED:
            payload = dict(payload or {})
            outcome = tool_result(payload.get("result", ""))
            payload["result"] = str(outcome)
            payload["tool_status"] = outcome.status.value
            payload["result_facts"] = outcome.facts.to_payload()
        event = RuntimeEvent(
            event_type=event_type,
            session_id=self.session_id,
            turn_id=turn_id,
            round_index=round_index,
            tool_call_id=tool_call_id,
            payload=payload or {},
        )
        try:
            self.event_sink.emit(event)
        except Exception:
            # Runtime observers are diagnostic consumers. Their failures must
            # not change the model/tool transaction they are observing.
            pass

    def _begin_tool_turn(self) -> None:
        """Give application executors a chance to reset per-turn control state."""

        begin_turn = getattr(self.tool_executor, "begin_turn", None)
        if begin_turn is not None:
            begin_turn()

    def _consume_tool_turn_stop(self) -> str | None:
        """Read an optional application request to end this turn after a tool reply."""

        consume = getattr(self.tool_executor, "consume_turn_stop_message", None)
        if consume is None:
            return None
        return consume()

    def _finish_stopped_turn(self, turn_id: str, round_index: int, message: str) -> str:
        """Close a valid tool transaction without inviting another tool-retry round."""

        self.messages.append({"role": "assistant", "content": message})
        self._emit_event(
            RuntimeEventType.TURN_COMPLETED,
            turn_id,
            round_index,
            payload={"content": message, "stopped_after_tool": True},
        )
        return message

    def _emit_context_compressed(self, turn_id: str, round_index: int, *, phase: str) -> None:
        self._emit_event(
            RuntimeEventType.CONTEXT_COMPRESSED,
            turn_id,
            round_index,
            payload={
                "phase": phase,
                "message_count": len(self.messages),
                # This is a model-view checkpoint only. Earlier raw events are
                # still retained by the Session store for audit and rebuild.
                "message_projection": copy.deepcopy(self.messages),
            },
        )

    def _emit_tool_requested(
        self,
        tc,
        turn_id: str,
        round_index: int,
        on_tool=None,
        *,
        assistant_content: str = "",
    ) -> None:
        self._emit_event(
            RuntimeEventType.TOOL_REQUESTED,
            turn_id,
            round_index,
            tool_call_id=tc.id,
            payload={
                "tool_name": tc.name,
                "arguments": dict(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                "assistant_content": assistant_content,
            },
        )
        if on_tool:
            on_tool(tc.name, tc.arguments)

    def _exec_tool_with_event(self, tc, turn_id: str, round_index: int) -> str:
        prepared = self._prepare_tool_call(tc)
        result = self._exec_tool(
            tc,
            turn_id=turn_id,
            round_index=round_index,
            prepared_call=prepared,
        )
        self._emit_event(
            RuntimeEventType.TOOL_COMPLETED,
            turn_id,
            round_index,
            tool_call_id=tc.id,
            payload={"tool_name": tc.name, "result": result, "interrupted": False},
        )
        return result

    def _exec_tool(
        self,
        tc,
        *,
        turn_id: str | None = None,
        round_index: int = 0,
        execution_event_sink: EventSink | None = None,
        prepared_call: PreparedToolCall | None = None,
    ) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return ToolResult(f"Error: unknown tool '{tc.name}'", ToolStatus.ERROR)
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            validate_tool_arguments(tool, tc.arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(f"Error: bad arguments for {tc.name}: {e}", ToolStatus.ERROR)
        try:
            if self.tool_executor is not None:
                execute_call = getattr(self.tool_executor, "execute_call", None)
                if execute_call is not None:
                    kwargs: dict[str, Any] = {"tool_call_id": tc.id}
                    if "prepared_call" in inspect.signature(execute_call).parameters and prepared_call is not None:
                        kwargs["prepared_call"] = prepared_call
                    if "execution_context" in inspect.signature(execute_call).parameters and turn_id is not None:
                        kwargs["execution_context"] = ToolExecutionContext(
                            session_id=self.session_id,
                            turn_id=turn_id,
                            round_index=round_index,
                            event_sink=execution_event_sink or self.event_sink,
                        )
                    return tool_result(execute_call(tool, dict(tc.arguments), **kwargs))
                return tool_result(self.tool_executor.execute(tool, dict(tc.arguments)))
            return tool_result(tool.execute(**tc.arguments))
        except Exception as e:
            return ToolResult(
                f"Error executing {tc.name}: {e}\n[effect unknown] Inspect effects before retrying.",
                ToolStatus.EFFECT_UNKNOWN,
            )

    def _prepare_tool_call(self, tc) -> PreparedToolCall:
        """Prepare one validated call without triggering a Tool effect."""

        tool = self._tool_by_name.get(tc.name)
        if tool is None or not isinstance(tc.arguments, dict):
            return PreparedToolCall.create(tc.name, {}, ToolExecutionDescription.unknown())
        # Scheduling must see the same basic argument contract as execution.
        # Invalid calls remain an exclusive, no-effect error rather than being
        # allowed to enter a custom tool's declared SAFE wave.
        try:
            validate_tool_arguments(tool, tc.arguments)
        except (TypeError, ValueError):
            return PreparedToolCall.create(tc.name, tc.arguments, ToolExecutionDescription.unknown())
        prepare_call = getattr(self.tool_executor, "prepare_call", None)
        if callable(prepare_call):
            try:
                prepared = prepare_call(tool, dict(tc.arguments))
            except Exception:
                return PreparedToolCall.create(tc.name, tc.arguments, ToolExecutionDescription.unknown())
            if isinstance(prepared, PreparedToolCall):
                return prepared
            if prepared is not None:
                return PreparedToolCall.create(tc.name, tc.arguments, ToolExecutionDescription.unknown())
        declared = declared_tool_description(tool, dict(tc.arguments))
        if declared is not None:
            return PreparedToolCall.create(tc.name, tc.arguments, declared)
        describe_call = getattr(self.tool_executor, "describe_call", None)
        if callable(describe_call):
            try:
                described = describe_call(tool, dict(tc.arguments))
            except Exception:
                return PreparedToolCall.create(tc.name, tc.arguments, ToolExecutionDescription.unknown())
            if isinstance(described, ToolExecutionDescription):
                return PreparedToolCall.create(tc.name, tc.arguments, described)
            if described is not None:
                return PreparedToolCall.create(tc.name, tc.arguments, ToolExecutionDescription.unknown())
        return PreparedToolCall.create(tc.name, tc.arguments, default_tool_description(tc.name, dict(tc.arguments)))

    def _describe_tool_call(self, tc) -> ToolExecutionDescription:
        """Compatibility wrapper for callers that only need scheduling facts."""

        return self._prepare_tool_call(tc).description

    def _exec_tools_in_plan(
        self,
        tool_calls,
        *,
        turn_id: str,
        round_index: int,
        cancellation_token: CancellationToken,
    ) -> _PlannedToolResults:
        """Execute contiguous safe reads concurrently, with exclusive barriers.

        The provider has already returned the complete Tool Call round.  This
        method never speculates on streaming output and only starts a later
        wave after the previous wave completed without cancellation or an
        application-requested stop.
        """

        prepared_calls = [self._prepare_tool_call(tc) for tc in tool_calls]
        descriptions = [prepared.description for prepared in prepared_calls]
        plan = ToolExecutionPlan.build(descriptions)
        results = ["[not executed]" for _ in tool_calls]
        buffered_events = [_BufferedEventSink() for _ in tool_calls]
        started: set[int] = set()
        cancelled = False
        stop_message: str | None = None

        for wave in plan.waves:
            if cancellation_token.cancelled:
                cancelled = True
                break
            if wave.concurrent:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(wave.indexes))) as pool:
                    futures = {
                        index: pool.submit(
                            self._exec_tool,
                            tool_calls[index],
                            turn_id=turn_id,
                            round_index=round_index,
                            execution_event_sink=buffered_events[index],
                            prepared_call=prepared_calls[index],
                        )
                        for index in wave.indexes
                    }
                    for index in wave.indexes:
                        started.add(index)
                        results[index] = futures[index].result()
            else:
                index = wave.indexes[0]
                started.add(index)
                results[index] = self._exec_tool(
                    tool_calls[index],
                    turn_id=turn_id,
                    round_index=round_index,
                    execution_event_sink=buffered_events[index],
                    prepared_call=prepared_calls[index],
                )

            # The application executor may emit execution-control facts from a
            # worker. Publish them only after the whole wave and in provider
            # order so Session replay is deterministic.
            for index in wave.indexes:
                buffered_events[index].flush_to(self.event_sink)
            if cancellation_token.cancelled:
                cancelled = True
                break
            stop_message = self._consume_tool_turn_stop()
            if stop_message is not None:
                break

        if cancelled:
            for index in range(len(tool_calls)):
                if index not in started:
                    results[index] = "[cancelled before execution]"
        elif stop_message is not None:
            for index in range(len(tool_calls)):
                if index not in started:
                    results[index] = "[not executed: an earlier call stopped this turn]"

        # Tool completions and the Tool messages appended by ``chat`` use the
        # original model order even if reads finished in a different order.
        for index, (tc, result) in enumerate(zip(tool_calls, results)):
            self._emit_event(
                RuntimeEventType.TOOL_COMPLETED,
                turn_id,
                round_index,
                tool_call_id=tc.id,
                payload={
                    "tool_name": tc.name,
                    "result": result,
                    "interrupted": cancelled and index not in started,
                },
            )
        return _PlannedToolResults(results, stop_message=stop_message, cancelled=cancelled)

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        backfilled = []
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })
                backfilled.append(tc)
        return backfilled

    def _emit_backfilled_tools(self, tool_calls, turn_id: str, round_index: int, result: str, *, interrupted: bool) -> None:
        for tc in tool_calls:
            self._emit_event(
                RuntimeEventType.TOOL_COMPLETED,
                turn_id,
                round_index,
                tool_call_id=tc.id,
                payload={"tool_name": tc.name, "result": result, "interrupted": interrupted},
            )

    def _check_tool_round_limit(self, tool_rounds: int) -> None:
        maximum = self.limits.max_tool_rounds
        if maximum is not None and tool_rounds >= maximum:
            raise RuntimeLimitExceeded("tool_rounds", tool_rounds, maximum)

    def _check_runtime_controls(
        self,
        token: CancellationToken,
        started_at: float,
        provider_calls: int,
        tool_rounds: int,
        total_tokens: int,
        cost_before: float | None,
        *,
        check_provider_limit: bool = True,
    ) -> None:
        token.raise_if_cancelled()
        elapsed = time.monotonic() - started_at
        if self.limits.max_turn_seconds is not None and elapsed > self.limits.max_turn_seconds:
            raise RuntimeLimitExceeded("turn_seconds", round(elapsed, 3), self.limits.max_turn_seconds)
        if self.limits.max_input_tokens is not None:
            input_tokens = estimate_tokens(self._full_messages())
            if input_tokens > self.limits.max_input_tokens:
                raise RuntimeLimitExceeded("input_tokens", input_tokens, self.limits.max_input_tokens)
        if (
            check_provider_limit
            and self.limits.max_provider_calls is not None
            and provider_calls >= self.limits.max_provider_calls
        ):
            raise RuntimeLimitExceeded("provider_calls", provider_calls, self.limits.max_provider_calls)
        if self.limits.max_tool_rounds is not None and tool_rounds > self.limits.max_tool_rounds:
            raise RuntimeLimitExceeded("tool_rounds", tool_rounds, self.limits.max_tool_rounds)
        if self.limits.max_total_tokens is not None and total_tokens > self.limits.max_total_tokens:
            raise RuntimeLimitExceeded("total_tokens", total_tokens, self.limits.max_total_tokens)
        if self.limits.max_cost_usd is not None:
            current_cost = self._estimated_cost()
            if current_cost is not None:
                used_cost = current_cost - (cost_before or 0.0)
                if used_cost > self.limits.max_cost_usd:
                    raise RuntimeLimitExceeded("cost_usd", round(used_cost, 8), self.limits.max_cost_usd)

    def _estimated_cost(self) -> float | None:
        value = getattr(self.llm, "estimated_cost", None)
        if callable(value):
            value = value()
        return float(value) if isinstance(value, (int, float)) else None

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
