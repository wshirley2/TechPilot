"""Central assembly for TechPilot Chat and future Managed Runs."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from techpilot.engine.agent import Agent, ToolExecutor
from techpilot.engine.config import Config, default_context_tokens_for_model
from techpilot.engine.events import EventSink
from techpilot.engine.llm import LLM, LiteLLM
from techpilot.engine.permissions import DenyPermissionPrompt, PermissionManager, PermissionPrompt
from techpilot.engine.runtime_control import CancellationToken, RuntimeLimits
from techpilot.engine.tools import get_tool
from techpilot.engine.tools.base import Tool

from ..chat.executor import RepositoryToolExecutor
from ..chat.permissions import ChatPermissionPolicy
from ..config.user import resolve_runtime_config
from ..repository import RepositoryProfiler
from ..repository.profiler import RepositoryProfile
from .contracts import (
    RuntimeMode,
    TaskRuntimeIdentity,
    TaskRuntimePaths,
    TaskRuntimeResult,
)
from .sessions import SessionEventSink, SessionProjection, SessionStore

ProviderFactory = Callable[[Config], object]
_CHAT_TOOL_NAMES = ("read_file", "glob", "grep", "edit_file", "write_file", "bash", "now")


@dataclass(frozen=True)
class RuntimeBootstrapInput:
    repository: Path
    event_sink: EventSink
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    tool_executor: ToolExecutor | None = None
    tools: list[Tool] | None = None
    system_context: str | None = None
    permission_mode: str | None = None
    permission_prompt: PermissionPrompt | None = None
    session_directory: Path | None = None
    resume_session_id: str | None = None
    limits: RuntimeLimits | None = None
    mode: RuntimeMode = RuntimeMode.CHAT
    task_id: str | None = None
    run_id: str | None = None
    source_repository: Path | None = None
    paths: TaskRuntimePaths | None = None


@dataclass(frozen=True)
class ActiveRole:
    """The Runtime-owned identity of the Role currently affecting a Chat."""

    role_id: str
    context: str
    tool_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RoleRuntimeSnapshot:
    active_role: ActiveRole | None
    tools: tuple[Tool, ...]


@dataclass
class TaskRuntime:
    identity: TaskRuntimeIdentity
    paths: TaskRuntimePaths
    repository: Path
    config: Config
    agent: Agent
    tools: list[Tool]
    profile: RepositoryProfile | None
    profile_warning: str | None
    permission_mode: str = "repository reads allowed; writes and commands policy-gated"
    permission_manager: PermissionManager | None = None
    session_store: SessionStore | None = None
    session_sink: SessionEventSink | None = None
    base_system_context: str = ""
    role_context: str | None = None
    active_role: ActiveRole | None = None
    base_tools: tuple[Tool, ...] = ()
    role_tool_catalog: dict[str, Tool] = field(default_factory=dict)
    recovery_notices: list[str] = field(default_factory=list)

    @property
    def runtime_mode(self) -> RuntimeMode:
        return self.identity.mode

    @property
    def mode(self) -> str:
        """Compatibility display label retained for existing CLI integrations."""

        return self.identity.mode.display_name

    @property
    def task_id(self) -> str | None:
        return self.identity.task_id

    @property
    def run_id(self) -> str | None:
        return self.identity.run_id

    @property
    def last_result(self) -> TaskRuntimeResult | None:
        return self.session_sink.last_result if self.session_sink is not None else None

    @property
    def session_path(self) -> Path | None:
        if self.session_store is None:
            return None
        return self.session_store.path_for(self.identity.session_id)

    def record_result(self, result: TaskRuntimeResult) -> None:
        """Record a complete Runtime result after orchestration-level work finishes."""

        if self.session_sink is not None:
            self.session_sink.record_result(self.identity.session_id, result)

    def run_turn(
        self,
        user_input: str,
        *,
        cancellation_token: CancellationToken | None = None,
        allow_tools: bool = True,
    ) -> str:
        """Run one controlled Agent turn through the shared Runtime boundary."""

        return self.agent.chat(
            user_input,
            cancellation_token=cancellation_token,
            allow_tools=allow_tools,
        )

    def activate_role(
        self,
        role_id: str,
        role_context: str,
        *,
        tool_names: tuple[str, ...] = (),
    ) -> None:
        """Activate a code-owned Role without creating a second Agent Runtime.

        Tool and system-context updates are applied as one Runtime transition.
        If either Agent update fails, the previously active Role remains in
        effect and no partially activated Role becomes visible to later turns.
        """

        # ``base_tools`` is the Runtime-owned default projection. An empty
        # default set is valid; falling back to ``self.tools`` would retain a
        # prior Role's tools across a switch.
        enabled = list(self.base_tools)
        for name in tool_names:
            tool = self.role_tool_catalog.get(name)
            if tool is None:
                raise ValueError(f"unknown role tool: {name}")
            if all(existing.name != name for existing in enabled):
                enabled.append(tool)
        next_role = ActiveRole(role_id=role_id, context=role_context, tool_names=tool_names)
        previous = self._role_snapshot()
        self._apply_role_state(next_role, enabled, rollback=previous)
        if self.session_sink is not None:
            self.session_sink.record(
                "role_activated",
                self.agent.session_id,
                {"role_id": role_id, "tool_names": list(tool_names)},
            )

    def clear_role(self) -> None:
        """Return subsequent turns to the default Runtime role."""

        if self.active_role is None:
            return
        previous = self._role_snapshot()
        self._apply_role_state(None, list(self.base_tools), rollback=previous)
        if self.session_sink is not None:
            self.session_sink.record("role_cleared", self.agent.session_id, {"role_id": previous.active_role.role_id})

    @contextmanager
    def role_scope(
        self,
        role_id: str,
        role_context: str,
        *,
        tool_names: tuple[str, ...] = (),
    ) -> Iterator[ActiveRole]:
        """Temporarily activate a Role and always restore the prior Runtime state."""

        previous = self._role_snapshot()
        self.activate_role(role_id, role_context, tool_names=tool_names)
        try:
            if self.active_role is None:  # Defensive: activation must have created this state.
                raise RuntimeError("Role activation did not establish an active Role")
            yield self.active_role
        finally:
            self._restore_role_snapshot(previous)

    def _role_snapshot(self) -> _RoleRuntimeSnapshot:
        return _RoleRuntimeSnapshot(active_role=self.active_role, tools=tuple(self.tools))

    def _restore_role_snapshot(self, previous: _RoleRuntimeSnapshot) -> None:
        if previous.active_role is None:
            self.clear_role()
            return
        self.activate_role(
            previous.active_role.role_id,
            previous.active_role.context,
            tool_names=previous.active_role.tool_names,
        )

    def _apply_role_state(
        self,
        active_role: ActiveRole | None,
        tools: list[Tool],
        *,
        rollback: _RoleRuntimeSnapshot,
    ) -> None:
        role_context = active_role.context if active_role is not None else None
        system_context = self._system_context(role_context)
        try:
            self.agent.update_tools(tools)
            self.agent.update_system_context(system_context)
        except Exception:
            self._restore_agent_role_state(rollback)
            raise
        self.tools = list(tools)
        self.role_context = role_context
        self.active_role = active_role

    def _restore_agent_role_state(self, previous: _RoleRuntimeSnapshot) -> None:
        previous_context = previous.active_role.context if previous.active_role is not None else None
        try:
            self.agent.update_tools(list(previous.tools))
            self.agent.update_system_context(self._system_context(previous_context))
        except Exception:
            # Preserve the original activation error. The Runtime fields still
            # describe the last fully established state; the next explicit
            # transition can retry the Agent projection.
            pass

    def _refresh_system_context(self) -> None:
        self.agent.update_system_context(self._system_context(self.role_context))

    def _system_context(self, role_context: str | None) -> str:
        context = "\n\n".join(part for part in (self.base_system_context, role_context) if part)
        return _with_runtime_identity(context, self.mode, self.config.model)

    def consume_recovery_notices(self) -> list[str]:
        """Return compatibility notices that must be shown after Session recovery."""

        notices = list(self.recovery_notices)
        self.recovery_notices.clear()
        return notices

    def ensure_persisted(self) -> None:
        """Require the latest Runtime and Session facts to be durable."""

        if self.session_sink is not None:
            self.session_sink.ensure_persisted()

    def set_model(self, model: str, *, record: bool = True) -> None:
        """Update both the provider configuration and model-visible runtime facts."""

        self.config.model = model
        if "TECHPILOT_MAX_CONTEXT" not in os.environ:
            self.config.max_context_tokens = default_context_tokens_for_model(model)
            self.agent.context.max_tokens = self.config.max_context_tokens
        if hasattr(self.agent.llm, "model"):
            self.agent.llm.model = model
        self._refresh_system_context()
        if record and self.session_sink is not None:
            self.session_sink.record("session_model_changed", self.agent.session_id, {"model": model})

    def resume_session(self, session_id: str) -> SessionProjection:
        """Load a Session projection without replaying prior filesystem effects."""

        if self.session_store is None:
            raise RuntimeError("Event-based Session storage is not configured")
        projection = self.session_store.replay(session_id)
        if projection.repository_root is not None and projection.repository_root != self.repository:
            raise ValueError("Session belongs to a different repository")
        if projection.mode != self.identity.mode.value:
            raise ValueError("Session belongs to a different Runtime mode")
        self.agent.session_id = projection.session_id
        self.agent.messages = projection.model_messages
        self.identity = TaskRuntimeIdentity(
            mode=self.identity.mode,
            session_id=projection.session_id,
            task_id=projection.task_id,
            run_id=projection.run_id,
            source_repository=(projection.source_repository_root or self.identity.source_repository),
            working_directory=self.repository,
        )
        if projection.model:
            self.set_model(projection.model, record=False)
        if self.permission_manager is not None:
            # Grants are intentionally process-local. A resumed session always
            # reuses C3's Trusted Diff and fresh approval path.
            self.permission_manager.clear_session_grants()
        set_task_id = getattr(self.agent.tool_executor, "set_task_id", None)
        if callable(set_task_id):
            set_task_id(projection.task_id)
        legacy_isolations = list(projection.pending_isolation_requests)
        if legacy_isolations:
            notice = (
                f"发现 {len(legacy_isolations)} 个旧版待隔离操作：它们此前未执行，"
                "当前版本不会自动恢复、执行或改写为源仓库写入。"
            )
            self.recovery_notices.append(notice)
            self.agent.messages.append({"role": "system", "content": notice})
            projection.warnings.append("旧版待隔离操作已冻结；当前 Chat 不会自动执行。")
        recovered_role_id = _last_active_role_id(projection)
        if recovered_role_id is not None:
            notice = f"发现历史 Role 激活记录（{recovered_role_id}）；当前版本不会自动恢复 Role、工具或历史授权。"
            self.recovery_notices.append(notice)
            self.agent.messages.append({"role": "system", "content": notice})
            projection.warnings.append("历史 Role 已保持关闭；当前 Chat 使用默认 Coding 状态。")
        if self.session_sink is not None:
            self.session_sink.last_result = projection.last_result
            self.session_sink.record("session_resumed", projection.session_id, {
                "repository_root": str(self.repository),
                "event_count": len(projection.events),
                "warnings": projection.warnings,
            })
        return projection


# Compatibility alias for integrations written before the unified Task Runtime contract.
ChatRuntime = TaskRuntime


def _last_active_role_id(projection: SessionProjection) -> str | None:
    """Read lifecycle facts without turning historical Role state back on."""

    role_id: str | None = None
    for event in projection.events:
        if event.event_type == "role_activated":
            candidate = event.payload.get("role_id")
            if isinstance(candidate, str):
                role_id = candidate
        elif event.event_type == "role_cleared":
            role_id = None
    return role_id


class RuntimeBootstrap:
    """Build the complete runtime once so CLI commands do not duplicate it."""

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        profiler: RepositoryProfiler | None = None,
    ):
        self.provider_factory = provider_factory
        self.profiler = profiler or RepositoryProfiler()

    def build(self, inputs: RuntimeBootstrapInput) -> TaskRuntime:
        repository = inputs.repository.resolve()
        if not repository.is_dir():
            raise ValueError(f"Repository directory does not exist: {inputs.repository}")
        mode = RuntimeMode(inputs.mode)
        paths = inputs.paths or TaskRuntimePaths.for_runtime(
            mode,
            repository,
            inputs.session_directory,
        )
        if inputs.paths is not None and inputs.session_directory is not None:
            raise ValueError("Pass either Runtime paths or a Session directory, not both")
        if paths.mode is not mode or paths.working_directory != repository:
            raise ValueError("Runtime paths do not match the requested mode and working directory")
        source_repository = (inputs.source_repository or repository).resolve()
        if not source_repository.is_dir():
            raise ValueError(f"Source repository directory does not exist: {source_repository}")
        _validate_runtime_scope(mode, inputs.task_id, inputs.run_id)

        config = resolve_runtime_config(
            model=inputs.model,
            base_url=inputs.base_url,
            api_key=inputs.api_key,
        )

        if self.provider_factory is None and not config.api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY or pass --api-key."
            )

        profile = None
        profile_warning = None
        try:
            profile = self.profiler.profile(repository)
        except Exception as error:
            profile_warning = f"Repository profile unavailable: {error}"

        tools = (
            inputs.tools
            if inputs.tools is not None
            else [tool for name in _CHAT_TOOL_NAMES if (tool := get_tool(name)) is not None]
        )
        session_store = SessionStore(paths.session_directory)
        resume_projection = None
        if inputs.resume_session_id:
            resume_projection = session_store.replay(inputs.resume_session_id)
            if resume_projection.repository_root is not None and resume_projection.repository_root != repository:
                raise ValueError("Session belongs to a different repository")
            if resume_projection.mode != mode.value:
                raise ValueError("Session belongs to a different Runtime mode")
            if resume_projection.model and not inputs.model:
                config.model = resume_projection.model

        # Recalculate the provider window after CLI, environment, and session
        # model overrides unless the user explicitly pins the context size.
        if "TECHPILOT_MAX_CONTEXT" not in os.environ:
            config.max_context_tokens = default_context_tokens_for_model(config.model)

        provider = self._build_provider(config)
        repository_context = _repository_summary(profile, profile_warning)
        base_system_context = "\n\n".join(
            part for part in (repository_context, inputs.system_context) if part
        )
        system_context = _with_runtime_identity(base_system_context, mode.display_name, config.model)
        permission_manager = None
        tool_executor = inputs.tool_executor
        if tool_executor is None:
            permission_manager = PermissionManager(
                ChatPermissionPolicy(profile.validation_commands if profile else ()),
                inputs.permission_prompt or DenyPermissionPrompt(),
            )
            tool_executor = RepositoryToolExecutor(repository, permission_manager, task_id=inputs.task_id)
        session_id = resume_projection.session_id if resume_projection is not None else None
        session_sink = SessionEventSink(session_store, inputs.event_sink)
        agent = Agent(
            llm=provider,
            tools=tools,
            max_context_tokens=config.max_context_tokens,
            tool_executor=tool_executor,
            event_sink=session_sink,
            session_id=session_id,
            working_directory=str(repository),
            assistant_name="TechPilot",
            system_context=system_context,
            limits=inputs.limits,
        )
        if resume_projection is not None:
            agent.messages = resume_projection.model_messages
        else:
            session_store.create(
                agent.session_id,
                repository_root=repository,
                model=config.model,
                mode=mode.value,
                task_id=inputs.task_id,
                run_id=inputs.run_id,
                source_repository_root=source_repository,
            )
        runtime = TaskRuntime(
            identity=TaskRuntimeIdentity(
                mode=mode,
                session_id=agent.session_id,
                task_id=inputs.task_id,
                run_id=inputs.run_id,
                source_repository=source_repository,
                working_directory=repository,
            ),
            paths=paths,
            repository=repository,
            config=config,
            agent=agent,
            tools=tools,
            profile=profile,
            profile_warning=profile_warning,
            permission_mode=(
                inputs.permission_mode
                or "repository reads allowed; writes and commands policy-gated"
            ),
            permission_manager=permission_manager,
            session_store=session_store,
            session_sink=session_sink,
            base_system_context=base_system_context,
            base_tools=tuple(tools),
            role_tool_catalog={
                name: tool
                for name in ("fetch_url", "agent")
                if (tool := get_tool(name)) is not None
            },
        )
        if resume_projection is not None:
            runtime.resume_session(resume_projection.session_id)
        return runtime

    def _build_provider(self, config: Config):
        if self.provider_factory is not None:
            return self.provider_factory(config)
        provider_class = LiteLLM if config.provider == "litellm" else LLM
        return provider_class(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


def _repository_summary(profile: RepositoryProfile | None, warning: str | None) -> str:
    if profile is None:
        return warning or "Repository profile is unavailable; inspect files with tools as needed."

    def values(items: list[str]) -> str:
        return ", ".join(items[:8]) or "(none detected)"

    commands = [" ".join(command) for command in profile.validation_commands]
    return "\n".join([
        f"Repository root: {profile.root}",
        f"Language: {profile.language}",
        f"Frameworks: {values(profile.frameworks)}",
        f"Entrypoints: {values(profile.entrypoints)}",
        f"Tests: {values(profile.test_files)}",
        f"Validation commands: {values(commands)}",
        "This is a lightweight profile. Read detailed files only when needed.",
    ])


def _with_runtime_identity(base_context: str, mode: str, model: str) -> str:
    """Inject only runtime facts that the Agent may safely explain to the user."""

    identity = "\n".join([
        "Runtime identity (authoritative facts):",
        "Product: TechPilot",
        f"Mode: {mode}",
        f"Current model: {model}",
        "For questions about product, mode, or model, answer from these facts without calling tools.",
        "Do not reveal API keys, endpoints, or other provider secrets.",
    ])
    return "\n\n".join(part for part in (base_context, identity) if part)


def _validate_runtime_scope(mode: RuntimeMode, task_id: str | None, run_id: str | None) -> None:
    """Reject incomplete correlation data before creating Session artifacts."""

    if run_id is not None and task_id is None:
        raise ValueError("A Runtime run id requires a task id")
    if mode is RuntimeMode.MANAGED_RUN and (task_id is None or run_id is None):
        raise ValueError("Managed Run Runtime requires task and run ids")
