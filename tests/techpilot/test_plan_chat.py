"""Minimal conversational Plan entry tests."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from techpilot.advanced.managed import ManagedRunService
from techpilot.advanced.plan_chat import PlanChatSession
from techpilot.advanced.planning import PlanningService, PlanStore
from techpilot.advanced.workspace import CopyWorkspaceBackend, WorkspaceCreationError, WorkspaceService
from techpilot.chat.session import ChatSession, TerminalEventSink
from techpilot.engine.llm import LLMResponse, ToolCall
from techpilot.execution.validation import ValidationCommandResult, ValidationService
from techpilot.runtime import RuntimeBootstrap, RuntimeBootstrapInput


class FakeProvider:
    model = "fake-plan-chat"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, tools=None, on_token=None):
        self.requests.append({"messages": messages, "tools": tools})
        response = next(self.responses)
        if on_token and response.content:
            on_token(response.content)
        return response


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\npythonpath = ['.']\n[tool.ruff]\nline-length = 100\n",
        encoding="utf-8",
    )
    (repository / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    return repository


class PassingValidationRunner:
    """Keep Plan Chat tests independent of a nested pytest child process."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def execute(self, command, workspace_path: Path, *, cancellation_token=None) -> ValidationCommandResult:
        del cancellation_token
        argv = list(command)
        self.commands.append(argv)
        return ValidationCommandResult(
            argv=argv,
            resolved_argv=argv,
            cwd=str(workspace_path.resolve()),
            status="passed",
            exit_code=0,
            duration_seconds=0.0,
            stdout="1 passed",
            stderr="",
        )


def _session(
    tmp_path: Path,
    repository: Path,
    provider: FakeProvider,
    inputs: list[str],
    *,
    validation_service: ValidationService | None = None,
):
    store = PlanStore(tmp_path / "plans")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    sink = TerminalEventSink(console)
    session = PlanChatSession(
        repository,
        planning_service=PlanningService(store),
        plan_store=store,
        managed_service=ManagedRunService(
            plan_store=store,
            workspace_service=WorkspaceService(CopyWorkspaceBackend(tmp_path / "runs")),
            runtime_bootstrap=RuntimeBootstrap(provider_factory=lambda config: provider),
            event_sink=sink,
            validation_service=validation_service,
        ),
        console=console,
        input_fn=lambda prompt: inputs.pop(0),
    )
    return session, store, output


def test_plan_chat_requires_explicit_approval_then_runs_in_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = _repository(tmp_path)
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            "write-1",
            "write_file",
            {"file_path": "README.md", "content": "# Demo\n\nPlan Chat verification\n"},
        )]),
        LLMResponse(content="Conversational plan completed."),
    ])
    validation_runner = PassingValidationRunner()
    session, store, output = _session(
        tmp_path,
        repository,
        provider,
        ["Append Plan Chat verification to README.md", "执行", "批准并执行。"],
        validation_service=ValidationService(validation_runner),
    )

    assert session.run() == 0

    records = store.list(repository=repository)
    assert len(records) == 1
    assert records[0].status == "approved"
    metadata_paths = list((tmp_path / "runs").glob("*/run.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    workspace = Path(metadata["workspace_path"])
    assert metadata["status"] == "succeeded"
    assert "Plan Chat verification" in (workspace / "README.md").read_text(encoding="utf-8")
    assert (repository / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    rendered = output.getvalue()
    assert "计划尚未批准" in rendered
    assert "Managed Run 执行与验证完成" in rendered
    assert "系统验证：passed" in rendered
    assert "Events:" in rendered
    assert validation_runner.commands
    assert (metadata_paths[0].parent / "events.jsonl").is_file()


def test_plan_chat_natural_language_revision_creates_next_version(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = _repository(tmp_path)
    session, store, output = _session(
        tmp_path,
        repository,
        FakeProvider([]),
        [
            "Append first note to README.md",
            "Append revised verification note to README.md",
            "/exit",
        ],
    )

    assert session.run() == 0

    records = store.list(repository=repository)
    assert {record.version for record in records} == {1, 2}
    assert len({record.plan.task_id for record in records}) == 1
    latest = next(record for record in records if record.version == 2)
    assert latest.plan.summary == "Append revised verification note to README.md"
    assert "已将输入作为新的完整任务描述" in output.getvalue()


def test_plan_chat_shortcuts_keep_safe_non_execution_defaults(tmp_path):
    repository = _repository(tmp_path)
    session, _store, output = _session(tmp_path, repository, FakeProvider([]), [])

    session.handle("Append a note to README.md")
    assert session.record is not None and session.record.status == "draft"
    assert session.handle("2").action == "continue"
    assert session.record.status == "draft"
    assert session.handle("3").action == "completed"
    assert session.record.status == "rejected"
    assert not list((tmp_path / "runs").glob("*/run.json"))
    assert "[1] 批准并执行" in output.getvalue()
    assert not (tmp_path / "runs").exists()


def test_chat_does_not_embed_plan_mode_or_create_a_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = _repository(tmp_path)
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    sink = TerminalEventSink(console)
    chat_provider = FakeProvider([
        LLMResponse(content="普通 Chat 正常。"),
        LLMResponse(content="这个请求仍由普通 Chat 处理。"),
    ])
    runtime = RuntimeBootstrap(provider_factory=lambda config: chat_provider).build(RuntimeBootstrapInput(
        repository=repository,
        event_sink=sink,
    ))
    inputs = iter([
        "介绍一下仓库",
        "我想先制定计划：Append Unified Plan verification to README.md",
        "/exit",
    ])
    prompts = []

    def input_fn(prompt):
        prompts.append(prompt)
        return next(inputs)

    session = ChatSession(runtime, console=console, input_fn=input_fn)

    assert session.run() == 0

    rendered = output.getvalue()
    assert rendered.count("TechPilot Chat") == 1
    assert "TechPilot Plan Chat" not in rendered
    assert "Plan > " not in prompts
    assert prompts == ["You > ", "You > ", "You > "]
    assert len(chat_provider.requests) == 2
    assert not (tmp_path / "runs").exists()


def test_workspace_creation_failure_stays_in_plan_mode_with_a_brief_retry_message(tmp_path):
    repository = _repository(tmp_path)
    store = PlanStore(tmp_path / "plans")
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)

    class FailingManagedService:
        def execute(self, plan_reference, **kwargs):
            raise WorkspaceCreationError("Could not create workspace: access denied (locked.txt)")

    session = PlanChatSession(
        repository,
        planning_service=PlanningService(store),
        plan_store=store,
        managed_service=FailingManagedService(),
        console=console,
    )

    session.handle("Append retry note to README.md")
    outcome = session.handle("批准并执行")

    assert outcome.action == "continue"
    assert session.record is not None and session.record.status == "approved"
    rendered = output.getvalue()
    assert "Managed Run 未启动" in rendered
    assert "原仓库未修改" in rendered
    assert "仓库：" in rendered


def test_plan_chat_reports_system_validation_failure_and_retains_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = _repository(tmp_path)
    (repository / "test_smoke.py").write_text("def test_smoke():\n    assert False\n", encoding="utf-8")
    session, _, output = _session(
        tmp_path,
        repository,
        FakeProvider([LLMResponse(content="Agent implementation finished.")]),
        ["Append a validation failure note to README.md", "批准并执行"],
    )

    assert session.run() == 1

    metadata_path = next((tmp_path / "runs").glob("*/run.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert (metadata_path.parent / "workspace").is_dir()
    assert (metadata_path.parent / "validation.json").is_file()
    rendered = output.getvalue()
    assert "系统验证：failed" in rendered
    assert "Managed Run 验证未通过" in rendered
