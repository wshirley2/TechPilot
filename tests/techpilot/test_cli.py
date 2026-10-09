import json
from types import SimpleNamespace

import pytest

from techpilot.cli import main
from techpilot.runtime import RuntimeBootstrap
from techpilot.runtime.sessions import SessionStore


def test_help_exposes_chat_and_session_utilities_without_retired_flows(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "chat" in help_text
    assert "sessions" in help_text
    assert "plan" not in help_text
    assert "workspace" not in help_text
    assert "profile" not in help_text


@pytest.mark.parametrize(("arguments", "message"), [
    (["run", "approved-plan-v1"], "Managed Run execution was removed"),
    (["plan", "create"], "Plan and Workspace commands were removed"),
    (["plans", "list"], "Plan and Workspace commands were removed"),
    (["workspace", "create"], "Plan and Workspace commands were removed"),
    (["profile", "."], "standalone profile command was removed"),
])
def test_retired_commands_fail_before_execution(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_default_chat_entry_uses_the_shared_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = tmp_path / "repository"
    repository.mkdir()
    captured = {}

    class Provider:
        model = "fake-unified"

    class FakeChatSession:
        def __init__(self, runtime, **kwargs):
            captured["runtime"] = runtime
            captured["kwargs"] = kwargs

        def run(self):
            return 9

    bootstrap = RuntimeBootstrap(provider_factory=lambda config: Provider())
    monkeypatch.setattr("techpilot.cli.RuntimeBootstrap", lambda: bootstrap)
    monkeypatch.setattr("techpilot.cli.ChatSession", FakeChatSession)

    result = main([
        "chat",
        str(repository),
        "--max-tool-rounds",
        "4",
    ])

    assert result == 9
    assert captured["runtime"].repository == repository.resolve()
    assert "plan_session" not in captured["kwargs"]


def test_chat_tui_option_uses_the_same_runtime_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = tmp_path / "repository"
    repository.mkdir()
    captured = {}

    class Provider:
        model = "fake-unified"

    class FakeTui:
        event_sink = SimpleNamespace(emit=lambda _event: None)
        permission_prompt = object()

        def bind_runtime(self, runtime):
            captured["runtime"] = runtime

        def run(self):
            return 7

    bootstrap = RuntimeBootstrap(provider_factory=lambda config: Provider())
    monkeypatch.setattr("techpilot.cli.RuntimeBootstrap", lambda: bootstrap)
    monkeypatch.setattr("techpilot.cli.TechPilotTui", FakeTui)
    monkeypatch.setattr("techpilot.cli.tui_supported", lambda: True)

    assert main(["chat", str(repository), "--tui"]) == 7
    assert captured["runtime"].repository == repository.resolve()


def test_chat_tui_option_falls_back_to_the_standard_cli_without_a_tty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TECHPILOT_LOAD_DOTENV", "0")
    repository = tmp_path / "repository"
    repository.mkdir()

    class Provider:
        model = "fake-unified"

    class FakeChatSession:
        def __init__(self, runtime, **_kwargs):
            assert runtime.repository == repository.resolve()

        def run(self):
            return 6

    bootstrap = RuntimeBootstrap(provider_factory=lambda config: Provider())
    monkeypatch.setattr("techpilot.cli.RuntimeBootstrap", lambda: bootstrap)
    monkeypatch.setattr("techpilot.cli.ChatSession", FakeChatSession)
    monkeypatch.setattr("techpilot.cli.tui_supported", lambda: False)

    assert main(["chat", str(repository), "--tui"]) == 6
    assert "requires an interactive TTY" in capsys.readouterr().err


def test_tui_workspace_trust_gate_runs_before_runtime_bootstrap(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    captured = {}

    class FakeTui:
        event_sink = SimpleNamespace(emit=lambda _event: None)
        permission_prompt = object()

        def bind_runtime(self, runtime):
            captured["runtime"] = runtime

        def run(self):
            return 3

    class FailingBootstrap:
        def build(self, _inputs):
            raise AssertionError("trust denial must happen before RuntimeBootstrap.build")

    monkeypatch.setattr("techpilot.cli.TechPilotTui", FakeTui)
    monkeypatch.setattr("techpilot.cli.tui_supported", lambda: True)
    monkeypatch.setattr("techpilot.cli.sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("techpilot.cli.confirm_workspace_access", lambda path, console: False)
    monkeypatch.setattr("techpilot.cli.RuntimeBootstrap", FailingBootstrap)

    assert main(["chat", str(repository), "--tui"]) == 0
    assert "runtime" not in captured


def test_sessions_commands_list_and_show_event_sessions(tmp_path, capsys):
    repository = tmp_path / "repository"
    repository.mkdir()
    store = SessionStore.for_repository(repository)
    store.create("session-for-cli", repository_root=repository, model="fake-model")

    assert main(["sessions", "list", str(repository)]) == 0
    assert "session-for-cli" in capsys.readouterr().out

    assert main(["sessions", "show", "session-for-cli", str(repository)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == "session-for-cli"
    assert payload["model"] == "fake-model"
