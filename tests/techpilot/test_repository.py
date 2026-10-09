from pathlib import Path

from techpilot.repository import RepositoryIndex, RepositoryProfiler

BENCHMARK_ROOT = Path(__file__).parents[2] / "benchmarks" / "cli_data_tool"


def test_repository_profiler_finds_python_project_parts():
    profile = RepositoryProfiler().profile(BENCHMARK_ROOT)

    assert profile.language == "python"
    assert "pyproject.toml" in profile.config_files
    assert "tests/test_export.py" in profile.test_files
    assert "src/cli_data_tool/cli.py" in profile.entrypoints
    assert ["python", "-m", "pytest", "-q"] in profile.validation_commands
    assert not any(path.endswith(".egg-info/PKG-INFO") for path in profile.files)


def test_repository_profile_extracts_fastapi_routes_and_import_graph(tmp_path):
    package = tmp_path / "app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "service.py").write_text("def list_items():\n    return []\n", encoding="utf-8")
    (package / "api.py").write_text(
        "from fastapi import APIRouter\n"
        "from app.service import list_items\n\n"
        "router = APIRouter()\n\n"
        '@router.get("/items")\n'
        "def get_items():\n"
        "    return list_items()\n",
        encoding="utf-8",
    )

    profile = RepositoryProfiler().profile(tmp_path)

    assert "fastapi" in profile.frameworks
    assert profile.routes["app/api.py"] == ["GET /items"]
    assert profile.import_graph["app/api.py"] == ["app/service.py"]


def test_repository_index_ignores_runtime_artifacts_and_local_secrets(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("API_KEY=\n", encoding="utf-8")
    for directory in (".tmp", "tmp", "runs", ".techpilot", ".venv", "pytest-audit-temp"):
        artifact = tmp_path / directory / "copied.py"
        artifact.parent.mkdir()
        artifact.write_text("SECRET = True\n", encoding="utf-8")

    index = RepositoryIndex.build(tmp_path)

    assert "README.md" in index.files
    assert ".env.example" in index.files
    assert ".env" not in index.files
    assert ".env.local" not in index.files
    assert not any(path.endswith("copied.py") for path in index.files)


def test_repository_index_prunes_ignored_directories_before_descending(tmp_path, monkeypatch):
    (tmp_path / "src" / "main.py").parent.mkdir()
    (tmp_path / "src" / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    ignored = tmp_path / ".venv" / "site-packages"
    ignored.mkdir(parents=True)
    (ignored / "large_generated.py").write_text("VALUE = 2\n", encoding="utf-8")
    seen_directories: list[Path] = []
    original_walk = __import__("os").walk

    def observing_walk(*args, **kwargs):
        for directory, child_directories, child_files in original_walk(*args, **kwargs):
            seen_directories.append(Path(directory))
            yield directory, child_directories, child_files

    monkeypatch.setattr("techpilot.repository.index.os.walk", observing_walk)

    index = RepositoryIndex.build(tmp_path)

    assert index.files == ["src/main.py"]
    assert ignored not in seen_directories
