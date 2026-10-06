import pytest

from agent.tools import ToolRegistry, build_tools
from agent.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    """A small project directory, with a file next to it that must stay out of reach."""
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "main.py").write_text('print("hello from main")\n', encoding="utf-8")
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    return Workspace(root)


@pytest.fixture
def registry(workspace):
    return ToolRegistry(workspace, build_tools())
