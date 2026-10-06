"""Every path the model supplies must stay inside the working directory."""
import os

import pytest

from agent.workspace import Workspace, WorkspaceError


def test_resolves_paths_inside_the_workspace(workspace):
    assert workspace.resolve("main.py") == workspace.root / "main.py"
    assert workspace.resolve("pkg/mod.py") == workspace.root / "pkg" / "mod.py"
    assert workspace.resolve("pkg/../main.py") == workspace.root / "main.py"
    assert workspace.resolve("not/created/yet.py") == workspace.root / "not" / "created" / "yet.py"


def test_the_workspace_root_itself_is_allowed(workspace):
    assert workspace.resolve(".") == workspace.root
    assert workspace.resolve("") == workspace.root


def test_an_absolute_path_inside_the_workspace_is_allowed(workspace):
    assert workspace.resolve(str(workspace.root / "main.py")) == workspace.root / "main.py"


@pytest.mark.parametrize("path", ["../secret.txt", "pkg/../../secret.txt", "..", "/etc/passwd", "/bin"])
def test_rejects_paths_that_leave_the_workspace(workspace, path):
    with pytest.raises(WorkspaceError, match="outside the permitted working directory"):
        workspace.resolve(path)


def test_rejects_an_absolute_path_to_a_sibling_file(workspace):
    with pytest.raises(WorkspaceError):
        workspace.resolve(str(workspace.root.parent / "secret.txt"))


def test_rejects_a_sibling_directory_that_shares_the_name_prefix(workspace):
    # "project-other" starts with "project" but is not inside it.
    (workspace.root.parent / "project-other").mkdir()
    with pytest.raises(WorkspaceError):
        workspace.resolve("../project-other/file.txt")


def test_rejects_a_symlink_that_points_outside(workspace):
    os.symlink(workspace.root.parent / "secret.txt", workspace.root / "link.txt")
    with pytest.raises(WorkspaceError, match="outside"):
        workspace.resolve("link.txt")


def test_rejects_a_path_through_a_symlinked_directory(workspace):
    os.symlink(workspace.root.parent, workspace.root / "up", target_is_directory=True)
    with pytest.raises(WorkspaceError, match="outside"):
        workspace.resolve("up/secret.txt")
    with pytest.raises(WorkspaceError, match="outside"):
        workspace.resolve("up/new_file.txt")


def test_a_symlink_that_stays_inside_is_allowed(workspace):
    os.symlink(workspace.root / "main.py", workspace.root / "alias.py")
    assert workspace.resolve("alias.py") == workspace.root / "main.py"


def test_rejects_a_path_with_a_null_byte(workspace):
    with pytest.raises(WorkspaceError, match="not a usable path"):
        workspace.resolve("main.py\x00.txt")


def test_the_working_directory_must_exist(tmp_path):
    with pytest.raises(NotADirectoryError):
        Workspace(tmp_path / "missing")
