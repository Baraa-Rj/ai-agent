"""The four tools, called directly."""
import os

import pytest

from agent import tools
from agent.tools import (
    ToolError,
    get_file_content,
    get_files_info,
    run_python_file,
    write_file,
)
from agent.workspace import WorkspaceError

# ---------------------------------------------------------------- get_files_info

def test_lists_entries_sorted_with_size_and_kind(workspace):
    assert get_files_info(workspace).splitlines() == [
        "- main.py: file_size=25 bytes, is_dir=False",
        f"- pkg: file_size={(workspace.root / 'pkg').lstat().st_size} bytes, is_dir=True",
    ]
    assert get_files_info(workspace, "pkg") == "- mod.py: file_size=10 bytes, is_dir=False"


def test_listing_an_empty_directory_still_returns_text(workspace):
    (workspace.root / "empty").mkdir()
    assert get_files_info(workspace, "empty") == "(empty directory)"


def test_listing_survives_a_broken_symlink(workspace):
    os.symlink(workspace.root / "gone.txt", workspace.root / "dangling")
    assert "- dangling:" in get_files_info(workspace)


def test_listing_a_file_or_a_missing_path_is_an_error(workspace):
    for path in ("main.py", "nope"):
        with pytest.raises(ToolError, match="is not a directory"):
            get_files_info(workspace, path)


def test_listing_outside_the_workspace_is_refused(workspace):
    with pytest.raises(WorkspaceError):
        get_files_info(workspace, "..")


# -------------------------------------------------------------- get_file_content

def test_reads_a_file(workspace):
    assert get_file_content(workspace, "pkg/mod.py") == "VALUE = 1\n"


def test_long_files_are_truncated_and_say_so(workspace):
    (workspace.root / "big.txt").write_text("x" * (tools.MAX_FILE_CHARS + 500), encoding="utf-8")
    content = get_file_content(workspace, "big.txt")
    assert content.startswith("x" * tools.MAX_FILE_CHARS)
    assert content.endswith(f'[...File "big.txt" truncated at {tools.MAX_FILE_CHARS} characters]')
    assert len(content) < tools.MAX_FILE_CHARS + 100


def test_a_file_of_exactly_the_limit_is_not_marked_truncated(workspace):
    (workspace.root / "exact.txt").write_text("x" * tools.MAX_FILE_CHARS, encoding="utf-8")
    assert get_file_content(workspace, "exact.txt") == "x" * tools.MAX_FILE_CHARS


def test_reading_an_empty_file_still_returns_text(workspace):
    (workspace.root / "empty.txt").write_text("", encoding="utf-8")
    assert get_file_content(workspace, "empty.txt") == "(empty file)"


def test_reading_a_missing_file_or_a_directory_is_an_error(workspace):
    for path in ("missing.py", "pkg"):
        with pytest.raises(ToolError, match="is not a file"):
            get_file_content(workspace, path)


def test_reading_a_binary_file_is_an_error(workspace):
    (workspace.root / "data.bin").write_bytes(b"\xff\xfe\x00\x81")
    with pytest.raises(ToolError, match="not UTF-8 text"):
        get_file_content(workspace, "data.bin")


def test_reading_outside_the_workspace_is_refused(workspace):
    with pytest.raises(WorkspaceError):
        get_file_content(workspace, "../secret.txt")


# -------------------------------------------------------------------- write_file

def test_writes_a_new_file_and_creates_parent_directories(workspace):
    message = write_file(workspace, "new/deep/file.txt", "héllo")
    assert message == 'Successfully wrote to "new/deep/file.txt" (5 characters written)'
    assert (workspace.root / "new" / "deep" / "file.txt").read_text(encoding="utf-8") == "héllo"


def test_overwrites_an_existing_file(workspace):
    write_file(workspace, "main.py", "print('changed')\n")
    assert (workspace.root / "main.py").read_text(encoding="utf-8") == "print('changed')\n"


def test_writing_to_a_directory_is_an_error(workspace):
    for path in ("pkg", "."):
        with pytest.raises(ToolError, match="is a directory"):
            write_file(workspace, path, "x")


def test_writing_outside_the_workspace_is_refused_and_writes_nothing(workspace):
    outside = workspace.root.parent / "escaped.txt"
    for path in ("../escaped.txt", str(outside)):
        with pytest.raises(WorkspaceError):
            write_file(workspace, path, "should not be written")
    assert not outside.exists()


def test_writing_through_a_symlink_to_an_outside_file_is_refused(workspace):
    secret = workspace.root.parent / "secret.txt"
    os.symlink(secret, workspace.root / "link.txt")
    with pytest.raises(WorkspaceError):
        write_file(workspace, "link.txt", "overwritten")
    assert secret.read_text(encoding="utf-8") == "top secret"


# --------------------------------------------------------------- run_python_file

def test_runs_a_script_and_returns_its_output(workspace):
    assert run_python_file(workspace, "main.py") == "STDOUT:\nhello from main\n"


def test_passes_arguments_and_runs_in_the_workspace_directory(workspace):
    (workspace.root / "show.py").write_text(
        "import os, sys\nprint(sys.argv[1:], os.path.basename(os.getcwd()))\n", encoding="utf-8")
    assert run_python_file(workspace, "show.py", ["3 + 5", "x"]) == "STDOUT:\n['3 + 5', 'x'] project\n"


def test_reports_a_failing_script_with_its_exit_code_and_stderr(workspace):
    (workspace.root / "fail.py").write_text(
        "import sys\nprint('oops', file=sys.stderr)\nsys.exit(3)\n", encoding="utf-8")
    assert run_python_file(workspace, "fail.py") == "Process exited with code 3\nSTDERR:\noops\n"


def test_reports_a_script_without_output(workspace):
    (workspace.root / "quiet.py").write_text("x = 1\n", encoding="utf-8")
    assert run_python_file(workspace, "quiet.py") == "No output produced"


def test_a_script_that_runs_too_long_is_stopped(workspace):
    (workspace.root / "loop.py").write_text("while True:\n    pass\n", encoding="utf-8")
    with pytest.raises(ToolError, match="timed out after 0.5 seconds"):
        run_python_file(workspace, "loop.py", timeout=0.5)


def test_long_output_is_cut_off(workspace):
    (workspace.root / "noisy.py").write_text(
        f"print('y' * {tools.MAX_OUTPUT_CHARS * 2})\n", encoding="utf-8")
    output = run_python_file(workspace, "noisy.py")
    assert output.endswith(f"[...output truncated at {tools.MAX_OUTPUT_CHARS} characters]")
    assert len(output) < tools.MAX_OUTPUT_CHARS + 100


def test_api_keys_are_not_passed_to_the_script(workspace, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-that-must-not-leak")
    monkeypatch.setenv("OPENROUTER_API_KEY", "another-key")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    (workspace.root / "env.py").write_text(
        "import os\n"
        "print(os.environ.get('ANTHROPIC_API_KEY'), os.environ.get('OPENROUTER_API_KEY'),"
        " os.environ.get('HARMLESS_SETTING'))\n", encoding="utf-8")
    assert run_python_file(workspace, "env.py") == "STDOUT:\nNone None visible\n"


def test_an_edit_is_picked_up_by_the_very_next_run(workspace):
    """Editing a module and re-running at once must execute the new code.

    The two versions have the same length and are written within the same
    second - exactly the case in which Python would reuse a stale .pyc file.
    """
    (workspace.root / "app.py").write_text("from pkg.mod import VALUE\nprint(VALUE)\n", encoding="utf-8")
    assert run_python_file(workspace, "app.py") == "STDOUT:\n1\n"
    write_file(workspace, "pkg/mod.py", "VALUE = 2\n")
    assert run_python_file(workspace, "app.py") == "STDOUT:\n2\n"
    # No bytecode is left behind in the project either.
    assert not list(workspace.root.rglob("__pycache__"))


def test_only_python_files_inside_the_workspace_can_be_run(workspace):
    (workspace.root / "notes.txt").write_text("print('no')\n", encoding="utf-8")
    with pytest.raises(ToolError, match="is not a Python file"):
        run_python_file(workspace, "notes.txt")
    with pytest.raises(ToolError, match="does not exist"):
        run_python_file(workspace, "missing.py")
    with pytest.raises(ToolError, match="does not exist"):
        run_python_file(workspace, "pkg")
    (workspace.root.parent / "outside.py").write_text("print('escaped')\n", encoding="utf-8")
    with pytest.raises(WorkspaceError):
        run_python_file(workspace, "../outside.py")
