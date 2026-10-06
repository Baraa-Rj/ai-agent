"""Tool dispatch and argument validation."""
import re

import pytest

from agent.tools import Tool, ToolRegistry, build_tools, validate_arguments
from agent.types import ToolCall, ToolSpec
from tests.helpers import call


def test_dispatches_a_call_to_the_named_tool(registry):
    result = registry.execute(call("get_file_content", "call_7", file_path="pkg/mod.py"))
    assert (result.call_id, result.content, result.is_error) == ("call_7", "VALUE = 1\n", False)


def test_a_tool_may_be_called_with_only_its_required_arguments(registry):
    result = registry.execute(call("get_files_info"))
    assert not result.is_error and "- main.py:" in result.content


def test_an_unknown_tool_is_an_error_result_not_an_exception(registry):
    result = registry.execute(call("delete_everything", path="/"))
    assert result.is_error and result.content == "Error: Unknown tool: delete_everything"


def test_arguments_the_provider_could_not_decode_are_reported_back(registry, workspace):
    broken = ToolCall("call_1", "write_file", None, error="arguments were not valid JSON (Expecting value)")
    result = registry.execute(broken)
    assert result.is_error
    assert result.content == "Error: arguments were not valid JSON (Expecting value)"
    assert sorted(p.name for p in workspace.root.iterdir()) == ["main.py", "pkg"]


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({}, 'missing required argument(s): file_path, content'),
        ({"file_path": "a.txt"}, "missing required argument(s): content"),
        ({"file_path": "a.txt", "content": "x", "mode": "a"}, "unexpected argument(s): mode"),
        ({"file_path": "a.txt", "content": "x", "working_directory": "/"}, "unexpected argument(s): working_directory"),
        ({"file_path": 42, "content": "x"}, 'argument "file_path" must be of type string'),
        ({"file_path": "a.txt", "content": None}, 'argument "content" must be of type string'),
    ],
)
def test_invalid_arguments_are_rejected_before_the_tool_runs(registry, workspace, arguments, message):
    result = registry.execute(ToolCall("call_1", "write_file", arguments))
    assert result.is_error and result.content == f"Error: {message}"
    assert not (workspace.root / "a.txt").exists()


@pytest.mark.parametrize("args", ["--flag", [1, 2], ["ok", None]])
def test_script_arguments_must_be_a_list_of_strings(registry, args):
    result = registry.execute(call("run_python_file", file_path="main.py", args=args))
    assert result.is_error and 'argument "args" must be' in result.content


def test_a_path_escape_becomes_an_error_result(registry, workspace):
    result = registry.execute(call("write_file", file_path="../escaped.txt", content="x"))
    assert result.is_error
    assert result.content == 'Error: "../escaped.txt" is outside the permitted working directory'
    assert not (workspace.root.parent / "escaped.txt").exists()


def test_a_tool_failure_becomes_an_error_result(registry):
    result = registry.execute(call("get_file_content", file_path="missing.py"))
    assert result.is_error and result.content == 'Error: "missing.py" is not a file'


def test_an_unexpected_exception_in_a_tool_does_not_escape(workspace):
    def explode(workspace):
        raise RuntimeError("disk on fire")

    registry = ToolRegistry(workspace, [Tool(ToolSpec("explode", "always fails", {"type": "object", "properties": {}}), explode)])
    result = registry.execute(call("explode"))
    assert result.is_error and result.content == "Error: RuntimeError: disk on fire"


def test_a_failing_script_is_a_normal_result_not_a_tool_error(registry, workspace):
    (workspace.root / "fail.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    result = registry.execute(call("run_python_file", file_path="fail.py"))
    assert not result.is_error and result.content.startswith("Process exited with code 2")


def test_python_execution_can_be_left_out(workspace):
    assert [tool.spec.name for tool in build_tools()] == [
        "get_files_info", "get_file_content", "write_file", "run_python_file"]
    registry = ToolRegistry(workspace, build_tools(allow_exec=False))
    assert [spec.name for spec in registry.specs] == ["get_files_info", "get_file_content", "write_file"]
    result = registry.execute(call("run_python_file", file_path="main.py"))
    assert result.is_error and result.content == "Error: Unknown tool: run_python_file"


def test_tool_definitions_are_well_formed():
    for tool in build_tools():
        spec = tool.spec
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", spec.name)
        assert len(spec.description) > 80
        assert spec.parameters["type"] == "object"
        assert set(spec.parameters.get("required", [])) <= set(spec.parameters["properties"])


def test_validate_arguments_accepts_valid_input_and_rejects_non_objects():
    schema = build_tools()[3].spec.parameters
    assert validate_arguments(schema, {"file_path": "a.py", "args": ["1", "2"]}) is None
    assert validate_arguments(schema, {"file_path": "a.py"}) is None
    for not_an_object in (None, "main.py", ["main.py"], 3):
        assert validate_arguments(schema, not_an_object) == "arguments must be a JSON object"
