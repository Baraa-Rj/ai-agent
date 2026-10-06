"""The tools the model can call, and the registry that validates and runs them."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agent.types import ToolCall, ToolResult, ToolSpec
from agent.workspace import Workspace, WorkspaceError

MAX_FILE_CHARS = 10_000
MAX_OUTPUT_CHARS = 10_000
PYTHON_TIMEOUT_SECONDS = 30
# Variables whose names end like this are not passed to executed scripts.
SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


class ToolError(Exception):
    """A tool could not do what was asked. The message is sent back to the model."""


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: Callable[..., str]  # handler(workspace, **arguments) -> text for the model


# --------------------------------------------------------------------------- tools


def get_files_info(workspace: Workspace, directory: str = ".") -> str:
    target = workspace.resolve(directory)
    if not target.is_dir():
        raise ToolError(f'"{directory}" is not a directory')
    # lstat: report a symbolic link itself instead of failing on a broken one.
    lines = [
        f"- {entry.name}: file_size={entry.lstat().st_size} bytes, is_dir={entry.is_dir()}"
        for entry in sorted(target.iterdir(), key=lambda entry: entry.name)
    ]
    return "\n".join(lines) or "(empty directory)"


def get_file_content(workspace: Workspace, file_path: str) -> str:
    target = workspace.resolve(file_path)
    if not target.is_file():
        raise ToolError(f'"{file_path}" is not a file')
    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read(MAX_FILE_CHARS)
            truncated = bool(f.read(1))
    except UnicodeDecodeError as exc:
        raise ToolError(f'"{file_path}" is not UTF-8 text') from exc
    if truncated:
        content += f'[...File "{file_path}" truncated at {MAX_FILE_CHARS} characters]'
    return content or "(empty file)"


def write_file(workspace: Workspace, file_path: str, content: str) -> str:
    target = workspace.resolve(file_path)
    if target.is_dir():
        raise ToolError(f'"{file_path}" is a directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return f'Successfully wrote to "{file_path}" ({len(content)} characters written)'


def run_python_file(
    workspace: Workspace,
    file_path: str,
    args: Sequence[str] | None = None,
    *,
    timeout: float = PYTHON_TIMEOUT_SECONDS,
) -> str:
    target = workspace.resolve(file_path)
    if not target.is_file():
        raise ToolError(f'"{file_path}" does not exist or is not a regular file')
    if target.suffix != ".py":
        raise ToolError(f'"{file_path}" is not a Python file')

    # The script runs as an ordinary child process: this is not a sandbox.
    # API keys and similar variables are kept out of its environment.
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().endswith(SECRET_ENV_SUFFIXES)
    }
    # Run without bytecode caches. A cached .pyc is trusted when the source's
    # size and modification time (in whole seconds) are unchanged, so an edit
    # of the same length followed at once by a run could execute the old code.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        with tempfile.TemporaryDirectory() as empty_cache:
            environment["PYTHONPYCACHEPREFIX"] = empty_cache
            completed = subprocess.run(
                [sys.executable, str(target), *(args or [])],
                cwd=workspace.root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # a non-zero exit code is reported to the model, not raised
            )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f'"{file_path}" timed out after {timeout:g} seconds') from exc

    parts = []
    if completed.returncode != 0:
        parts.append(f"Process exited with code {completed.returncode}")
    if completed.stdout:
        parts.append(f"STDOUT:\n{completed.stdout}")
    if completed.stderr:
        parts.append(f"STDERR:\n{completed.stderr}")
    if not completed.stdout and not completed.stderr:
        parts.append("No output produced")
    output = "\n".join(parts)
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n[...output truncated at {MAX_OUTPUT_CHARS} characters]"
    return output


# ---------------------------------------------------------------- tool definitions

GET_FILES_INFO = Tool(
    ToolSpec(
        name="get_files_info",
        description=(
            "Lists the entries of a directory inside the working directory, with each "
            "entry's size in bytes and whether it is a directory. Use it to explore the "
            "project before reading or changing files. It does not recurse into "
            "subdirectories and does not return file contents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "Directory to list, relative to the working directory. "
                        "Defaults to the working directory itself."
                    ),
                },
            },
        },
    ),
    get_files_info,
)

GET_FILE_CONTENT = Tool(
    ToolSpec(
        name="get_file_content",
        description=(
            "Reads a UTF-8 text file inside the working directory and returns its "
            f"content. Files longer than {MAX_FILE_CHARS} characters are cut off at that "
            "point and the result says so. Use it before editing a file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to read, relative to the working directory.",
                },
            },
            "required": ["file_path"],
        },
    ),
    get_file_content,
)

WRITE_FILE = Tool(
    ToolSpec(
        name="write_file",
        description=(
            "Writes text to a file inside the working directory, creating the file and "
            "any missing parent directories, and replacing the file's entire previous "
            "content. Pass the complete new content of the file, not a fragment."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to write, relative to the working directory.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        },
    ),
    write_file,
)

RUN_PYTHON_FILE = Tool(
    ToolSpec(
        name="run_python_file",
        description=(
            "Runs a .py file from the working directory with the Python interpreter and "
            "returns its exit code (when it is not zero), standard output and standard "
            f"error. The run is stopped after {PYTHON_TIMEOUT_SECONDS} seconds. Use it to "
            "run a program or its tests and check the result of a change."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the Python file to run, relative to the working directory.",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional command-line arguments passed to the script.",
                },
            },
            "required": ["file_path"],
        },
    ),
    run_python_file,
)


def build_tools(allow_exec: bool = True) -> list[Tool]:
    tools = [GET_FILES_INFO, GET_FILE_CONTENT, WRITE_FILE]
    if allow_exec:
        tools.append(RUN_PYTHON_FILE)
    return tools


# ------------------------------------------------------------------------ registry

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "array": list,
    "object": dict,
}


def validate_arguments(schema: dict[str, Any], arguments: Any) -> str | None:
    """Check `arguments` against the small part of JSON Schema the tools use.

    Returns a message describing the first problem, or None when they are valid.
    """
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"
    properties = schema.get("properties", {})
    unexpected = sorted(set(arguments) - set(properties))
    if unexpected:
        return f"unexpected argument(s): {', '.join(unexpected)}"
    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    for name, value in arguments.items():
        expected = properties[name].get("type")
        python_type = _JSON_TYPES.get(expected)
        if python_type is None:
            continue
        # bool is a subclass of int in Python, but not a JSON number.
        if not isinstance(value, python_type) or (expected != "boolean" and isinstance(value, bool)):
            return f'argument "{name}" must be of type {expected}'
        item_type = _JSON_TYPES.get(properties[name].get("items", {}).get("type"))
        if expected == "array" and item_type and not all(isinstance(item, item_type) for item in value):
            return f'argument "{name}" must be an array of {properties[name]["items"]["type"]}s'
    return None


class ToolRegistry:
    """Looks up, validates and runs tool calls against one workspace.

    Whatever goes wrong becomes an error result for the model to read. A tool
    call never raises into the agent loop.
    """

    def __init__(self, workspace: Workspace, tools: Sequence[Tool]) -> None:
        self._workspace = workspace
        self._tools = {tool.spec.name: tool for tool in tools}

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.error is not None:
            return self._error(call, call.error)
        tool = self._tools.get(call.name)
        if tool is None:
            return self._error(call, f"Unknown tool: {call.name}")
        problem = validate_arguments(tool.spec.parameters, call.arguments)
        if problem is not None:
            return self._error(call, problem)
        try:
            content = tool.handler(self._workspace, **call.arguments)
        except (ToolError, WorkspaceError) as exc:
            return self._error(call, str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must never take the agent loop down
            return self._error(call, f"{type(exc).__name__}: {exc}")
        return ToolResult(call.id, content)

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(call.id, f"Error: {message}", is_error=True)
