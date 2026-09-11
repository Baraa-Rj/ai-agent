import os
import subprocess


def run_python_file(
    working_directory: str, file_path: str, args: list[str] | None = None
) -> str:

    working_dir_abs = os.path.abspath(working_directory)
    file_abspath = os.path.abspath(os.path.join(working_dir_abs, file_path))

    if os.path.commonpath([working_dir_abs, file_abspath]) != working_dir_abs:
        return (
            f'Error: Cannot execute "{file_path}" '
            f"as it is outside the permitted working directory"
        )

    if not os.path.isfile(file_abspath):
        return f'Error: "{file_path}" does not exist or is not a regular file'

    if not file_abspath.endswith(".py"):
        return f'Error: "{file_path}" is not a Python file'

    try:
        command = ["python", file_abspath]

        if args:
            command.extend(args)

        result = subprocess.run(
            command,
            cwd=working_dir_abs,
            text=True,
            timeout=30,
            capture_output=True,
        )

        string_result = ""
        if result.returncode != 0:
            string_result += f"Process exited with code {result.returncode}\n"

        if not result.stdout and not result.stderr:
            string_result += "No output produced"
        else:
            if result.stdout:
                string_result += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                string_result += f"STDERR:\n{result.stderr}\n"

        return string_result
    except Exception as e:
        return f"Error: executing Python file: {e}"


schema_run_python_file = {
    "type": "function",
    "function": {
        "name": "run_python_file",
        "description": (
            "Executes a Python file in the working directory and returns its output"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to the Python file to execute, relative to the working directory"
                    ),
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of arguments to pass to the Python script"
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
}


def run_python_file_schema() -> dict:
    return schema_run_python_file
