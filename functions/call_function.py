import json

from collections.abc import Callable

from functions.get_files_info import get_files_info, get_files_info_schema
from functions.get_file_content import get_file_content, get_file_content_schema
from functions.write_file import write_file, write_file_schema
from functions.run_python_file import run_python_file, run_python_file_schema

available_functions = [
    get_files_info_schema(),
    get_file_content_schema(),
    write_file_schema(),
    run_python_file_schema(),
]

function_map: dict[str, Callable[..., str]] = {
    "get_files_info": get_files_info,
    "get_file_content": get_file_content,
    "write_file": write_file,
    "run_python_file": run_python_file,
}


def get_available_functions() -> list:
    return available_functions


def call_function(tool_call, verbose: bool = False) -> dict:
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")

    if verbose:
        print(f" - Calling function: {function_name}({function_args})")
    else:
        print(f" - Calling function: {function_name}")

    if function_name not in function_map:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Error: Unknown function: {function_name}",
        }

    function_args["working_directory"] = "./calculator"
    result = function_map[function_name](**function_args)

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }
