from functions.get_files_info import get_files_info_schema
from functions.get_file_content import get_file_content_schema
from functions.write_file import write_file_schema
from functions.run_python_file import run_python_file_schema

available_functions = [
    get_files_info_schema(),
    get_file_content_schema(),
    write_file_schema(),
    run_python_file_schema(),
]


def get_available_functions() -> list:
    return available_functions
 