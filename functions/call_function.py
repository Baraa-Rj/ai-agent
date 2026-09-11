from functions.get_files_info import get_files_info_schema

available_functions = [
    get_files_info_schema(),
]


def get_available_functions() -> list:
    return available_functions
 