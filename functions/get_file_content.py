import os

MAX_CHARS = 10000


def get_file_content(working_directory: str, file_path: str) -> str:
    working_dir_abs = os.path.abspath(working_directory)
    file_abspath = os.path.abspath(os.path.join(working_dir_abs, file_path))

    if os.path.commonpath([working_dir_abs, file_abspath]) != working_dir_abs:
        return f'Error: Cannot read "{file_path}" as it is outside the permitted working directory'

    if not os.path.isfile(file_abspath):
        return f'Error: "{file_path}" is not a file'

    try:
        with open(file_abspath, "r") as f:
            content = f.read(MAX_CHARS)

            if f.read(1):
                content += f'[...File "{file_path}" truncated at {MAX_CHARS} characters]'

            return content

    except Exception as e:
        return f"Error: {str(e)}"
