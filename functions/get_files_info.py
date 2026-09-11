import os


def get_files_info(working_directory: str, directory: str = ".") -> str:
    try:
        working_directory_abspath = os.path.abspath(working_directory)
        directory_abspath = os.path.normpath(
            os.path.join(working_directory_abspath, directory)
        )
        if (
            os.path.commonpath([working_directory_abspath, directory_abspath])
            != working_directory_abspath
        ):
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'
        if not os.path.isdir(directory_abspath):
            return f'Error: "{directory}" is not a directory'
        lines = []
        for entry in os.listdir(directory_abspath):
            complete_path = os.path.join(directory_abspath, entry)
            size = os.path.getsize(complete_path)
            lines.append(
                f"- {entry}: file_size={size} bytes, is_dir={os.path.isdir(complete_path)}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"
