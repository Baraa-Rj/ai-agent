import os
def write_file(working_directory: str, file_path: str, content: str) -> str:
    working_dir_abs = os.path.abspath(working_directory)
    file_abspath = os.path.abspath(os.path.join(working_dir_abs, file_path))

    if os.path.commonpath([working_dir_abs, file_abspath]) != working_dir_abs:
        return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'

    try:
       os.makedirs(os.path.dirname(file_abspath), exist_ok=True)
       with open(file_abspath, "w") as f:
           f.write(content)
       return f'Successfully wrote to "{file_path}" ({len(content)} characters written)'
 
    except Exception as e:
        return f"Error: {str(e)}"