import sys
from pathlib import Path
def check_file_exists(file_path):
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: The file {file_path} does not exist.")
        sys.exit(1)

def ensure_dir_exists(dir_path):
    path = Path(dir_path)
    if not path.exists():
        print(f"Creating directory: {dir_path}")
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        print(f"Error: {dir_path} exists but is not a directory.")
        sys.exit(1)