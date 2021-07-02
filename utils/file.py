from pathlib import Path

def file_exists(path: str) -> bool:
    return Path(path).is_file()

