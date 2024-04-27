from pathlib import Path


def get_root_dir() -> Path:
    """
    Get the root directory of the project.

    :return: Path
    """
    return Path(__file__).resolve().parents[2]
