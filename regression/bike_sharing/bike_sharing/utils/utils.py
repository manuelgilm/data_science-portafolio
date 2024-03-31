from pathlib import Path


def get_project_root() -> Path:
    """
    Project root directory.

    :return: Path
    """
    return Path(__file__).parent.parent.parent
