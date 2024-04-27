from pathlib import Path


def get_root_dir() -> Path:
    """
    Get the root directory of the project.

    :return: root directory.
    """
    return Path(__file__).parent.parent.parent
