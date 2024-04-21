import pkgutil
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional


def get_project_root() -> Path:
    """
    Project root directory.

    :return: Path
    """
    return Path(__file__).parent.parent.parent


def read_config(
    name: str, package_name: Optional[str] = "bike_sharing"
) -> Dict[str, Any]:
    """
    Read configuration file.

    :param path: Path to the configuration file
    :return: Dictionary of configuration
    """
    try:
        data = pkgutil.get_data(package_name, "configs/" + name + ".yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {name} not found")

    return data
