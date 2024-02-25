import pkgutil
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """
    Get the root of the project.
    """
    return Path(__file__).parent.parent.parent


def read_config(name: str):
    """
    Read the configuration file.

    :param name: The name of the configuration file.
    """
    data = pkgutil.get_data("credit_approval", f"configs/{name}.yaml")
    config = yaml.safe_load(data)
    return config
