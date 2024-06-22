from pathlib import Path
from pkgutil import get_data

import yaml


def get_project_root() -> Path:
    """
    Get the root directory of the project.

    :return: The root directory of the project.
    """
    return Path(__file__).parent.parent.parent


def get_config(path: str = "config.yaml") -> dict:
    """
    Get the configuration from the config.yaml file.

    :param path: The path to the config.yaml file.
    :return: The configuration from the config.yaml file.
    """
    file = get_data("iris_classifier", "configs/" + path)
    config = yaml.safe_load(file)
    return config
