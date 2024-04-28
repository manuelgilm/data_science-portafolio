from pathlib import Path
from typing import Any
from typing import Dict

import yaml
import pkgutil


def get_root_dir() -> Path:
    """
    Get the root directory of the project.

    :return: root directory.
    """
    return Path(__file__).parent.parent.parent


def read_config(path: str = "configuration.yaml") -> Dict[str, Any]:
    """
    Read a yaml config file.

    :param path: path to the config file.
    :return: config.
    """
    data_bin = pkgutil.get_data("wine_quality_v1.configs", path)
    config = yaml.safe_load(data_bin)
    return config
