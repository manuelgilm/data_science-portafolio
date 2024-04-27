import pkgutil
from typing import Dict

import yaml


def get_configs() -> Dict[str, any]:
    """
    Get configs.

    :return: Configs.
    """
    config = pkgutil.get_data(__name__, "config.yaml")
    return yaml.safe_load(config)
