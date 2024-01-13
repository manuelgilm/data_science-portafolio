from typing import Dict 
from typing import Any
import pkgutil
import yaml

def get_config(path:str)->Dict[str, Any]:
    """
    Read configuration file.

    :param path: Path to the configuration file.
    :return: Configuration file.
    """

    resource = pkgutil.get_data(__name__, path)
    if resource is not None:
        config = yaml.safe_load(resource)
        return config
    else:
        raise FileNotFoundError(f"Config file {path} not found.")