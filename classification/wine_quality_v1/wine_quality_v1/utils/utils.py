from pathlib import Path
from typing import Any
from typing import Dict

import yaml
import pkgutil
import pickle
import pandas as pd 
def get_root_dir() -> Path:
    """
    Get the root directory of the project.

    :return: root directory.
    """
    return Path(__file__).parent.parent.parent


def read_config(path: str = "configuration") -> Dict[str, Any]:
    """
    Read a yaml config file.

    :param path: path to the config file.
    :return: config.
    """
    data_bin = pkgutil.get_data("wine_quality_v1.configs", path+".yaml")
    config = yaml.safe_load(data_bin)
    return config

def read_pickle(path: str) -> Any:
    """
    Read a pickle file.

    :param path: path to the pickle file.
    :return: data.
    """

    root_dir = get_root_dir()
    path = root_dir / path
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    
    with open(str(path), "rb") as f:
        data = pickle.load(f)

    return data

def read_set(name: str)-> pd.DataFrame:
    """
    Read a csv file.

    :param name: name of the set.
    :return: data.
    """
    root_dir = get_root_dir()
    
    config = read_config("data_preparation")
    path = root_dir / config["output_folder"] / config["sets"][name]
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    
    data = pd.read_csv(path)
    return data