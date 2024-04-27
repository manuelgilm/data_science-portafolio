import pkgutil
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
import yaml
from sklearn.datasets import load_wine
from ucimlrepo import fetch_ucirepo


def get_wine_dataset_uci() -> Tuple[pd.DataFrame, List[str]]:
    """
    Get wine dataset from UCI.

    :return: wine dataset and feature names.
    """
    wine_quality = fetch_ucirepo(id=186)
    df = wine_quality.data.features
    feature_names = df.columns.tolist()
    df["quality"] = wine_quality.data.targets

    return df, feature_names


def get_wine_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """
    Get wine dataset from sklearn.

    :return: wine dataset and feature names.
    """

    wine = load_wine()
    feature_names = wine.feature_names
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    print(read_config())
    return df, feature_names


def read_config(path: str = "configuration.yaml") -> Dict[str, Any]:
    """
    Read a yaml config file.

    :param path: path to the config file.
    :return: config.
    """
    data_bin = pkgutil.get_data("wine_quality_v1.configs", path)
    config = yaml.safe_load(data_bin)
    return config
