import pkgutil
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
import yaml
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from wine_quality_v1.utils.utils import read_config


def get_wine_dataset_uci() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Get wine dataset from UCI.

    :return: wine dataset and feature names.
    """
    wine_quality = fetch_ucirepo(id=186)
    categorical_features = wine_quality.variables[
        (wine_quality.variables["type"] == "Categorical")
        & (wine_quality.variables["role"] == "Feature")
    ]["name"].values.tolist()
    numerical_features = wine_quality.variables[
        (wine_quality.variables["type"] == "Continuous")
        & (wine_quality.variables["role"] == "Feature")
    ]["name"].values.tolist()
    target = wine_quality.variables[(wine_quality.variables["role"] == "Target")][
        "name"
    ].values[0]
    metadata = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target": target,
    }

    df = wine_quality.data.features
    df[target] = wine_quality.data.targets
    return df, metadata


def get_wine_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """
    Get wine dataset from sklearn.

    :return: wine dataset and feature names.
    """

    wine = load_wine()
    feature_names = wine.feature_names
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    return df, feature_names
