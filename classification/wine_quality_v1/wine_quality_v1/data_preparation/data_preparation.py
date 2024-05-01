import pkgutil
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
import yaml
from sklearn.datasets import load_wine
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


def replace_outlier(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Replace outliers with the median value.

    :param df: dataframe.
    :param feature: feature.
    :return: dataframe.
    """
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    median = df[feature].median()
    df.loc[(df[feature] < lower_bound) | (df[feature] > upper_bound), feature] = median
    return df


def remove_outlier(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove outliers from the dataset.

    :param df: dataframe.
    :param metadata: metadata.
    :return: dataframe.
    """
    for feature in metadata["numerical_features"]:
        df = replace_outlier(df, feature)
    return df


def transform_to_binary(
    df: pd.DataFrame, feature: str, threshold: int = 5
) -> pd.DataFrame:
    """
    Transform a feature to binary.

    :param df: dataframe.
    :param threshold: threshold.
    :param feature: feature.
    :return: dataframe.
    """
    df[feature] = df[feature].apply(lambda x: 1 if x > threshold else 0)
    return df
