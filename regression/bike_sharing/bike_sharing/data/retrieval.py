from typing import Dict
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def get_dataset() -> Tuple[pd.DataFrame, Dict]:
    """
    Get the dataset from UCI ML Repository.

    :return: Tuple of data and metadata
    """

    # fetch dataset
    bike_sharing = fetch_ucirepo(id=275)

    # data (as pandas dataframes)
    X = bike_sharing.data.features
    y = bike_sharing.data.targets
    df = X.copy()

    categorical_features = bike_sharing.variables[
        (bike_sharing.variables["type"] == "Categorical")
        & (bike_sharing.variables["role"] == "Feature")
    ]["name"].tolist()
    numerical_features = bike_sharing.variables[
        (bike_sharing.variables["type"] == "Continuous")
        & (bike_sharing.variables["role"] == "Feature")
    ]["name"].tolist()
    binary_features = bike_sharing.variables[
        (bike_sharing.variables["type"] == "Binary")
        & (bike_sharing.variables["role"] == "Feature")
    ]["name"].tolist()
    target = bike_sharing.variables[
        bike_sharing.variables["role"] == "Target"
    ]["name"].tolist()[0]
    df[target] = y

    metadata = {
        "features": {
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "binary_features": binary_features,
        },
        "target": target,
    }
    return df, metadata


def get_train_test_data(
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Get the train and test data.

    :param feature_names: List of feature names
    :param target: Target variable name
    :param test_size: Test size
    :return: Tuple of train and test data
    """
    df, metadata = get_dataset()
    feature_names = [
        feature_name
        for feature_type in metadata["features"].keys()
        for feature_name in metadata["features"][feature_type]
    ]
    target = metadata["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_names], df[target], test_size=test_size, random_state=42
    )

    # remove outliers for training data
    x_train[target] = y_train
    x_train = remove_outliers(x_train)
    y_train = x_train[target]
    x_train.drop(columns=[target], inplace=True)

    # remove outliers for test data
    x_test[target] = y_test
    x_test = remove_outliers(x_test)
    y_test = x_test[target]
    x_test.drop(columns=[target], inplace=True)

    x_train[metadata["features"]["categorical_features"]] = x_train[
        metadata["features"]["categorical_features"]
    ].astype("int")
    x_test[metadata["features"]["categorical_features"]] = x_test[
        metadata["features"]["categorical_features"]
    ].astype("int")

    return x_train, x_test, y_train, y_test, metadata


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers.

    :param data: Input data
    :return: Preprocessed data
    """
    data = data.copy()
    q1 = data.cnt.quantile(0.25)
    q3 = data.cnt.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    data_preprocessed = data.loc[
        (data.cnt >= lower_bound) & (data.cnt <= upper_bound)
    ]

    return data_preprocessed
