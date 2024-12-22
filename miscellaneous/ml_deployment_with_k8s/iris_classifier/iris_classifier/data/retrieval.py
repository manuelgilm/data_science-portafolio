from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def get_dataset() -> pd.DataFrame:
    """
    Retrieve the iris dataset from sklearn.
    :return: The iris dataset as a pandas dataframe.
    """
    metadata = load_iris(as_frame=True)
    df = metadata.frame
    return df


def get_train_test_data(
    test_size: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get the train and test data from the iris dataset.

    :param test_size: The size of the test data.
    :return: The train and test data.
    """

    df = get_dataset()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
