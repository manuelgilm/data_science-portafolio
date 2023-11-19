from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_train_test_data(n_features) -> Tuple[pd.DataFrame]:
    """
    Get  train and test data for model training.

    :return: x_train, x_test, y_train, y_test
    """
    x, y = make_regression(
        n_samples=1000, n_features=n_features, n_targets=1, random_state=42
    )
    x = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.DataFrame(y, columns=["target"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=np.random.randint(0, 100)
    )
    return x_train, x_test, y_train, y_test


def get_processing_pipeline(numerical_features: list) -> Tuple[pd.DataFrame]:
    """
    Get processing pipeline for model training.

    :param numerical_features: list of numerical features

    :return: processing_pipeline
    """

    column_transformer = ColumnTransformer(
        [("numerical_imputer", SimpleImputer(strategy="median"), numerical_features)]
    )
    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", RandomForestRegressor(random_state=42)),
        ]
    )

    return pipeline


def get_regression_metrics(
    estimator: Pipeline, x: pd.DataFrame, y: pd.DataFrame, prefix: str
):
    """
    Get regression metrics for model evaluation.

    :param estimator: model to evaluate
    :param x: features
    :param y: target
    :param prefix: prefix for metric names

    :return: metrics
    """
    y_pred = estimator.predict(x)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    metrics = {
        f"{prefix}_r2": r2,
        f"{prefix}_mse": mse,
        f"{prefix}_mae": mae,
    }

    return metrics
