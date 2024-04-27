from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_pipeline(
    numerical_features: List[str], categorical_features: List[str]
) -> Pipeline:
    """
    Returns a pipeline for the wine dataset.

    :param numerical_features: numerical features.
    :param categorical_features: categorical features.
    :return: pipeline.

    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier()),
        ]
    )

    return pipeline


def train(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    run_name: str,
):
    """
    Train a model.

    :param pipeline: sklearn pipeline.
    :param x_train: features.
    :param y_train: target.
    """
    pipeline.fit(x_train, y_train)

    return pipeline
