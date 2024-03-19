from typing import List

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_pipeline(
    numerical_columns: List[str], regressor: BaseEstimator
) -> Pipeline:
    """
    Get the pipeline for the model.

    :param numerical_columns: List of numerical columns
    :param regressor: Regressor
    :return: sklearn pipeline
    """

    transformers = ColumnTransformer(
        [
            (
                "numerical_processor",
                SimpleImputer(strategy="median"),
                numerical_columns,
            )
        ]
    )

    pipeline = Pipeline([("preprocessor", transformers), ("model", regressor)])

    return pipeline
