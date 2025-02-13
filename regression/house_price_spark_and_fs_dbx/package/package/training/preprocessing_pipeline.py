from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_pipeline(
    numerical_features: List[str], categorical_features: List[str]
) -> Pipeline:
    """
    Get sklearn pipeline.

    :param numerical_features: List of numerical features.
    :param categorical_features: List of categorical features.
    :return: Sklearn pipeline.
    """
    transformer = ColumnTransformer(
        [
            (
                "numerical_imputer",
                SimpleImputer(strategy="median"),
                numerical_features,
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        [("transformer", transformer), ("regressor", RandomForestRegressor())]
    )

    return pipeline
