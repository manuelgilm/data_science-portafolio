from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_pipeline(
    numerical_features: List[str], categorical_features: List[str]
) -> Pipeline:
    """
    Get the pipeline for the model.

    :param numerical_features: List of numerical features
    :param categorical_features: List of categorical features
    :return: Pipeline
    """
    # Preprocessing for numerical data
    numerical_imputer = SimpleImputer(strategy="median")

    # Preprocessing for categorical data
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    transformer = ColumnTransformer(
        [
            ("numerical_imputer", numerical_imputer, numerical_features),
            ("categorical_imputer", categorical_imputer, categorical_features),
        ]
    )

    model = RandomForestRegressor()
    pipeline = Pipeline([("preprocessor", transformer), ("model", model)])
    return pipeline


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers.

    :param data: Input data
    :return: Preprocessed data
    """
    q1 = data.cnt.quantile(0.25)
    q3 = data.cnt.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    data_preprocessed = data.loc[
        (data.cnt >= lower_bound) & (data.cnt <= upper_bound)
    ]

    return data_preprocessed
