from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from typing import Dict
from typing import Any
from typing import Optional

def get_pipeline(
    numerical_features: List[str], categorical_features: List[str], type: Optional[str] = "classification"
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
    model = (
        RandomForestClassifier()
        if type == "classification"
        else RandomForestRegressor()
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def train(
    params: Dict[str, Any],
    numerical_features: List[str],
    categorical_features: List[str],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
):
    """
    Train a model.

    :param params: parameters to train.
    :param numerical_features: numerical features.
    :param categorical_features: categorical features.
    :param x_train: features.
    :param y_train: target.
    """

    pipeline = get_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )
    # parse params to intenger
    params_ = {key: int(value) for key, value in params.items()}

    pipeline.set_params(**params_)
    pipeline.fit(x_train, y_train)

    return pipeline
