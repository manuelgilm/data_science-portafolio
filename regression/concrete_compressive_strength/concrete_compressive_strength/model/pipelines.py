from typing import List

from sklearn.ensemble import RandomForestRegressor 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def get_pipeline(numerical_columns:List[str])->Pipeline:
    """
    Get the pipeline for the model.

    :param numerical_columns: List of numerical columns
    :return: sklearn pipeline
    """

    transformers = ColumnTransformer([
        ('numerical_processor', SimpleImputer(strategy='median'), numerical_columns)
    ])

    pipeline = Pipeline([
        ('preprocessor', transformers),
        ('model', RandomForestRegressor())
    ])

    return pipeline