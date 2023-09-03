import mlflow

from mlflow.models import infer_signature
from mlflow.models import ModelSignature
from mlflow.types import ParamSchema
from mlflow.types import ParamSpec

import pandas as pd


def create_or_set_experiment(experiment_name: str) -> str:
    """
    Create  or set experiment in MLflow.

    :param experiment_name: name of the experiment

    :return: experiment_id
    """
    client = mlflow.tracking.MlflowClient()
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id

    return experiment_id


def get_custom_signature(
    x: pd.DataFrame, y: pd.DataFrame, params: dict
) -> ModelSignature:
    """
    Get signature for custom model.

    :param x: features
    :param y: target
    :param params: model parameters

    :return: signature
    """
    data_schema = infer_signature(x, y)

    type_map = {"str": "string", "int": "integer", "float": "float", "bool": "boolean"}

    params_spec = [
        ParamSpec(name=k, dtype=type_map[type(v).__name__], default=None)
        for k, v in params.items()
    ]
    params_schema = ParamSchema(params_spec)
    signature = ModelSignature(
        inputs=data_schema.inputs, outputs=data_schema.outputs, params=params_schema
    )

    return signature
