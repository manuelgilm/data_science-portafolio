import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec
from mlflow.types import ParamSchema
from mlflow.types import ParamSpec
from mlflow.types import Schema


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

    input_schema = Schema(
        [ColSpec(type="double", name=col) for col in x.columns]
    )
    output_schema = Schema(
        [ColSpec(type="integer", name=col) for col in y.columns]
    )

    type_map = {
        "str": "string",
        "int": "integer",
        "float": "float",
        "bool": "boolean",
    }

    params_spec = [
        ParamSpec(name=k, dtype=type_map[type(v).__name__], default=None)
        for k, v in params.items()
    ]
    params_schema = ParamSchema(params_spec)
    signature = ModelSignature(
        inputs=input_schema, outputs=output_schema, params=params_schema
    )

    return signature
