import mlflow
from mlflow.entities import Experiment
from wine_quality_v1.utils.utils import get_root_dir
from wine_quality_v1.utils.utils import read_config

from typing import Dict
from typing import Any
from typing import Optional


def get_or_create_experiment(experiment_name: str, tags: Dict[str, Any]) -> Experiment:
    """
    Get or create an experiment in MLflow.

    :param experiment_name: name of the experiment.
    :param tags: tags to add to the experiment.
    :return: Experiment
    """

    root_dir = get_root_dir()
    mlflow.set_tracking_uri((root_dir / "mlruns").as_uri())

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(name=experiment_name, tags=tags)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name)
    return experiment


def get_model_uri(stage: Optional[str] = "prod"):
    """
    Get the model from the MLflow registry.

    :param stage: The stage of the model.
    :return: The model.
    """
    config = read_config()
    name = config["registered_model_name"]
    alias = config["aliases"][stage]
    model_uri = f"models:/{name}@{alias}"
    return model_uri


def get_model(stage: Optional[str] = "prod"):
    """
    Get the model from the MLflow registry.

    :param stage: The stage of the model.
    :return: The model.
    """
    model_uri = get_model_uri(stage)
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except mlflow.exceptions.MlflowException as e:
        raise e

    return model
