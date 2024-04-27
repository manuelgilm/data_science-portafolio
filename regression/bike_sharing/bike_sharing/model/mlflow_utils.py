from typing import Dict
from typing import Optional

import mlflow
import pandas as pd
from bike_sharing.utils.utils import get_project_root
from bike_sharing.utils.utils import read_config
from mlflow.entities.experiment import Experiment


def get_or_create_experiment(
    experiment_name: str, tags: Optional[Dict[str, str]] = None
) -> Experiment:
    """
    Get or create an experiment.

    :param experiment_name: The name of the experiment.
    :param tags: A dictionary of tags to add to the experiment.
    :return: MLflow Experiment
    """

    # Get the root project directory
    project_dir = get_project_root()

    # set the tracking uri
    tracking_uri = (project_dir / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # If the experiment does not exist, create it
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, tags=tags)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment


def evaluate_regressor(
    run_id: str,
    eval_data: pd.DataFrame,
    label: str,
    model_uri: str,
    baseline_model_uri: Optional[str] = None,
):
    """
    Evaluate the model.

    :param eval_data: Data to evaluate the model on
    :param label: Target variable
    :return: Evaluation results
    """

    if mlflow.active_run():
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets=label,
            model_type="regressor",
            evaluators=["default"],
            baseline_model=baseline_model_uri,
        )
    else:

        with mlflow.start_run(run_id=run_id):
            result = mlflow.evaluate(
                model=model_uri,
                data=eval_data,
                targets=label,
                model_type="regressor",
                evaluators=["default"],
                baseline_model=baseline_model_uri,
            )

    return result


def get_model_uri(stage: Optional[str] = "production"):
    """
    Get the model from the MLflow registry.

    :param stage: The stage of the model.
    :return: The model.
    """
    config = read_config("training")
    name = config["mlflow"]["registered_model_name"]
    alias = config["mlflow"]["aliases"][stage]
    model_uri = f"models:/{name}@{alias}"
    return model_uri


def get_model(stage: Optional[str] = "production"):
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
