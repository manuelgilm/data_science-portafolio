from typing import Any
from typing import Dict

import mlflow
import numpy as np
import pandas as pd


def get_predictions(
    run_id: str, x_test: pd.DataFrame, config: Dict[str, Any]
) -> np.ndarray:
    """
    Get predictions from the iris classifier.

    :param run_id: The run id of the model to use for predictions.
    :param x_test: The test data to use for predictions.
    :param config: The configuration for the model.
    :return: The predictions from the iris classifier.
    """

    model_uri = f"runs:/{run_id}/{config['model_artifact']}"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    predictions = loaded_model.predict(x_test)
    return predictions


def get_latest_run_id(experiment_name: str) -> str:
    """
    Get the latest run id from the experiment.

    :param experiment_name: The name of the experiment.
    :return: The latest run id from the experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    latest_run = mlflow.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["start_time desc"],
        max_results=1,
    )
    return latest_run.iloc[0]["run_id"]
