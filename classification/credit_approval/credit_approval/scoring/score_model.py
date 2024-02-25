from typing import Union

import mlflow
import numpy as np
import pandas as pd
from credit_approval.data.featurization import get_train_test_sets
from credit_approval.data.retrieval import get_data_info
from credit_approval.model.model_utils import get_latest_run_id
from credit_approval.utils.utils import read_config


def score_model(model_uri: str, data: pd.DataFrame):
    """
    Score the model on the given data.

    :param model_uri: The URI of the model to score.
    :param data: The data to score the model on.
    :return: A dictionary of metrics.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(data)
    return y_pred


def get_predictions() -> Union[pd.DataFrame, np.ndarray]:
    """
    Get the predictions for the model.
    """
    config = read_config("configs")
    experiment_name = config["experiment_name"]
    info = get_data_info()
    target = info[info["role"] == "Target"]["name"].values[0]
    _, x_test, _, _ = get_train_test_sets(target=target)
    run_id = get_latest_run_id(experiment_name)
    if run_id is None:
        raise ValueError("No runs found for the given experiment name.")
    model_uri = "runs:/{}/model".format(run_id)
    get_train_test_sets(target=target)
    return score_model(model_uri, x_test)
