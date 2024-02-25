from typing import Dict
from typing import Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from credit_approval.utils.utils import get_project_root
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def get_classification_metrics(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    prefix: str,
) -> Dict[str, float]:
    """
    Get the classification metrics.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: The classification metrics.
    """
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
    }


def get_performance_plots(
    y_pred, y_true, prefix: str
) -> Dict[str, plt.figure]:
    """
    Get the performance plots for the model
    :param y_pred: Predicted labels
    :param y_true: True labels
    :param prefix: Prefix for the plot names
    :return: None
    """
    # ROC curve
    roc_display = RocCurveDisplay.from_predictions(y_true, y_pred)

    # Precision-recall curve
    pr_display = PrecisionRecallDisplay.from_predictions(y_true, y_pred)

    # Confusion matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    return {
        f"{prefix}_roc_curve.png": roc_display.figure_,
        f"{prefix}_precision_recall_curve.png": pr_display.figure_,
        f"{prefix}_confusion_matrix.png": cm_display.figure_,
    }


def create_experiment(experiment_name: str) -> str:
    """
    Create an experiment in MLflow
    :param experiment_name: Name of the experiment
    :return: Experiment ID
    """
    artifact_path = get_project_root() / "mlruns"
    mlflow.set_tracking_uri(artifact_path.as_uri())

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        print(e)
        experiment_id = mlflow.get_experiment_by_name(
            experiment_name
        ).experiment_id
    finally:
        mlflow.set_experiment(experiment_name)

    return experiment_id


def get_latest_run_id(experiment_name: str) -> str:
    """
    Get the latest run ID for the given experiment
    :param experiment_name: Name of the experiment
    :return: Run ID
    """
    artifact_path = get_project_root() / "mlruns"
    mlflow.set_tracking_uri(artifact_path.as_uri())

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError("No experiment found with the given name.")
    try:
        run = mlflow.search_runs(
            experiment_ids=experiment.experiment_id,
            order_by=["start_time desc"],
        ).iloc[0]
    except IndexError as e:
        print(e)
        return None

    return run.run_id
