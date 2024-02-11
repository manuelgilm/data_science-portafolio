from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from iris_classifier.utils.utils import get_project_root
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an experiment in MLflow.

    :param experiment_name: The name of the experiment to get or create.
    :return: The ID of the experiment.
    """
    artifact_path = get_project_root() / "mlruns"
    mlflow.set_tracking_uri(artifact_path.as_uri())
    try:
        experiment_id = mlflow.get_experiment_by_name(
            name=experiment_name
        ).experiment_id
    except Exception as e:
        print(e)
        experiment_id = mlflow.create_experiment(name=experiment_name)
    finally:
        mlflow.set_experiment(experiment_name)

    return experiment_id


def get_classification_metrics(
    y_pred: pd.DataFrame, y_test: pd.DataFrame, prefix: str
) -> Dict[str, float]:
    """
    Get the classification metrics for the estimator.

    :param y_pred: The predicted labels.
    :param y_test: The test labels.
    :return: A dictionary of classification metrics.
    """

    return {
        f"{prefix}_precision": precision_score(
            y_test, y_pred, average="weighted"
        ),
        f"{prefix}_recall": recall_score(y_test, y_pred, average="weighted"),
        f"{prefix}_f1": f1_score(y_test, y_pred, average="weighted"),
        f"{prefix}_accuracy": accuracy_score(y_test, y_pred),
    }


def get_confusion_matrix(
    estimator: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    prefix: str,
) -> Dict[str, plt.figure]:
    """
    Get the confusion matrix for the estimator.

    :param estimator: The estimator to get the figures for.
    :param x_test: The test data.
    :param y_test: The test labels.
    :return: A dictionary of figures.
    """

    return {
        f"{prefix}_confusion_matrix": ConfusionMatrixDisplay.from_estimator(
            estimator, x_test, y_test
        ).figure_,
    }
