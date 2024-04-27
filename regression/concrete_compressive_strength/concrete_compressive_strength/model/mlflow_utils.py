from typing import Dict

import matplotlib.pyplot as plt
import mlflow
from concrete_compressive_strength.utils.utils import get_root_dir


def create_experiment(experiment_name: str) -> str:
    """
    Create an experiment in MLflow
    :param experiment_name: Name of the experiment
    :return: Experiment ID
    """
    artifact_path = get_root_dir() / "mlruns"
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


def log_figures(figures: Dict[str, plt.Figure]) -> None:
    """
    Log figures to MLflow.

    :param figures: Dictionary of figure names and figure objects.
    :return: None
    """
    for name, fig in figures.items():
        mlflow.log_figure(fig, name)
