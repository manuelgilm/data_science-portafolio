import mlflow
from mlflow.entities import Experiment
from wine_quality_v1.utils.utils import get_root_dir

from typing import Dict
from typing import Any


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
        experiment_id = mlflow.create_experiment(experiment_name, tags)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name)
    return experiment
