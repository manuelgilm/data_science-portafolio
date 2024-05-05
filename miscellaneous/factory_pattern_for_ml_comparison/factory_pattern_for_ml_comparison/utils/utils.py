from pathlib import Path
import mlflow
from mlflow.entities import Experiment
from typing import Any
from typing import Dict
from typing import Optional


def get_project_root() -> Path:
    """
    Get the root directory of the project
    """
    return Path(__file__).parent.parent.parent


def get_or_create_experiment(
    name: str, tags: Optional[Dict[str, Any]] = {}
) -> Experiment:
    """
    Get or create an MLflow experiment

    :param name: Name of the experiment
    :param tags: Tags to associate with the experiment

    """
    root_dir = get_project_root()
    tracking_uri = root_dir / "mlruns"

    mlflow.set_tracking_uri(tracking_uri.as_uri())
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(name, tags=tags)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment.name)
    return experiment
