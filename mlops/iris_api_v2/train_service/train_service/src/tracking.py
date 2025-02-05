from typing import Dict
from typing import Optional

import mlflow
from mlflow.entities import Experiment


def get_or_create_experiment(
    name: str, tags: Optional[Dict[str, str]] = None
) -> Experiment:
    """
    Get or create an experiment in MLflow

    :param name: The name of the experiment
    :param tags: The tags to add to the experiment
    :return: The experiment object
    """
    # get the experiment by name
    experiment = mlflow.get_experiment_by_name(name)
    if not experiment:
        print(f"Experiment '{name}' not found")
        # create the experiment if it does not exist
        experiment = mlflow.create_experiment(name, tags)
        print(f"Experiment '{name}' created")
        print(f"Experiment ID: {experiment.experiment_id}")

    # set the experiment as the active experiment
    experiment = mlflow.set_experiment(experiment_name=name)

    return experiment

