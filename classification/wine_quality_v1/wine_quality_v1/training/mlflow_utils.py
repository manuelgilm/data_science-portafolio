import mlflow


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an experiment in MLflow.

    :param experiment_name: name of the experiment.
    :return: experiment id.
    """
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(
            experiment_name
        ).experiment_id
    finally:
        mlflow.set_experiment(experiment_name)
    return experiment_id
