import mlflow


def get_or_create_experiment(
    experiment_name: str, artifact_location: str = "artifacts"
) -> str:
    """
    Get or create experiment in mlflow

    :param experiment_name: Name of the experiment.
    :param artifact_location: Location of the artifacts.
    :return: Id of the experiment.
    """

    try:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    finally:
        mlflow.set_experiment(experiment_name)
    return experiment_id
