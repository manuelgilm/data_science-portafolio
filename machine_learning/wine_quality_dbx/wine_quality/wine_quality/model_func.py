import mlflow

def create_mlflow_experiment(experiment_name: str) -> None:
    """
    This function creates a mlflow experiment.

    params: experiment_name: str

    """
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)