import mlflow 


def set_or_create_experiment(experiment_name:str)->str:
    """
    Get or create an experiment.

    :param experiment_name: Name of the experiment. 
    :return: Experiment ID.
    """

    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id

