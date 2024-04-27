import os

import mlflow
import pandas as pd


def read_source_data(config: dict, table_name: str, env: str = "local"):
    """
    Read source data from the data source
    :param config: config file
    :return: pandas dataframe
    """
    root_path = config["env"][env]["source"]
    filepath = os.path.join(root_path, config["source_tables"][table_name])
    df = pd.read_csv(filepath)

    return df


def set_or_create_mlflow_experiment(experiment_name: str) -> str:
    """
    Set or create mlflow experiment
    :param experiment_name: experiment name
    :return: experiment id
    """
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        experiment_id = experiment.experiment_id
    except:
        experiment_id = client.create_experiment(name=experiment_name)

    return experiment_id
