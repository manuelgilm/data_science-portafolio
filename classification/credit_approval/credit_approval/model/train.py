from typing import Union

import mlflow
import numpy as np
import pandas as pd
from credit_approval.data.featurization import get_train_test_sets
from credit_approval.data.retrieval import get_data_info
from credit_approval.model.model_utils import create_experiment
from credit_approval.model.model_utils import get_classification_metrics
from credit_approval.model.model_utils import get_performance_plots
from credit_approval.model.pipelines import get_dag_pipeline
from credit_approval.utils.utils import read_config
from skdag import DAG


def train(
    x_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.DataFrame, np.ndarray],
) -> DAG:
    """
    Train the model using the training data.

    :param x_train: The features of the training data.
    :param y_train: The target of the training data.
    :return: The trained model.
    """
    info = get_data_info()
    categorical_columns = list(
        info[(info["type"] == "Categorical") & (info["role"] != "Target")][
            "name"
        ].values
    )
    numerical_columns = list(info[info["type"] == "Continuous"]["name"].values)

    dag = get_dag_pipeline(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

    dag.fit({"input": x_train}, y_train)

    return dag


def train_model() -> None:
    """
    Train the model.
    """
    config = read_config("configs")
    experiment_name = config["experiment_name"]
    artifact_name = config["model_artifact"]
    info = get_data_info()
    target = info[info["role"] == "Target"]["name"].values[0]
    x_train, x_test, y_train, y_test = get_train_test_sets(target=target)

    experiment_id = create_experiment(experiment_name)
    print(f"Experiment ID: {experiment_id}")
    with mlflow.start_run():
        trained_dag = train(x_train, y_train)
        mlflow.sklearn.log_model(trained_dag, artifact_name)
        predictions = trained_dag.predict({"input": x_test})
        metrics = get_classification_metrics(
            y_test, predictions["model"], "test"
        )
        mlflow.log_metrics(metrics)
        for name, plot in get_performance_plots(
            predictions["model"], y_test, "test"
        ).items():
            mlflow.log_figure(plot, name)
