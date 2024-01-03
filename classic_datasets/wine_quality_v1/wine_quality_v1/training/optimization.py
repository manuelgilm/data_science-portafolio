from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials

from wine_quality_v1.training.pipelines import get_pipeline
from wine_quality_v1.training.mlflow_utils import get_or_create_experiment
import mlflow
from functools import partial
from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

from typing import Dict
from typing import Any
import pandas as pd


def objective_function(
    params: Dict[str, Any],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    experiment_id: str,
)->float:
    """
    Function to minimize.

    :param params: parameters to optimize.
    :param x_train: features.
    :param x_test: features.
    :param y_train: target.
    :param y_test: target.
    :param experiment_id: experiment id.
    :return: metric to minimize.
    """
    numerical_feautures = x_train.columns.tolist()
    pipeline = get_pipeline(
        numerical_features=numerical_feautures,
        categorical_features=[],
    )
    # cast params to intenger using a loop
    params_ = {key:int(value) for key, value in params.items()}
    pipeline.set_params(**params_)

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        print("Run ID:", run.info.run_id)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        precision = precision_score(y_test, predictions, average="weighted")
        report = classification_report(y_test, predictions, output_dict=True)

        mlflow.log_metrics(
            {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
        )
        return -report["weighted avg"]["f1-score"]


def optimize(
    experiment_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
)->str:
    """ """

    experiment_id = get_or_create_experiment(experiment_name)
    search_space = {
        "classifier__n_estimators": hp.quniform(
            "classifier__n_estimators", low=10, high=100, q=2
        ),
        "classifier__max_depth": hp.quniform(
            "classifier__max_depth", low=2, high=10, q=1
        )
    }
    
    trials = Trials()
    run_name = f"run-{str(datetime.now())}"
    with mlflow.start_run(run_name=run_name) as run:
        print("Run ID:", run.info.run_id)
        best = fmin(
            fn=partial(
                objective_function,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                experiment_id=experiment_id,
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
            show_progressbar=True,
        )

        pipeline = get_pipeline(
            numerical_features=x_train.columns.tolist(),
            categorical_features=[],
        )
        best_ = {key:int(value) for key, value in best.items()}
        pipeline.set_params(**best_)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        report = classification_report(y_test, predictions, output_dict=True)
        mlflow.log_metrics(
            {
                "best_accuracy": accuracy_score(y_test, predictions),
                "best_f1": f1_score(y_test, predictions, average="weighted"),
                "best_recall": recall_score(y_test, predictions, average="weighted"),
                "best_precision": precision_score(y_test, predictions, average="weighted")
            }
        )
        mlflow.sklearn.log_model(pipeline, "best_model")

        return run.info.run_id
