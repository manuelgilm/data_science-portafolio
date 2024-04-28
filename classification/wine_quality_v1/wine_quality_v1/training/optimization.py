from datetime import datetime
from functools import partial
from typing import Any
from typing import Dict
from typing import List
import mlflow
import pandas as pd
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from wine_quality_v1.training.mlflow_utils import get_or_create_experiment
from wine_quality_v1.training.pipelines import get_pipeline
from wine_quality_v1.training.pipelines import train
from wine_quality_v1.training.evaluation import get_classification_metrics


def objective_function(
    params: Dict[str, Any],
    numerical_features: List[str],
    categorical_features: List[str],
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    experiment_id: str,
) -> float:
    """
    Function to minimize.

    :param params: parameters to optimize.
    :param numerical_features: numerical features.
    :param categorical_features: categorical features.
    :param x_train: features.
    :param x_test: features.
    :param y_train: target.
    :param y_test: target.
    :param experiment_id: experiment id.
    :return: metric to minimize.
    """
    pipeline = train(
        params=params,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        x_train=x_train,
        y_train=y_train,
    )

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        print("Run ID:", run.info.run_id)
        predictions = pipeline.predict(x_val)
        test_metrics = get_classification_metrics(
            y_pred=predictions, y_true=y_val, prefix="val"
        )
        mlflow.log_metrics(metrics=test_metrics)
        return -test_metrics["val_f1"]


def optimize(
    experiment_id: str,
    numerical_feautres: List[str],
    categorical_features: List[str],
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
) -> str:
    """ """

    search_space = {
        "classifier__n_estimators": hp.quniform(
            "classifier__n_estimators", low=10, high=100, q=2
        ),
        "classifier__max_depth": hp.quniform(
            "classifier__max_depth", low=2, high=10, q=1
        ),
    }

    trials = Trials()
    run_name = f"run-{str(datetime.now())}"
    with mlflow.start_run(run_name=run_name) as run:
        print("Run ID:", run.info.run_id)
        best_params = fmin(
            fn=partial(
                objective_function,
                numerical_features=numerical_feautres,
                categorical_features=categorical_features,
                x_train=x_train,
                x_val=x_val,
                y_train=y_train,
                y_val=y_val,
                experiment_id=experiment_id,
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
            show_progressbar=True,
        )
    return best_params
