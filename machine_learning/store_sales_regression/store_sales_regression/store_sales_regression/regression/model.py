from functools import partial

import pandas as pd
import numpy as np
from typing import Union
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
import pickle 
import mlflow
from mlflow.models import infer_signature
import tempfile
import datetime 
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def create_pipeline(feature_names: list) -> Pipeline:
    imputer = ColumnTransformer(
        [
            (
                "impute_missing",
                SimpleImputer(strategy="constant", fill_value=0),
                feature_names,
            )
        ]
    )

    pipeline = Pipeline([("preprocessing", imputer), ("model", RandomForestRegressor())])
    return pipeline


def objective_function(params, **kwargs):

    params = {k:int(v) for k,v in params.items() if k.startswith("model")}
    pipeline = create_pipeline(feature_names=kwargs["feature_names"])   
    pipeline.set_params(**params)

    pipeline.fit(kwargs["x_train"], kwargs["y_train"])
    metrics = ("r2", "neg_root_mean_squared_error")
    cv_results = cross_validate(
        estimator=pipeline,
        X=kwargs["x_train"],
        y=kwargs["y_train"],
        cv=5,
        scoring=metrics,
        return_train_score=True,
    )
    print("CV_RESULTS")
    print(cv_results)
    mlflow_logger(
        experiment_id=kwargs["experiment_id"],
        cv_results=cv_results,
        params=params,
        signature=infer_signature(kwargs["x_train"], kwargs["y_train"]),
        input_example=kwargs["x_train"].iloc[0:5],
        estimator=pipeline,
    )

    return {"loss": -cv_results["test_r2"].mean(), "status": hyperopt.STATUS_OK}


def mlflow_logger(
    experiment_id: str,
    cv_results: dict,
    params: dict,
    signature: mlflow.models.Model.signature,
    input_example: Union[np.array, pd.DataFrame],
    estimator,
) -> None:
    with mlflow.start_run(nested=True, experiment_id=experiment_id) as run:
        # logging testing metrics
        mlflow.log_metric("test_r2", cv_results["test_r2"].mean())
        mlflow.log_metric("test_neg_root_mean_squared_error", cv_results["test_neg_root_mean_squared_error"].mean())

        # logging training metrics
        mlflow.log_metric("train_r2", cv_results["train_r2"].mean())
        mlflow.log_metric("train_neg_root_mean_squared_error", cv_results["train_neg_root_mean_squared_error"].mean())
        # logging params
        mlflow.log_params(params)

        # logging model
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model_pipeline",
            signature=signature,
            input_example=input_example,
        )


def train_model(
    x_train: Union[pd.DataFrame, np.array],
    x_test: Union[pd.DataFrame, np.array],
    y_train: Union[pd.DataFrame, np.array],
    y_test: Union[pd.DataFrame, np.array],
    feature_names: list,
    experiment_id:str
):
    

    trials = Trials()
    space = {
        "model__n_estimators": hp.quniform("model__n_estimators", low = 10 , high = 1000, q = 10),
        "model__max_depth": hp.quniform("model__max_depth",  low = 2 , high = 100, q = 2),
    }
    extra_params = {
                "x_train": x_train,
                "y_train": y_train,
                "feature_names": feature_names,
                "experiment_id": experiment_id,
            }
    with mlflow.start_run(run_name="Random Forest Regressor Tunning", experiment_id=experiment_id) as run:
        best_params = fmin(
            fn=partial(objective_function, **extra_params),
            space=space,
            trials=trials,
            algo=tpe.suggest,
            max_evals=10)
        
        train_best_model(x_train, y_train, x_test, y_test, feature_names, best_params)
        # save trials information in pickle format
        date = str(datetime.datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = tmpdir + f"\\{date}_trials.pkl"
            with open(trials_file, "wb") as f:
                pickle.dump(trials, f)

            mlflow.log_artifact(trials_file, artifact_path="trials_tunning")
        


def train_best_model(x_train, y_train, x_test, y_test, feature_names, params):
    params = {k:int(v) for k,v in params.items() if k.startswith("model")}
    pipeline = create_pipeline(feature_names=feature_names)
    pipeline.set_params(**params)   
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "r2_score":r2_score(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
    }
    
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="best_model_pipeline",
        signature=infer_signature(x_train, y_train),
        input_example=x_train.iloc[0:5],
    )

