from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Dict, Union
import pandas as pd
import mlflow


def get_pipeline(
    model_type: str,
    feature_names: list,
) -> Pipeline:
    """
    Creates a pipeline for a given model and parameters.

    :param model: model
    :return: pipeline
    """

    imputer = ColumnTransformer(
        [
            (
                "impute_missing",
                SimpleImputer(strategy="constant", fill_value=0),
                feature_names,
            ),
        ]
    )

    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor()
    elif model_type == "GradientBoostingRegressor":
        model = GradientBoostingRegressor()
    elif model_type == "DecisionTreeRegressor":
        model = DecisionTreeRegressor()

    pipeline = Pipeline([("preprocessing", imputer), ("model", model)])
    return pipeline


def train_model(df: pd.DataFrame, feature_names):
    store = df["store"].iloc[0]
    item = df["item"].iloc[0]
    run_id = df["run_id"].iloc[0]
    # get model pipeline
    rf_pipeline = get_pipeline(
        model_type="RandomForestRegressor", feature_names=feature_names
    )
    gb_pipeline = get_pipeline(
        model_type="GradientBoostingRegressor", feature_names=feature_names
    )
    dt_pipeline = get_pipeline(
        model_type="DecisionTreeRegressor", feature_names=feature_names
    )

    # parameters for grid search
    parameters = {
        "model__n_estimators": [10, 20, 30],
        "model__max_depth": [5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
    }

    # get cross validation models
    rf_cv_model = get_cv_model(rf_pipeline, parameters)
    gb_cv_model = get_cv_model(gb_pipeline, parameters)
    dt_cv_model = get_cv_model(dt_pipeline, parameters)

    # fit models
    rf_cv_model.fit(df[feature_names], df["sales"])
    gb_cv_model.fit(df[feature_names], df["sales"])
    dt_cv_model.fit(df[feature_names], df["sales"])

    with mlflow.start_run(run_id=run_id, nested=True) as outer_run:
        experiment_id = outer_run.info.experiment_id
        print("experiment_id:", experiment_id)
        run_name = f"store_{store}_item_{item}"
        with mlflow.start_run(
            run_name=run_name, experiment_id=experiment_id, nested=True
        ) as inner_run:
            print(f"RUNNING EXPERIMENT:{run_name}")

            # random forest model
            mlflow.sklearn.log_model(
                sk_model=rf_cv_model,
                artifact_path=f"{store}_{item}_randomforest_regressor",
            )
            mlflow.log_params(rf_cv_model.best_params_)
            mlflow.log_metric("r2_score", rf_cv_model.best_score_)

            # gradient boosting model
            mlflow.sklearn.log_model(
                sk_model=gb_cv_model,
                artifact_path=f"{store}_{item}_gradientboosting_regressor",
            )
            mlflow.log_params(gb_cv_model.best_params_)
            mlflow.log_metric("r2_score", gb_cv_model.best_score_)

            # decision tree model
            mlflow.sklearn.log_model(
                sk_model=dt_cv_model,
                artifact_path=f"{store}_{item}_decisiontree_regressor",
            )
            mlflow.log_params(dt_cv_model.best_params_)
            mlflow.log_metric("r2_score", dt_cv_model.best_score_)
            mlflow.set_tags({"store": store, "item": item})

            # creating model paths
            rf_artifact_uri = (
                f"runs:/{inner_run.info.run_id}/{store}_{item}_randomforest_regressor"
            )
            gb_artifact_uri = f"runs:/{inner_run.info.run_id}/{store}_{item}_gradientboosting_regressor"
            dt_artifact_uri = (
                f"runs:/{inner_run.info.run_id}/{store}_{item}_decisiontree_regressor"
            )
            columns = [
                "store",
                "item",
                "rf_artifact_uri",
                "gb_artifact_uri",
                "dt_artifact_uri",
                "rf_best_score",
                "gb_best_score",
                "dt_best_score",
            ]
            data_df = [
                [
                    store,
                    item,
                    rf_artifact_uri,
                    gb_artifact_uri,
                    dt_artifact_uri,
                    rf_cv_model.best_score_,
                    gb_cv_model.best_score_,
                    dt_cv_model.best_score_,
                ]
            ]
            output_df = pd.DataFrame(data=data_df, columns=columns)
    return output_df


def get_cv_model(estimator, parameters):
    # split data using TimeseriesSplit /numSplit = 5 by default
    tscv = TimeSeriesSplit()
    cv_model = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        cv=tscv,
        scoring="r2_score",
        verbose=1,
        n_jobs=-1,
    )

    return cv_model
