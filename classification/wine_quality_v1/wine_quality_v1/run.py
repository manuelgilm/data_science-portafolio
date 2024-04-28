from datetime import datetime

import mlflow
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from wine_quality_v1.data_preparation.data_preparation import get_wine_dataset_uci
from wine_quality_v1.data_preparation.data_preparation import read_config
from wine_quality_v1.training.mlflow_utils import get_or_create_experiment
from wine_quality_v1.training.optimization import optimize
from wine_quality_v1.training.pipelines import get_pipeline
from wine_quality_v1.training.pipelines import train
from wine_quality_v1.training.evaluation import get_classification_metrics


def main():
    config = read_config()
    experiment_name = config["experiment_name"]
    exp_tags = config["tags"]
    alias = config["aliases"]["prod"]
    registered_model_name = config["registered_model_name"]
    run_name = f"run-{str(datetime.now())}"
    df, metadata = get_wine_dataset_uci()
    feature_names = metadata["numerical_features"] + metadata["categorical_features"]
    label = metadata["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_names], df[label], test_size=0.2, random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42
    )

    experiment = get_or_create_experiment(experiment_name, exp_tags)
    best_params = optimize(
        experiment_id=experiment.experiment_id,
        numerical_feautres=metadata["numerical_features"],
        categorical_features=metadata["categorical_features"],
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
    )
    best_pipeline = train(
        params=best_params,
        numerical_features=metadata["numerical_features"],
        categorical_features=metadata["categorical_features"],
        x_train=x_train,
        y_train=y_train,
    )
    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment.experiment_id
    ) as run:
        print("Run ID:", run.info.run_id)
        predictions = best_pipeline.predict(x_test)
        metrics = get_classification_metrics(y_pred=predictions, y_true=y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_pipeline, "best_model")

        model_version = mlflow.register_model(
            f"runs:/{run.info.run_id}/best_model",
            registered_model_name,
        )

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias,
            version=model_version.version,
        )
