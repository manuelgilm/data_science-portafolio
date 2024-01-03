from wine_quality_v1.data_preparation.data_preparation import get_wine_dataset_uci
from wine_quality_v1.data_preparation.data_preparation import read_config

from wine_quality_v1.training.pipelines import get_pipeline
from wine_quality_v1.training.optimization import optimize
from wine_quality_v1.training.mlflow_utils import get_or_create_experiment


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import mlflow
from datetime import datetime


def main():
    config = read_config()
    experiment_name = config["experiment_name"]
    run_name = f"run-{str(datetime.now())}"
    label = config["label"]
    df, feature_names = get_wine_dataset_uci()
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_names], df[label], test_size=0.2, random_state=42
    )

    pipeline = get_pipeline(
        numerical_features=feature_names,
        categorical_features=[],
    )

    experiment_id = get_or_create_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.sklearn.autolog()
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        report = classification_report(y_test, predictions, output_dict=True)
        mlflow.log_metrics(report["weighted avg"])

    run_id = optimize(
        experiment_name=experiment_name,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
