from functools import partial

import mlflow
from bike_sharing.data.retrieval import get_train_test_data
from bike_sharing.model.evaluation import get_predicted_vs_true_plot
from bike_sharing.model.evaluation import get_regression_metrics
from bike_sharing.model.mlflow_utils import evaluate_regressor
from bike_sharing.model.mlflow_utils import get_or_create_experiment
from bike_sharing.model.optimize import get_search_space
from bike_sharing.model.optimize import objective_function
from bike_sharing.model.pipelines import get_pipeline
from bike_sharing.utils.utils import read_config
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import tpe
from mlflow.models.signature import infer_signature


def train_model():
    """
    Train the model.
    """
    configs = read_config("training")

    # Get the dataset
    x_train, x_test, y_train, y_test, metadata = get_train_test_data()

    eval_data = x_test.copy()
    eval_data[metadata["target"]] = y_test

    # Get the pipeline
    pipeline = get_pipeline(
        numerical_features=metadata["features"]["numerical_features"],
        categorical_features=metadata["features"]["categorical_features"],
    )

    experiment_name = configs["mlflow"]["experiment_name"]
    tags = configs["mlflow"]["tags"]
    artifact_path = configs["mlflow"]["artifact_path"]

    experiment = get_or_create_experiment(
        experiment_name=experiment_name, tags=tags
    )

    # metadata
    input_example = x_train.iloc[:5]
    model_signature = infer_signature(
        model_input=input_example, model_output=y_train
    )

    with mlflow.start_run(
        run_name="train_model", experiment_id=experiment.experiment_id
    ) as run:
        # Fit the model
        pipeline.fit(x_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
            signature=model_signature,
            input_example=input_example,
        )

        # log params
        mlflow.log_params(pipeline.get_params())

        # log metrics
        evaluate_regressor(
            run_id=run.info.run_id,
            eval_data=eval_data,
            label=metadata["target"],
            model_uri=f"runs:/{run.info.run_id}/{artifact_path}",
        )

        # log the predicted vs true plot
        fig, title = get_predicted_vs_true_plot(
            y_pred=pipeline.predict(x_test), y_true=y_test, prefix="test"
        )
        mlflow.log_figure(figure=fig, artifact_file=title)


def optimize():
    """
    Optimize the model.
    """
    configs = read_config("training")
    x_train, x_test, y_train, y_test, metadata = get_train_test_data()

    categorical_features = metadata["features"]["categorical_features"]
    numerical_features = metadata["features"]["numerical_features"]

    experiment_name = configs["mlflow"]["experiment_name"]
    tags = configs["mlflow"]["tags"]
    run_name = configs["mlflow"]["run_name"]

    experiment = get_or_create_experiment(
        experiment_name=experiment_name, tags=tags
    )
    search_space = get_search_space()

    trials = Trials()

    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment.experiment_id
    ):
        best = fmin(
            fn=partial(
                objective_function,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                experiment_id=experiment.experiment_id,
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )
        # cast all params to int

        best = {key: int(value) for key, value in best.items()}
        mlflow.log_params(best)
        pipeline = get_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        pipeline.set_params(**best)

        pipeline.fit(x_train, y_train)
        metrics = get_regression_metrics(y_test, pipeline.predict(x_test))
        artifact_path = configs["mlflow"]["artifact_path"]
        mlflow.sklearn.log_model(pipeline, artifact_path)
        mlflow.log_metrics(metrics)

        # log the predicted vs true plot
        fig, title = get_predicted_vs_true_plot(
            y_pred=pipeline.predict(x_test), y_true=y_test, prefix="test"
        )
        mlflow.log_figure(figure=fig, artifact_file=title)
