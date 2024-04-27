import mlflow
import pandas as pd

from training_package.mlflow_utils import create_or_set_experiment
from training_package.mlflow_utils import get_custom_signature

from training_package.training import get_processing_pipeline
from training_package.training import get_regression_metrics

from typing import Tuple


class MultiRegressor(mlflow.pyfunc.PythonModel):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_id = create_or_set_experiment(experiment_name)
        self.models = {}

    def fit(self, X_train: Tuple[pd.DataFrame], Y_train: Tuple[pd.DataFrame]):
        """
        Fit model.

        :param x_train: features
        :param y_train: target
        """
        metrics = {}
        params = {}
        for x_train, y_train, i in zip(X_train, Y_train, range(len(X_train))):
            regressor = get_processing_pipeline(x_train.columns)
            regressor.fit(x_train, y_train)
            self.models[f"regressor{i}"] = regressor
            metrics.update(
                get_regression_metrics(
                    regressor, x_train, y_train, prefix=f"regressor{i}_train"
                )
            )

            model_params = regressor.named_steps["model"].get_params()
            model_params = {
                f"regressor{i}__{k}": v for k, v in model_params.items() if v
            }
            params.update(model_params)

        signature = get_custom_signature(
            x_train, y_train, params={"model_name": "Name of the model"}
        )

        with mlflow.start_run(run_name="multi_model", experiment_id=self.experiment_id) as run:
            mlflow.pyfunc.log_model(
                "multimodel", python_model=self, signature=signature
            )
            mlflow.log_metrics(metrics)
            mlflow.log_params(params)

        return run.info.run_id

    def evaluate(
        self, X_test: Tuple[pd.DataFrame], Y_test: Tuple[pd.DataFrame], run_id
    ) -> dict:
        """
        Evaluate model.

        :param x_test: features
        :param y_test: target
        :return: metrics
        """
        metrics = {}
        for x_test, y_test, i in zip(X_test, Y_test, range(len(X_test))):
            metrics.update(get_regression_metrics(
                self.models[f"regressor{i}"],
                x_test,
                y_test,
                prefix=f"regressor{i}_test",
            ))

        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_metrics(metrics)

    def predict(self, context, model_input, params):
        """
        Predict.

        :param context: context
        :param model_input: model input
        :param params: parameters
        """
        model_name = params["model_name"]
        model = self.models[model_name]
        return model.predict(model_input)
