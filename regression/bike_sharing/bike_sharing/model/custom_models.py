from typing import List

import mlflow
import pandas as pd
from bike_sharing.model.pipelines import get_pipeline
from bike_sharing.monitoring.distances import JSDistance


class CustomRegressor(mlflow.pyfunc.PythonModel):
    """
    Custom regressor class.

    :param run_id: MLflow run ID
    :param numerical_features: list of numerical features
    :param categorical_features: list of categorical features

    Methods:
    --------
    fit_estimator(x_train, y_train)
        Fit the model.

    predict(context, model_input)
        Predict the target variable.

    """

    def __init__(
        self,
        run_id: str,
        numerical_features: List[str],
        categorical_features: List[str],
    ) -> None:
        """
        Initialize the CustomRegressor class.

        :param run_id: MLflow run ID
        :param numerical_features: list of numerical features
        :param categorical_features: list of categorical features
        """
        self.model = get_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        self.run_id = run_id
        self.client = mlflow.MlflowClient()

    def fit_estimator(self, x_train, y_train):
        """
        Fit the model.

        :param x_train: training data
        :param y_train: target variable
        :return: trained model
        """
        self.model.fit(x_train, y_train)
        return self.model

    def predict(self, context, model_input, params):
        """
        Predict the target variable.

        :param context: MLflow context
        :param model_input: input data
        :param params: model parameters
        :return: predicted target variable
        """
        x_ref = pd.read_csv(context.artifacts["x_ref_path"])
        if x_ref:
            js_distance = JSDistance()
            js_distance.score(
                x_ref=x_ref,
                x_new=model_input,
                feature_names=self.numerical_features,
            )  # Only numerical features for now
            js_distance.log_drift(self.run_id)

        return self.model.predict(model_input)
