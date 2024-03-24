from typing import List
from typing import Optional

from mlflow.data.pandas_dataset import PandasDataset
import mlflow
from concrete_compressive_strength.model.mlflow_utils import create_experiment
from concrete_compressive_strength.model.pipelines import get_pipeline

import pandas as pd


class Regressor:
    """
    Regressor class to train and evaluate the model.

    :param regressor: Regressor class
    :param experiment_name: Name of the experiment


    Methods:
    --------

    fit(X, y, feature_names)
        Fit the model to the data.

    Evaluate(eval_data, label, baseline_model_uri)
        Evaluate the model.

    __str__()
        Return the name of the regressor.

    """

    def __init__(self, regressor=None, experiment_name=str):
        """
        Initialize the Regressor class.

        :param regressor: Regressor class
        :param experiment_name: Name of the experiment
        """
        self.regressor = regressor
        self.regressor_name = self.__str__()
        self.experiment_name = experiment_name
        self.model_uri = None
        self.run_id = None

    def fit(self, x, y, feature_names: Optional[List[str]] = None) -> str:
        """
        Fit the model to the data.

        :param X: Features
        :param y: Target

        """
        create_experiment(self.experiment_name)

        if not feature_names:
            feature_names = x.columns.to_list()

        pipeline = get_pipeline(feature_names, regressor=self.regressor())
        with mlflow.start_run(run_name=self.regressor_name) as run:
            # fit the model
            pipeline.fit(x, y)

            # log the model
            artifact_path = self.regressor_name + "_artifact"
            mlflow.sklearn.log_model(pipeline, artifact_path)
            model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

            # end the run
            mlflow.end_run()

        self.model_uri = model_uri
        self.run_id = run.info.run_id

    def evaluate(
        self, eval_data, label: str, baseline_model_uri: Optional[str] = None
    ):
        """
        Evaluate the model.

        :param eval_data: Data to evaluate the model on
        :param label: Target variable
        :return: Evaluation results
        """
        if not self.model_uri:
            raise ValueError(
                "Model has not been trained yet. Please train the model first."
            )

        with mlflow.start_run(run_id=self.run_id):
            result = mlflow.evaluate(
                model=self.model_uri,
                data=eval_data,
                targets=label,
                model_type="regressor",
                evaluators=["default"],
                baseline_model=baseline_model_uri,
            )
            mlflow.end_run()

        return result

    def __str__(self) -> str:
        """
        Return the name of the regressor.
        """

        return self.regressor().__class__.__name__

class CustomPandasDataset(mlflow.pyfunc.PythonModel):
    def __init__(self, df:pd.DataFrame, source:str, targets:str,name:str):
        """
        Initialize the CustomPandasDataset class.
        """
        self.dataset = mlflow.data.from_pandas(df, source=source, targets=targets, name=name)

    def predict(self, context, model_input):
        """
        Predict on the model input.
        """
        return model_input
    
    def log_dataset(self, run_id:str):
        """
        log the dataset.

        :param run_id: Run ID
        :param experiment_id: Experiment ID

        """

        with mlflow.start_run(run_id=run_id):
            mlflow.pyfunc.log_model(artifact_path=self.dataset.name, python_model=self)


    def get_dataset(self):
        """
        Get the dataset.
        """
        return self.dataset.df