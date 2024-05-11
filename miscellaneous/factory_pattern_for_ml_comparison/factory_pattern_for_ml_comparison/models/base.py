from abc import ABC
from abc import abstractmethod

import mlflow
from mlflow.pyfunc import PythonModel

from typing import Optional
from typing import Dict
from typing import Any

from factory_pattern_for_ml_comparison.models.utils import log_dictionary
from factory_pattern_for_ml_comparison.models.utils import check_active_run


class BaseModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, **params):
        pass


class CustomModel(BaseModel, PythonModel):

    def __init__(self, model_name: str):
        self.model_name = model_name

    def log_to_mlflow(self, metadata: Dict[str, Any] = {}):
        """
        Log metadata to MLflow

        :param run_id: Run ID
        :param metadata: Metadata to log
        """

        run_id = check_active_run()
        if not run_id:
            print("No active run found")
            print("Use the Method in a MLflow run")
            return

        self.__log_metadata(metadata)
        # self.__log_model_params()
        self.__log_model()

    def __log_model(self):
        """
        Log model to MLflow
        """
        mlflow.pyfunc.log_model(
            artifact_path=self.__class__.__name__,
            python_model=self,
            conda_env=None,
            code_path=None,
            registered_model_name=None,
        )

    def __log_model_params(self):
        """
        Log model parameters to MLflow
        """
        params = self.get_params()
        log_dictionary(params, "params")

    def __log_metadata(self, metadata: Dict[str, Any]):
        """
        Log metadata to MLflow

        :param run_id: Run ID
        :param metadata: Metadata to log
        """
        self.__valid_metadata(metadata)
        for key, value in metadata.items():
            if metadata.get(key, None) is None:
                continue
            log_dictionary(metadata.get(key, None), key)

    def __valid_metadata(self, metadata: Dict[str, Any]):
        """
        Check if metadata is valid

        :param metadata: Metadata to check
        """
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata must be a dictionary, got {type(metadata)}")
        keys = list(metadata.keys())
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("Keys in metadata must be strings")
        values = metadata.values()
        if not all(isinstance(value, dict) for value in values):
            raise ValueError("Values in metadata must be dictionaries")
        for key in keys:
            if key not in ["metrics", "tags", "params"]:
                raise ValueError(f"Invalid metadata key: {key}")

        return True
