from abc import ABC
from abc import abstractmethod

import mlflow
from mlflow.pyfunc import PythonModel

from typing import Optional
from typing import Dict
from typing import Any


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

    def log_to_mlflow(self, run_id, metadata: Dict[str, Any] = None):
        if not run_id:
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
            else:
                raise ValueError("No run_id provided and no active run found.")
        metrics = metadata.get("metrics", {})
        tags = metadata.get("tags", {})
        figures = metadata.get("figures", {})
        with mlflow.start_run(run_id=run_id) as run:
            print(f"Logging to MLflow run_id={run.info.run_id}")
            mlflow.log_params(self.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_figures(figures)
            mlflow.set_tags(tags)
            mlflow.pyfunc.log_model(artifact_path=self.model_name, python_model=self)
