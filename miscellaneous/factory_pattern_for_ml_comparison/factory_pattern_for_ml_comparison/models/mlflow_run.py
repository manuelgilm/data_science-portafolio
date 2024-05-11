import mlflow
import tempfile

from typing import Dict
from typing import Any
from typing import Optional

from mlflow.entities import Run
import pickle
import os

from factory_pattern_for_ml_comparison.utils.utils import get_project_root
from factory_pattern_for_ml_comparison.models.utils import log_dictionary


class CustomRun:

    def __init__(
        self, experiment_name: str, run_name: str, tags: Optional[Dict[str, Any]] = None
    ) -> None:

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = self.__create_run(run_name, experiment_name, tags)
        self.run_path = get_project_root() / experiment_name / run_name
        self.run_filename = None

    def log(self, metadata: Dict[str, Any], include_object: bool = False):
        """
        Log metadata to MLflow

        :param metadata: Metadata to log
        """
        if include_object:
            self.save()
            if "artifacts" not in metadata.keys():
                metadata["artifacts"] = {}

            metadata["artifacts"].update(
                {str(self.run_path / self.run_filename): "custom_run"}
            )
        with mlflow.start_run(run_id=self.run.info.run_id):
            self.__log_metadata(metadata)

    def __create_run(
        self, run_name: str, experiment_name: str, tags: Optional[Dict[str, Any]]
    ) -> Run:
        """
        Create a run

        :param run_name: Name of the run
        :param experiment_name: Name of the experiment
        :param tags: Tags to associate with the run
        :return: Run object
        """
        client = mlflow.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_name} not found")

        run = client.create_run(
            experiment_id=experiment.experiment_id, tags=tags, run_name=run_name
        )
        return run

    def save(self):
        """
        Save the run
        """
        os.makedirs(self.run_path, exist_ok=True)
        self.run_filename = f"{self.run.info.run_id}.pkl"
        pickle.dump(self, open(self.run_path / self.run_filename, "wb"))

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
            if key not in ["metrics", "tags", "params", "figures", "artifacts"]:
                raise ValueError(f"Invalid metadata key: {key}")

        return True
    
    def __valid_figures(self, figures: Dict[str, Any]):
        """
        Check if figures is valid

        :param figures: Figures to check
        """
        if not isinstance(figures, dict):
            raise ValueError(f"Figures must be a dictionary, got {type(figures)}")
        keys = list(figures.keys())
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("Keys in figures must be strings")
        values = figures.values()
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Values in figures must be strings")
        return True
    
    def __valid_artifacts(self, artifacts: Dict[str, Any]):
        """
        Check if artifacts is valid

        :param artifacts: Artifacts to check
        """
        if not isinstance(artifacts, dict):
            raise ValueError(f"Artifacts must be a dictionary, got {type(artifacts)}")
        keys = list(artifacts.keys())
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("Keys in artifacts must be strings")
        values = artifacts.values()
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Values in artifacts must be strings")
        return True
