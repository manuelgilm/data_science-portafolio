from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import mlflow
import numpy as np
import pandas as pd
from scipy.spatial import distance


class JSDistance:
    """
    Jensen-Shannon distance detector.

    :param threshold: threshold value for drift detection

    Methods:
    --------

    score(x_ref, x_new, feature_names)
        Score the distance between two datasets.

    calculate_histogram(s1, s2, bins)
        Calculate the histogram of the data.

    calculate_js_distance(p, q)
        Calculate the Jensen-Shannon distance between two
        probability distributions.

    log_drift(run_id)
        Log the drift detection results with MLflow.

    log_with_mlflow(run_id)
        Log the drift detection results with MLflow.

    load_with_mlflow(run_id)
        Load the drift detection model from MLflow.

    """

    def __init__(self, threshold: int = 0.1) -> None:
        """
        Initialize the detector.
        """
        self.threshold = threshold
        self.result = {
            "method": "js_distance",
            "features": {},
            "threshold": self.threshold,
        }

    def score(
        self,
        x_ref: pd.DataFrame,
        x_new: pd.DataFrame,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Score the distance between two datasets.

        :param x_ref: reference dataset
        :param x_new: new dataset
        :param feature_names: list of feature names
        :return: dictionary containing the drift result
        """

        for feature in feature_names:
            hist1, hist2 = self.calculate_histogram(
                x_ref[feature], x_new[feature]
            )
            js_distance = self.calculate_js_distance(hist1, hist2)
            is_drift = js_distance > self.threshold
            self.result["features"][feature] = (js_distance, is_drift)

        return self.result

    def calculate_histogram(
        self,
        s1: Union[np.ndarray, pd.Series],
        s2: Union[np.ndarray, pd.Series],
        bins: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the histogram of the data.

        :param s1: reference raw data
        :param s2: new raw data
        :return: histogram of the data
        """

        global_min = min(min(s1), min(s2))
        global_max = max(max(s1), max(s2))

        hist1, _ = np.histogram(s1, bins=bins, range=(global_min, global_max))
        hist2, _ = np.histogram(s2, bins=bins, range=(global_min, global_max))

        return hist1, hist2

    def calculate_js_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate the Jensen-Shannon distance between two
        probability distributions.

        :param p: probability distribution
        :param q: probability distribution
        :return: Jensen-Shannon distance
        """

        js_stat = distance.jensenshannon(p, q, base=2)
        js_stat = np.round(js_stat, 4)
        return js_stat

    def log_drift(self, run_id: str) -> None:
        """
        Log the drift detection results with MLflow.

        :param run_id: MLflow run ID
        :return: None
        """
        if self.result["features"] == {}:
            raise ValueError("No feature drift to log. Use the score method")

        client = mlflow.MlflowClient()

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_id=run_id):
            for feature in self.result["features"]:
                # get metric history
                metric_name = f"js_distance_{feature}"
                step = len(client.get_metric_history(run_id, metric_name))
                mlflow.log_metric(
                    key=metric_name,
                    value=self.result["features"][feature][0],
                    step=step,
                )

    def log_with_mlflow(self, run_id: str):
        """
        Log the drift detection results with MLflow.

        :param run_id: MLflow run ID
        :return: None
        """

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_id=run_id):
            mlflow.pyfunc.log_model(
                python_model=self,
                registered_model_name=self.__class__.__name__,
            )

    def load_with_mlflow(self, run_id: str) -> "JSDistance":
        """
        Load the drift detection model from MLflow.

        :param run_id: MLflow run ID
        :return: Unwrapped model
        """
        model_uri = f"runs:/{run_id}/{self.__class__.__name__}"
        model = mlflow.pyfunc.load_model(model_uri)
        unwrapped_model = model.unwrap_python_model()
        return unwrapped_model
