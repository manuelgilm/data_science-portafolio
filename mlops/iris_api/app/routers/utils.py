from app.schemas.iris_features import IrisFeatures
from typing import List
from typing import Dict
from typing import Any


def process_features(features: IrisFeatures) -> List[float]:
    """
    Process the features from the IrisFeatures schema

    :param features: IrisFeatures
    :return: List[float]
    """
    return [
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]
    ]


def get_drift_data(drift: Dict[str, Any]):
    """
    Calculate drift data from the drift prediction.

    :param drift: Dict[str, Any]
    :return: Dict[str, Any]
    """

    data = drift["data"]
    drift_data = {
        "threshold": data["threshold"],
        "p_value_feature_0": data["p_val"][0],
        "p_value_feature_1": data["p_val"][1],
        "p_value_feature_2": data["p_val"][2],
        "p_value_feature_3": data["p_val"][3],
        "distance_feature_0": data["distance"][0],
        "distance_feature_1": data["distance"][1],
        "distance_feature_2": data["distance"][2],
        "distance_feature_3": data["distance"][3],
    }
    return drift_data
