from app.schemas.iris_features import IrisFeatures
from typing import List


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
