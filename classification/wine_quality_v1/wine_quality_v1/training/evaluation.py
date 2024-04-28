from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score

import pandas as pd
import numpy as np

from typing import Union
from typing import Optional
from typing import Dict


def get_classification_metrics(
    y_pred: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.DataFrame, np.ndarray],
    prefix: Optional[str] = "test",
) -> Dict[str, float]:
    """
    Get classification metrics.

    :param y_pred: predicted values.
    :param y_true: true values.
    :param prefix: prefix for the metrics.
    :return: dictionary with metrics.
    """
    accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")

    metrics = {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_f1": f1,
        f"{prefix}_recall": recall,
        f"{prefix}_precision": precision,

    }

    return metrics

