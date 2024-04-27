from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def get_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> Dict[str, float]:
    """
    Get regression metrics.

    :param y_true: True values
    :param y_pred: Predicted values
    :param prefix: Prefix for the metrics
    :return: Dictionary of metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {f"{prefix}_r2": r2, f"{prefix}_mae": mae, f"{prefix}_mse": mse}


def get_predicted_vs_true_plot(
    y_pred: np.ndarray, y_true: np.ndarray, prefix: str
) -> Dict[str, plt.Figure]:
    """
    Get a plot of predicted vs true values.

    :param y_pred: Predicted values
    :param y_true: True values
    :param prefix: Prefix for the plot
    :return: Dictionary of figures
    """

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=4
    )
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title(f"{prefix} - Predicted vs True")
    return {f"{prefix}_predicted_vs_true.png": fig}
