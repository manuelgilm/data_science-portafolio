from typing import Dict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_predicted_vs_true_plot(
    y_pred: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series],
    prefix: str,
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
    return fig, f"{prefix}_predicted_vs_true.png"
