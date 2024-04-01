from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


class KSTest:
    """
    Kolmogorov-Smirnov distance detector.

    :param threshold: threshold value for drift detection

    Methods:
    --------

    score(x_ref, x_new, feature_names)
        Score the distance between two datasets.

    calculate_ks(distribution1, distribution2)
        Calculate the Kolmogorov-Smirnov statistic between two distributions.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        """
        Initialize the KSTest class

        :param threshold: The threshold value for the KS test

        """
        self.threshold = threshold
        self.result = {
            "method": "ks_test",
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
            ks_stat, p_value = self.calculate_ks(
                x_ref[feature], x_new[feature]
            )
            is_drift = p_value < self.threshold
            self.result["features"][feature] = (ks_stat, is_drift)

        return self.result

    def calculate_ks(
        self,
        distribution1: Union[np.ndarray, pd.Series],
        distribution2: Union[np.ndarray, pd.Series],
    ) -> Tuple[float, float]:
        """
        Calculate the Kolmogorov-Smirnov statistic between two distributions.

        :param distribution1: first distribution
        :param distribution2: second distribution
        :return: KS statistic and p-value
        """

        stat, p_value = stats.ks_2samp(distribution1, distribution2)
        return stat, p_value


class TwoWayChiSquaredTest:
    """
    Two-way Chi-Squared Hypothesis Test.

    :param alpha: threshold value for drift detection

    Methods:
    --------

    score(x_ref, x_new, feature_names)
        Calculate the Chi-Squared statistic between two distributions.

    calculate_chi_squared(distribution1, distribution2)
        Calculate the Chi-Squared statistic between two distributions.


    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Initialize the TwoWayChiSquaredTest class

        :param threshold: The threshold value for the Chi-Squared test

        """
        self.threshold = alpha
        self.result = {
            "method": "chi_squared_test",
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
        Calculate the Chi-Squared statistic between two distributions.

        :param x_ref: reference dataset
        :param x_new: new dataset
        :param feature_names: list of feature names
        :return: dictionary containing the drift result
        """
        corrected_alpha = self.threshold / len(feature_names)
        for feature in feature_names:
            p_value = self.calculate_chi_squared(
                x_ref[feature], x_new[feature]
            )
            is_drift = p_value < corrected_alpha
            self.result["features"][feature] = (p_value, is_drift)

        self.result["threshold"] = corrected_alpha
        return self.result

    def calculate_chi_squared(
        self,
        distribution1: Union[np.ndarray, pd.Series],
        distribution2: Union[np.ndarray, pd.Series],
    ) -> Tuple[float, float]:
        """
        Calculate the Chi-Squared statistic between two distributions.

        :param distribution1: first distribution
        :param distribution2: second distribution
        :return: Chi-Squared statistic and p-value
        """
        feature = distribution1.name
        # for feature in self.categorical_columns:
        pdf_count1 = (
            pd.DataFrame(distribution1.value_counts())
            .sort_index()
            .rename(columns={feature: "pdf1"})
        )
        pdf_count2 = (
            pd.DataFrame(distribution2.value_counts())
            .sort_index()
            .rename(columns={feature: "pdf2"})
        )
        pdf_counts = pdf_count1.join(pdf_count2, how="outer")
        obs = np.array([pdf_counts["pdf1"], pdf_counts["pdf2"]])
        _, p, _, _ = stats.chi2_contingency(obs)
        return p
