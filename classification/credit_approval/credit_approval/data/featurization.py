from typing import Tuple

import pandas as pd
from credit_approval.data.retrieval import get_data
from sklearn.model_selection import train_test_split


def get_train_test_sets(
    target: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get the features of the credit approval dataset.
    """
    df = get_data()
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=target, axis=1),
        df[target],
        test_size=test_size,
        random_state=42,
    )

    return x_train, x_test, y_train, y_test
