from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import pandas as pd
from typing import Tuple


def get_train_test_data(
    n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)
    return x_train, x_test, y_train, y_test
