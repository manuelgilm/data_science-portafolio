from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Optional
from typing import Tuple
import pandas as pd


def get_test_train_data(
    retrain: Optional[bool] = False, test_size: Optional[float] = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get the test and train data from the data source

    :param retrain: If the model is being retrained
    :param test_size: The size of the test data
    :return: The test and train data
    """
    data = load_iris(as_frame=True)

    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size
    )

    if retrain:
        # Load previous user data with features and target
        # Build a train dataset from the previous user data

        # x_train = pd.concat([x_train, previous_user_data[features]])
        # y_train = pd.concat([y_train, previous_user_data[target]])
        raise NotImplementedError("Retraining is not implemented yet")

    return x_train, x_test, y_train, y_test
