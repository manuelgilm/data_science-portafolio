from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Optional
import pandas as pd
from typing import Tuple


class Trainer:
    def __init__(self, model:Optional[RandomForestClassifier] = None):
        self.model = model

    def fit_model(self):
        pass

    def _new_data_check(self) -> bool:
        """
        Check if new data is available for training.

        :return: True if new data is available, False otherwise
        """
        return False

    def _get_dataset_from_path(self, path: str) -> Optional[pd.DataFrame]:
        """
        Get the data from the given path

        :param path: The path to the data
        :return: The data
        """
        df = pd.read_csv(path)
        if df.empty:
            return None

        return df

    def get_train_test_data(
        self, test_size: Optional[float] = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get the training and testing data.

        :param test_size: The size of the test data
        """

        if self._new_data_check():
            return self.get_training_data_from_prediction_data()

        features, target = self.get_training_data_from_original_dataset()
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )

        return x_train, x_test, y_train, y_test

    def get_training_data_from_original_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the training data from the original dataset.

        :return: The features and target variable
        """
        dataset = self._get_dataset_from_path("data/raw/iris.csv")
        if dataset is None:
            dataset = load_iris(as_frame=True)
            # save the data to the raw folder
            dataset.data.to_csv("data/raw/iris.csv", index=False)

        features = dataset.drop(columns=["target"])
        target = dataset["target"]

        return features, target

    def get_training_data_from_prediction_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass

    def train_model(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> RandomForestClassifier:
        """
        Train a Random Forest model on the given data

        :param x_train: The training data
        :param y_train: The target variable
        :return: The trained model
        """
        self.model.fit(x_train, y_train)
        return self.model
