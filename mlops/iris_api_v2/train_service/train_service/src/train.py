from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from train_service.src.utils import load_pickle, save_pickle
from typing import Optional
import pandas as pd
from pathlib import Path 
from typing import Tuple
import os 

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
        if Path(path).exists():
            return load_pickle(path)
        return None

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
        dataset_path = "app/data/raw/iris.pkl"
        dataset = self._get_dataset_from_path(dataset_path)
        if dataset is None:
            dataset = load_iris(as_frame=True)
            # save the data to the raw folder
            # this will be a volume in the docker container
            save_pickle(dataset, dataset_path)

        features = dataset.data
        target = dataset.target

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
