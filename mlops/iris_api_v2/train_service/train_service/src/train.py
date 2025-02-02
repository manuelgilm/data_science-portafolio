from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest model on the given data

    :param x_train: The training data
    :param y_train: The target variable
    :return: The trained model
    """
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model
