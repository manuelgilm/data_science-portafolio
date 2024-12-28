from pathlib import Path
from typing import Union
import pickle
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_model(path: Union[str, Path]):
    """
    Load a model from a file

    :param path: str or Path
    :return: model
    """
    # check if the model exists
    if not Path(path).exists():
        return None

    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def train_model():
    """
    Train a model and save it to a file.

    """

    # Create a dummy dataset with similar characteristics to the iris dataset but larger
    X, y = make_classification(
        n_samples=10000,  # Increase the number of samples
        n_features=4,  # Same number of features as the iris dataset
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open("app/ml_core/model.pkl", "wb") as f:
        pickle.dump(model, f)
    # save the train data to use as reference
    with open("app/ml_core/train_data.pkl", "wb") as f:
        pickle.dump((X_train, y_train), f)
