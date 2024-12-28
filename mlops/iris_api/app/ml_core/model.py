from pathlib import Path
from typing import Union
import pickle


def load_model(path: Union[str, Path]):
    # check if the model exists
    if not Path(path).exists():
        return None

    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def train_model():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib

    iris = load_iris()
    X, y = iris.data, iris.target
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

