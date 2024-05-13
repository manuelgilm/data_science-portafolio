import pandas as pd
from iris_classifier.data.retrieval import get_dataset
from iris_classifier.data.retrieval import get_train_test_data


def test_data_shape():
    """
    Test the shape of the iris dataset.
    """
    df = get_dataset()

    assert df.shape == (150, 5)


def test_data_type():
    """
    Test the type of the iris dataset.
    """
    df = get_dataset()

    assert isinstance(df, pd.DataFrame)


def test_get_train_test_data():
    """
    Test the train test split.
    """
    x_train, x_test, y_train, y_test = get_train_test_data()

    assert x_train.shape[0] == 112
    assert x_test.shape[0] == 38
    assert y_train.shape[0] == 112
    assert y_test.shape[0] == 38
