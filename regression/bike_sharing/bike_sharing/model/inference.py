import pandas as pd
from bike_sharing.data.retrieval import get_train_test_data
from bike_sharing.model.mlflow_utils import get_model


def get_predictions():
    """
    Get the predictions from the model.
    """
    _, x_test, _, y_test, _ = get_train_test_data()
    model = get_model()
    predictions = model.predict(x_test)

    predictions = pd.DataFrame(predictions, columns=["predictions"])
    predictions["actuals"] = y_test.values
    predictions["datetime"] = x_test.index
    predictions["residuals"] = (
        predictions["actuals"] - predictions["predictions"]
    )

    print(predictions.head(20))
