import json

import mlflow
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from training_package.training import get_regression_metrics
from training_package.training import get_train_test_data

x_train, x_test, y_train, y_test = get_train_test_data(n_features=20)
scoring_data = x_test.to_dict(orient="split")


def get_predictions(model_name: str, x: pd.DataFrame):
    """
    Get predictions from model.

    :param model_name: name of the model
    :param x: features

    """
    data = {
        "dataframe_split": x.to_dict(orient="split"),
        "params": {"model_name": model_name},
    }

    headers = {"Content-Type": "application/json"}
    endpoint = "http://127.0.0.1:5000/invocations"

    response = requests.post(endpoint, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        prediction = response.json()["predictions"]
        return prediction
    else:
        return response.text


predictions_model0 = get_predictions(model_name="regressor0", x=x_test)
predictions_model1 = get_predictions(model_name="regressor1", x=x_test)

metrics1 = {
    "regressor1_r2_score": r2_score(y_test, predictions_model1),
    "regressor1_mean_squared_error": mean_squared_error(
        y_test, predictions_model1
    ),
    "regressor1_mean_absolute_error": mean_absolute_error(
        y_test, predictions_model1
    ),
}

metrics0 = {
    "regressor2_r2_score": r2_score(y_test, predictions_model0),
    "regressor2_mean_squared_error": mean_squared_error(
        y_test, predictions_model0
    ),
    "regressor2_mean_absolute_error": mean_absolute_error(
        y_test, predictions_model0
    ),
}

print(metrics1)
print(metrics0)
run_id = "fba41be6fcae43178bd56c72ddfa9368"
