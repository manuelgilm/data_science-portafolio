from sklearn.model_selection import train_test_split
from wine_quality_v1.data_preparation.data_preparation import get_wine_dataset_uci
from wine_quality_v1.training.mlflow_utils import get_model
from wine_quality_v1.training.evaluation import get_classification_metrics

import pandas as pd


def get_predictions():
    """
    Get the predictions from the model.
    """
    df, metadata = get_wine_dataset_uci()
    feature_names = metadata["numerical_features"] + metadata["categorical_features"]
    label = metadata["target"]

    _, x_test, _, y_test = train_test_split(
        df[feature_names], df[label], test_size=0.2, random_state=42
    )

    model = get_model()
    predictions = model.predict(x_test)

    predictions = pd.DataFrame(predictions, columns=["predictions"])
    predictions["actuals"] = y_test.values

    metrics = get_classification_metrics(
        y_pred=predictions["predictions"], y_true=predictions["actuals"]
    )

    print(predictions.head(20))

    print("METRICS")
    print(metrics)
