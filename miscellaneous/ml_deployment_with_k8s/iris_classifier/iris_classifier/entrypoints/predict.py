from iris_classifier.data.retrieval import get_train_test_data
from iris_classifier.inference.scoring import get_latest_run_id
from iris_classifier.inference.scoring import get_predictions
from iris_classifier.utils.utils import get_config


def predict():
    """
    Predict using the iris classifier.
    """
    config = get_config()
    x_train, x_test, y_train, y_test = get_train_test_data()
    run_id = get_latest_run_id(config["experiment_name"])
    if run_id is None:
        print("No run id found")
        return

    predictions = get_predictions(run_id, x_test, config)
    map_predictions = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predictions = [map_predictions[pred] for pred in predictions]
    pred_df = x_test.copy()
    pred_df["target"] = y_test
    pred_df["target_mapped"] = pred_df["target"].map(map_predictions)
    pred_df["prediction"] = predictions

    print(pred_df.head(10))
