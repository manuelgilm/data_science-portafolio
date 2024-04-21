import mlflow
from bike_sharing.model.evaluation import get_regression_metrics
from bike_sharing.model.pipelines import get_pipeline


def objective_function(
    params,
    x_train,
    y_train,
    x_test,
    y_test,
    numerical_features,
    categorical_features,
    experiment_id: str,
):
    """
    Define the objective function to optimize.

    :param params: Parameters to optimize
    :param x_train: Training features
    :param y_train: Training target
    :param x_test: Test features
    :param y_test: Test target
    :param numerical_features: Numerical features
    :param categorical_features: Categorical features
    :param experiment_id: Experiment ID
    :return: R2 score
    """

    pipeline = get_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )
    # cast all params to int
    params = {key: int(value) for key, value in params.items()}

    pipeline.set_params(**params)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    regression_metrics = get_regression_metrics(y_test, predictions)
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        mlflow.log_metrics(regression_metrics)

    return -regression_metrics["r2_score"]
