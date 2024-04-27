import mlflow
from iris_classifier.data.retrieval import get_train_test_data
from iris_classifier.train.ml_utils import get_classification_metrics
from iris_classifier.train.ml_utils import get_confusion_matrix
from iris_classifier.train.ml_utils import get_or_create_experiment
from iris_classifier.train.pipelines import ClassifierPipeline
from iris_classifier.utils.utils import get_config


def train():
    """
    Train the iris classifier.
    """
    config = get_config()
    x_train, x_test, y_train, y_test = get_train_test_data()
    experiment_id = get_or_create_experiment(config["experiment_name"])

    clf_pipeline = ClassifierPipeline()
    pipeline = clf_pipeline.get_pipeline(df=x_train)
    pipeline.fit(x_train, y_train)
    cm = get_confusion_matrix(pipeline, x_test, y_test, "test")
    metrics = get_classification_metrics(
        pipeline.predict(x_test), y_test, "test"
    )

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.sklearn.log_model(pipeline, config["model_artifact"])
        mlflow.log_params(pipeline.get_params())
        mlflow.log_metrics(metrics)
        for key, value in cm.items():
            mlflow.log_figure(value, key + ".png")
