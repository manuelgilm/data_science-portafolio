import mlflow
from concrete_compressive_strength.data.retrieval import get_dataset
from concrete_compressive_strength.data.retrieval import process_column_names

# fmt: off
from concrete_compressive_strength.model.evaluation import get_predicted_vs_true_plot  # noqa
from concrete_compressive_strength.model.evaluation import get_regression_metrics  # noqa
from concrete_compressive_strength.model.mlflow_utils import create_experiment
from concrete_compressive_strength.model.mlflow_utils import log_figures
from concrete_compressive_strength.model.pipelines import get_pipeline
from sklearn.model_selection import train_test_split

# fmt: on


def train():
    """
    Train the model and save it to disk.
    """
    # get the dataset
    df, metadata = get_dataset()

    target = metadata[metadata["role"] == "Target"]["name"].values[0]
    target = target.lower().replace(" ", "_")

    # process column names
    df = process_column_names(df)
    print(df.head())
    # get the numerical columns
    numerical_columns = df.columns.to_list()
    numerical_columns.remove(target)
    print(numerical_columns, target)
    # get the pipeline
    pipeline = get_pipeline(numerical_columns)

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), df[target], test_size=0.2, random_state=0
    )
    experiment_id = create_experiment("concrete_compressive_strength")
    print(experiment_id)
    with mlflow.start_run(run_name="concrete_compressive_strength"):
        # fit the model
        pipeline.fit(X_train, y_train)

        # make predictions
        y_pred = pipeline.predict(X_test)

        # evaluate the model
        regression_metrics = get_regression_metrics(y_test, y_pred, "test")
        print(regression_metrics)

        # get predicted vs true plot
        figures = get_predicted_vs_true_plot(y_pred, y_test, "test")

        # log figures
        log_figures(figures)
        # log the model
        mlflow.sklearn.log_model(pipeline, "model")

        # log the metrics
        mlflow.log_metrics(regression_metrics)

        # end the run
        mlflow.end_run()
