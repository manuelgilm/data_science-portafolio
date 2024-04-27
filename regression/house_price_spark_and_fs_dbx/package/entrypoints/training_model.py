import mlflow
from databricks.feature_store import FeatureStoreClient
from package.configs.utils import get_configs
from package.training.preprocessing_pipeline import get_pipeline
from package.training.retrieval import get_train_test_ids
from package.training.retrieval import get_train_testing_sets
from package.training.train import train_model
from package.utils.utils import set_or_create_experiment

if __name__ == "__main__":
    configs = get_configs()

    model_name = configs["model_name"]
    database_name = configs["database_name"]
    table_name = configs["table_name"]
    target = configs["target"]

    numerical_features = configs["numerical_features"]
    pipeline = get_pipeline(
        numerical_features=numerical_features,
        categorical_features=[],
    )
    train_sdf, test_sdf = get_train_test_ids(
        database_name=database_name, table_name=table_name, label=target
    )
    training_set, testing_set = get_train_testing_sets(
        database_name=database_name,
        table_name=table_name,
        feature_names=numerical_features,
        label=target,
        train_ids=train_sdf,
        test_ids=test_sdf,
    )

    set_or_create_experiment(experiment_name=configs["experiment_name"])

    train_sdf = training_set.load_df()
    train_pdf = train_sdf.toPandas()
    x_train = train_pdf.drop(target, axis=1)
    y_train = train_pdf[target]

    run_id, pipeline = train_model(
        pipeline=pipeline, run_name=configs["run_name"], x=x_train, y=y_train
    )
    # log model using feature store
    with mlflow.start_run(run_id=run_id) as run:
        FeatureStoreClient().log_model(
            model=pipeline,
            artifact_path="ml_regressor_fs",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )
