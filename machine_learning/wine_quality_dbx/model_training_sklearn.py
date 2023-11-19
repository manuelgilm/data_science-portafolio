import mlflow
from databricks.feature_store import FeatureStoreClient
from wine_quality.data_preparation import get_configurations
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

from wine_quality.features import get_train_test_ids, get_training_testing_data
from wine_quality.model_func import create_mlflow_experiment


def get_sklearn_pipeline(config: dict, feature_names: list) -> Pipeline:
    """
    This function returns the sklearn pipeline.
    """
    # default values (mean)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    transformer = ColumnTransformer(
        [
            ("numeric_imputer", imputer, feature_names),
        ]
    )

    pipeline = Pipeline(
        steps=[("transformer", transformer), ("model", RandomForestClassifier())]
    )

    return pipeline


if __name__ == "__main__":
    fs = FeatureStoreClient()
    configs = get_configurations(filename="common_params")
    train_ids, test_ids, feature_names = get_train_test_ids(configs=configs)
    train_set, test_set = get_training_testing_data(
        configs=configs,
        feature_names=feature_names,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    train_sdf = train_set.load_df()
    train_pdf = train_sdf.toPandas()
    test_sdf = test_set.load_df()
    test_pdf = test_sdf.toPandas()

    print(train_sdf.show())

    experiment_name = configs["experiment_name"]
    create_mlflow_experiment(experiment_name=experiment_name)

    pipeline = get_sklearn_pipeline(config=configs, feature_names=feature_names)
    pipeline.fit(train_pdf[feature_names], train_pdf["target"])

    predictions = pipeline.predict(test_pdf[feature_names])

    roc = roc_auc_score(test_pdf["target"], predictions)
    precision = precision_score(test_pdf["target"], predictions)
    recall = recall_score(test_pdf["target"], predictions)

    mlflow.log_metric("roc", roc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # logging model with mlflow
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="sklearn_pipeline")

    fs.log_model(
        pipeline,
        artifact_path="fs_sklearn_pipeline",
        flavor=mlflow.sklearn,
        training_set=train_set,
    )
