from databricks.feature_store import FeatureStoreClient
from databricks.feature_store import FeatureLookup

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from wine_quality.features import get_train_test_ids, get_training_testing_data
from wine_quality.data_preparation import get_configurations

import mlflow

def get_pipeline(columns: list) -> Pipeline:
    """
    This function creates a pipeline for training the model.

    params: columns: list
    return: pipeline: pyspark pipeline
    """

    imputed_columns = [col + "_imputed" for col in columns]
    numerical_imputer = Imputer(
        inputCols=columns, outputCols=imputed_columns, strategy="mean"
    )
    assembler = VectorAssembler(inputCols=imputed_columns, outputCol="features")
    rf = RandomForestClassifier(featuresCol="features", labelCol="target", numTrees=10)
    pipeline = Pipeline(stages=[numerical_imputer, assembler, rf])

    return pipeline

def create_mlflow_experiment(experiment_name: str) -> None:
    """
    This function creates a mlflow experiment.

    params: experiment_name: str

    """
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)


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
    test_sdf = test_set.load_df()

    experiment_name = "/Shared/databricks_certification/experiments/wine_quality_classification"
    create_mlflow_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name="wine_quality_classification") as run:
        pipeline = get_pipeline(columns=feature_names)
        pipeline_model = pipeline.fit(train_sdf)
        predictions = pipeline_model.transform(test_sdf)
        evaluator = BinaryClassificationEvaluator(
            labelCol="target", rawPredictionCol="prediction", metricName="areaUnderROC"
        )

        # get metrics
        roc = evaluator.evaluate(predictions)
        prc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)

        mlflow.log_metric("roc", roc)
        mlflow.log_metric("prc", prc)
        mlflow.spark.log_model(
            pipeline_model, "pipeline", input_example=train_sdf.limit(5).toPandas()
        )
        # logging model with mlflow
        fs.log_model(
            model=pipeline_model,
            artifact_path="fs_pipeline",
            flavor=mlflow.spark,
            training_set=train_set,
        )
