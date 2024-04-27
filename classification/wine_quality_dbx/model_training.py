import mlflow
from databricks.feature_store import FeatureLookup
from databricks.feature_store import FeatureStoreClient
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from wine_quality.data_preparation import get_configurations
from wine_quality.features import get_train_test_ids
from wine_quality.features import get_training_testing_data
from wine_quality.model_func import create_mlflow_experiment
from wine_quality.model_func import get_pipeline

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

    experiment_name = configs["experiment_name"]
    create_mlflow_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name="wine_quality_classification") as run:
        pipeline = get_pipeline(columns=feature_names)
        pipeline_model = pipeline.fit(train_sdf)
        predictions = pipeline_model.transform(test_sdf)
        evaluator = BinaryClassificationEvaluator(
            labelCol="target",
            rawPredictionCol="prediction",
            metricName="areaUnderROC",
        )

        # get metrics
        roc = evaluator.evaluate(predictions)
        prc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)

        metrics = MulticlassMetrics(
            predictions.select("prediction", "target").rdd
        )
        precision = metrics.precision(1)
        recall = metrics.recall(1)

        mlflow.log_metric("roc", roc)
        mlflow.log_metric("prc", prc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.spark.log_model(
            pipeline_model,
            "pipeline",
            input_example=train_sdf.limit(5).toPandas(),
        )
        # logging model with mlflow
        fs.log_model(
            model=pipeline_model,
            artifact_path="fs_pipeline",
            flavor=mlflow.spark,
            training_set=train_set,
        )
