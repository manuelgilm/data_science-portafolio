import mlflow 
from databricks.feature_store import FeatureStoreClient

from wine_quality.features import get_train_test_ids, get_training_testing_data
from wine_quality.data_preparation import get_configurations
from wine_quality.model_func import create_mlflow_experiment, get_pipeline

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

if __name__=="__main__":

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

    evaluator = BinaryClassificationEvaluator(
        labelCol="target", rawPredictionCol="prediction", metricName="areaUnderROC"
    )

    pipeline = get_pipeline(columns=feature_names)
    params = ParamGridBuilder().addGrid(pipeline.getStages()[-1].maxDepth, [2, 5]).build()
    
    cv_pipeline = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=params,
        evaluator=evaluator,
        numFolds=2,
    )

    with mlflow.start_run(run_name="wine_quality_classification") as run:
        
        cv_pipeline_model = cv_pipeline.fit(train_sdf)
        predictions = cv_pipeline_model.transform(test_sdf)
        

        # get metrics
        roc = evaluator.evaluate(predictions)
        prc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)

        metrics = MulticlassMetrics(predictions.select("prediction", "target").rdd)
        precision = metrics.precision(1)
        recall = metrics.recall(1)

        mlflow.log_metric("roc", roc)
        mlflow.log_metric("prc", prc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.spark.log_model(
            cv_pipeline_model, "cv_pipeline", input_example=train_sdf.limit(5).toPandas()
        )
        # logging model with mlflow
        fs.log_model(
            model=cv_pipeline_model,
            artifact_path="fs_cv_pipeline",
            flavor=mlflow.spark,
            training_set=train_set,
        )
