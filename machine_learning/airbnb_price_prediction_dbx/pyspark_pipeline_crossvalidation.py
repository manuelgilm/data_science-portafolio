import mlflow
from databricks.feature_store import FeatureStoreClient
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from utils import util_functions as uf

if __name__ == "__main__":
    fs = FeatureStoreClient()
    config = uf.get_configurations(filename="feature_preparation")
    experiment_name = config["experiment_name"]
    categorical_columns, numerical_columns = uf.get_feature_names(config=config)
    categorical_features = [col + "_indexed" for col in categorical_columns]
    numerical_features = [col + "_imputed" for col in numerical_columns]

    train_ids, test_ids = uf.get_train_test_ids(config=config, fs=fs)
    training_set, testing_set = uf.get_train_test_sets(
        config=config,
        fs=fs,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_names=categorical_columns + numerical_columns,
    )
    train_sdf = training_set.load_df()
    test_sdf = testing_set.load_df()
    model_pipeline = uf.get_pipeline(
        config=config,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(model_pipeline.getStages()[-1].maxBins, [50, 100])
        .addGrid(model_pipeline.getStages()[-1].maxDepth, [5, 10])
        .build()
    )
    # get the evaluation metrics
    evaluator = RegressionEvaluator(
        labelCol="price", predictionCol="prediction", metricName="rmse"
    )

    cv = CrossValidator(
        estimator=model_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=RegressionEvaluator(labelCol="price", predictionCol="prediction"),
        numFolds=3,
    )
    uf.set_or_create_mlflow_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name="airbnb_price_prediction"):
        cv_model = cv.fit(train_sdf)
        predictions = cv_model.transform(test_sdf)

        rmse = evaluator.evaluate(predictions)
        r2 = evaluator.setMetricName("r2").evaluate(predictions)

        # log the evaluation metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # log the model
        mlflow.spark.log_model(
            cv_model, "cv_pipeline", input_example=train_sdf.limit(5).toPandas()
        )

        # log model using feature store
        fs.log_model(
            model=cv_model,
            artifact_path="fs_cv_pipeline",
            flavor=mlflow.spark,
            training_set=training_set,
        )
