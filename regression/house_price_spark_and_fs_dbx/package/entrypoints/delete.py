from pyspark.sql import SparkSession
from package.configs.utils import get_configs
import mlflow

if __name__ == "__main__":
    configs = get_configs()
    model_name = configs["model_name"]
    database_name = configs["database_name"]
    table_name = configs["table_name"]
    experiment_name = configs["experiment_name"]

    spark = SparkSession.getActiveSession()
    spark.sql(f"DROP TABLE IF EXISTS {database_name}.{table_name}")
    mlflow.MlflowClient().delete_registered_model(name=model_name)

    dbutils.fs.rm("dbfs:/user/hive/warehouse/california_housing", True)

    experiments = [experiment_name, "/Shared/dbx/projects/package"]
    for experiment in experiments:
        experiment_id = mlflow.get_experiment_by_name(name=experiment).experiment_id
        mlflow.delete_experiment(experiment_id=experiment_id)
