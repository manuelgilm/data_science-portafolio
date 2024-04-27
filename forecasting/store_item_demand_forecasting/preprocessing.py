import os
import sys

from demand_forecasting.utils import create_schema
from demand_forecasting.utils import create_training_and_testing_datasets
from demand_forecasting.utils import get_column_names
from demand_forecasting.utils import read_csv
from demand_forecasting.utils import read_source_table
from demand_forecasting.utils import read_yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if __name__ == "__main__":
    print("Running entrypoint...")
    spark = SparkSession.builder.appName(
        "Store_item_demand_forecasting"
    ).getOrCreate()
    env = "local"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    config_path = "configs/data.yaml"
    configs = read_yaml(config_path)
    data_path = os.path.join(
        configs["env"][env]["source"], configs["source_tables"]["train"]
    )
    data = read_source_table(table_name="train", configs=configs, spark=spark)
    data = data.withColumn("date", F.to_date(F.col("date")))
    aggregated_sales = (
        data.groupBy("date", "store", "item")
        .sum("sales")
        .withColumnRenamed("sum(sales)", "sales")
        .orderBy("date", "store", "item")
    )

    training_dataset, testing_dataset = create_training_and_testing_datasets(
        start_training="2014-01-01",
        start_testing="2017-12-31",
        sdf=aggregated_sales,
    )

    # save training and testing datasets
    training_dataset.write.format("parquet").mode("overwrite").save(
        "data/preprocessed/training.parquet"
    )
    testing_dataset.write.format("parquet").mode("overwrite").save(
        "data/preprocessed/testing.parquet"
    )

    print("TRAINING DATA RANGE")
    print(training_dataset.select(F.min("date"), F.max("date")).show())

    print("TESTING DATA RANGE")
    print(testing_dataset.select(F.min("date"), F.max("date")).show())

    # some analysis
    # get average sales per item
    avg_sales_per_item = (
        training_dataset.groupBy("item")
        .avg("sales")
        .withColumnRenamed("avg(sales)", "avg_sales")
        .orderBy("item")
    )
    print("AVERAGE SALES PER ITEM")
    print(avg_sales_per_item.show())

    # get average sales per store
    avg_sales_per_store = (
        training_dataset.groupBy("store")
        .avg("sales")
        .withColumnRenamed("avg(sales)", "avg_sales")
        .orderBy("store")
    )
    print("AVERAGE SALES PER STORE")

    # get average sales per store and month
    avg_sales_per_store_month = (
        training_dataset.withColumn("month", F.month("date"))
        .groupBy("store", "month")
        .avg("sales")
        .withColumnRenamed("avg(sales)", "avg_sales")
        .orderBy("store", "month")
    )
    print("AVERAGE SALES PER STORE AND MONTH")
    print(avg_sales_per_store_month.show())

    # get average sales per store and month and year
    avg_sales_per_store_month_year = (
        training_dataset.withColumn("month", F.month("date"))
        .withColumn("year", F.year("date"))
        .groupBy("store", "month", "year")
        .avg("sales")
        .withColumnRenamed("avg(sales)", "avg_sales")
        .orderBy("store", "month", "year")
    )
    print("AVERAGE SALES PER STORE AND MONTH AND YEAR")
    print(avg_sales_per_store_month_year.show())
