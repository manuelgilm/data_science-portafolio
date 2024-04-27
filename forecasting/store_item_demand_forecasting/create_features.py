import os
import sys

from demand_forecasting.feature_processing import create_features_from_pandas_dataframe
from demand_forecasting.utils import create_schema
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if __name__ == "__main__":
    print("Running entrypoint...")
    spark = SparkSession.builder.appName(
        "Store_item_demand_forecasting"
    ).getOrCreate()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    training_data = spark.read.format("parquet").load(
        "data/preprocessed/training.parquet"
    )

    # create schema from small sample
    pdf_sample = training_data.sample(
        withReplacement=False, fraction=0.01, seed=42
    ).toPandas()
    dummy_features = create_features_from_pandas_dataframe(pdf_sample)
    schema = create_schema(dummy_features, spark)

    # create features only for 2 stores and 5 items
    training_data = training_data.where("store <= 2 and item <= 5")
    features = training_data.groupBy("date", "store", "item").applyInPandas(
        func=create_features_from_pandas_dataframe, schema=schema
    )

    # save features
    features.write.format("parquet").mode("overwrite").save(
        "data/features/features.parquet"
    )
