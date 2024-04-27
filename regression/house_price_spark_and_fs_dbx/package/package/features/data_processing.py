import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.datasets import fetch_california_housing


def load_data() -> pd.DataFrame:
    """
    Download the California housing dataset and return it as a pandas dataframe.

    :return: California housing dataset as a pandas dataframe.
    """
    data = fetch_california_housing(
        data_home="package/data/", as_frame=True, download_if_missing=False
    )
    return data.frame


def get_feature_dataframe() -> DataFrame:
    """
    Get the feature dataframe.

    :return: Feature dataframe.
    """
    spark = SparkSession.getActiveSession()
    pdf = load_data()
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("id", F.monotonically_increasing_id())
    return sdf
