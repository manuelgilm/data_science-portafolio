from typing import Callable
from typing import Optional
from typing import Union

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features from a Spark DataFrame.

    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    # create features based on date
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["dayofyear"] = df["date"].dt.dayofyear

    return df


def create_aggregating_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on aggregations.

    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    # create features based on rolling windos
    windows = [1, 7, 14, 28]
    for w in windows:
        # create features based on lags
        df[f"lag_{w}"] = df["sales"].shift(w)
        # average sales
        df[f"avg_sales_{w}"] = df["sales"].rolling(w, min_periods=1).mean()
        # std sales
        df[f"std_sales_{w}"] = df["sales"].rolling(w, min_periods=1).std()
        # min sales
        df[f"min_sales_{w}"] = df["sales"].rolling(w, min_periods=1).min()
        # max sales
        df[f"max_sales_{w}"] = df["sales"].rolling(w, min_periods=1).max()
        # median sales
    return df


def create_features_from_pandas_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features from a Spark DataFrame.

    :param df: pandas DataFrame
    :return: pandas DataFrame
    """

    # Aggregate all sales per item and store
    df["date"] = pd.to_datetime(df["date"])
    df = create_date_features(df)
    df = create_aggregating_features(df)
    df = df.fillna(0)

    return df


def create_feature_from_spark_dataframe(sdf: DataFrame) -> DataFrame:
    """
    Create feature using spark functions.

    :param sdf: Spark DataFrame
    :return: Spark DataFrame
    """

    # create features based on date
    sdf = sdf.withColumn("sales", F.col("sales").cast("double"))
    sdf = sdf.withColumn("year", F.year("date"))
    sdf = sdf.withColumn("month", F.month("date"))
    sdf = sdf.withColumn("day", F.dayofmonth("date"))
    sdf = sdf.withColumn("dayofweek", F.dayofweek("date"))
    sdf = sdf.withColumn("quarter", F.quarter("date"))
    sdf = sdf.withColumn("dayofyear", F.dayofyear("date"))

    windows = [1, 7, 14, 28]
    window = Window.partitionBy("date", "store", "item").orderBy("date")
    for w in windows:
        # features based on lags
        sdf = sdf.withColumn(f"lag_{w}", F.lag("sales", w).over(window))
        # average sales
        sdf = sdf.withColumn(f"avg_sales_{w}", F.avg("sales").over(window))
        # std sales
        sdf = sdf.withColumn(f"std_sales_{w}", F.stddev("sales").over(window))
        # min sales
        sdf = sdf.withColumn(f"min_sales_{w}", F.min("sales").over(window))
        # max sales
        sdf = sdf.withColumn(f"max_sales_{w}", F.max("sales").over(window))

    sdf = sdf.fillna(0)

    return sdf


def create_features_using_apply_in_pandas(
    sdf: DataFrame, func: Callable, schema: str, keys: list
) -> DataFrame:
    """
    Creates features using applyInPandas.

    :param sdf: Spark DataFrame
    :param schema: schema of the Spark DataFrame
    :param keys: keys to group by
    :return: Spark DataFrame
    """

    features = sdf.groupBy(*keys).applyInPandas(func, schema=schema)

    return features
