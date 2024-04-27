import os
from typing import Optional
from typing import Union

import pandas as pd
import yaml
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def read_yaml(filepath: str) -> dict:
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return config


def read_csv(
    filepath: str, spark: SparkSession
) -> Optional[Union[DataFrame, pd.DataFrame]]:
    """
    Retrieves a csv file from a specified filepath and returns a DataFrame.

    params: filepath: location of data
    params: spark: SparkSession object

    returns: DataFrame
    """

    if spark:
        return spark.read.csv(filepath, header=True, inferSchema=True)
    else:
        return pd.read_csv(filepath)


def read_source_table(
    table_name: str,
    configs: dict,
    env: str = "local",
    spark: SparkSession = None,
) -> Optional[Union[DataFrame, pd.DataFrame]]:
    """"""
    data_base_path = configs["env"][env]["source"]
    filepath = os.path.join(
        data_base_path, configs["source_tables"][table_name]
    )
    df = read_csv(filepath=filepath, spark=spark)
    return df


def get_column_names(df: DataFrame) -> list:
    """
    Retrieves the column names of a DataFrame.

    params: df: DataFrame

    returns: list
    """
    return list(df.columns)


def create_schema(df: DataFrame, spark: SparkSession) -> str:
    """
    Creates a schema for a DataFrame.

    :param df: DataFrame
    :param spark: SparkSession object

    returns: schema
    """
    sdf = spark.createDataFrame(df)
    schema = sdf.schema
    return schema


def create_training_and_testing_datasets(
    start_training: str, start_testing: str, sdf: DataFrame
) -> DataFrame:
    """
    Splits spark dataframe into training and testing data given a time interval.

    :param start_training: start date of training data
    :param start_testing: start date of testing data
    :param sdf: spark DataFrame
    :return: spark DataFrame
    """

    training_data = sdf.filter(F.col("date") < start_testing)
    testing_data = sdf.filter(F.col("date") >= start_testing)

    return training_data, testing_data
