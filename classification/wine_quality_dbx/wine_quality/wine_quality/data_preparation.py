import yaml
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def get_dataframe(spark, path: str) -> DataFrame:
    """
    This function reads the csv file from the specified path and returns a spark dataframe.
    params: spark: spark session
    params: path: path to the csv file
    return: sdf: spark dataframe
    """
    sdf = spark.read.csv(path, sep=";", header=True, inferSchema=True)
    return sdf


def get_configurations(filename: str) -> dict:
    """Reads the configuration file from dbfs and returns a dictionary.

    params: filename: str
    return: config: dict
    """
    with open(
        f"/dbfs/FileStore/configs/wine_quality/{filename}.yaml", "rb"
    ) as f:
        print(f"Reading the configuration file from dbfs: {filename}.yaml")
        config = yaml.load(f)
    return config


def create_feature_table(
    sdf: DataFrame,
    database_name: str,
    feature_table_name: str,
    feature_table_path: str,
    description: str,
) -> None:
    """
    This function creates the feature table in delta lake format.
    params: sdf: spark dataframe
    """

    fs = FeatureStoreClient()

    fs.create_table(
        name=f"{database_name}.{feature_table_name}",
        df=sdf,
        schema=sdf.schema,
        primary_keys=["id"],
        description=description,
        path=feature_table_path,
    )
