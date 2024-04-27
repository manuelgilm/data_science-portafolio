import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from databricks.feature_store import FeatureStoreClient
from utils import util_functions as uf

def data_preparation(config: dict, sdf: DataFrame) -> DataFrame:
    """
    This function performs the following operations:
    1. Selects the columns to keep
    2. Removes the dollar sign from the price column and casts it to double type
    3. Casts all the numerical columns to double type
    4. Filters the prices greater than 0
    5. Filters the minimum nights less than 365

    params: config: dict
    params: sdf: spark dataframe
    return: sdf: spark dataframe

    """
    columns_to_keep = config["columns_to_keep"]
    sdf = sdf.select(columns_to_keep)
    sdf = sdf.withColumn("price", F.translate("price", "$", "").cast(DoubleType()))

    # Casting all the numerical columns to double type
    numerical_columns = [
        field.name
        for field in sdf.schema.fields
        if isinstance(field.dataType, IntegerType)
    ]
    for column in numerical_columns:
        sdf = sdf.withColumn(column, F.col(column).cast(DoubleType()))

    # filtering prices greater than 0
    sdf = sdf.filter(F.col("price") > 0)

    # filtering minimum nights less than 365
    sdf = sdf.filter(F.col("minimum_nights") < 365)
    return sdf


def create_feature_table(config: dict, fs: FeatureStoreClient, sdf: DataFrame) -> None:
    """Creates the feature table in delta lake format.

    params: config: dict
    params: fs: feature store client
    params: sdf: spark dataframe
    """
    feature_table_name = f"{config['database_name']}.{config['feature_table_name']}"
    fs.create_table(
        name=feature_table_name,
        df = sdf,
        primary_keys=["id"],
        schema=sdf.schema,
        description="Feature table for airbnb price prediction",
        partition_columns=["id"],
        path=config["feature_table_path"],
    )


if __name__ == "__main__":
    fs = FeatureStoreClient()
    config = uf.get_configurations("feature_preparation")
    sdf = uf.read_data(config)
    sdf = data_preparation(config, sdf)
    create_feature_table(config, fs, sdf)
