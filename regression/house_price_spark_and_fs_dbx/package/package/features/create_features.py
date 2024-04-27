from databricks.feature_store import FeatureStoreClient
from pyspark.sql import DataFrame


def create_feature_table(table_name: str, database_name: str, sdf: DataFrame):
    """
    Creates a feature table in the feature store.

    :param table_name: Name of the feature table.
    :param database_name: Name of the database.
    :param sdf: Spark dataframe.
    """

    fs = FeatureStoreClient()

    fs.create_table(
        name=f"{database_name}.{table_name}",
        primary_keys=["id"],
        schema=sdf.schema,
        df=sdf,
        description="California housing dataset",
    )
