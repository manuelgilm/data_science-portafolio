from databricks.feature_store import FeatureStoreClient

table_name = "default.airbnb_feature_table"
spark.sql(f"DROP TABLE IF EXISTS {table_name}".format(table_name))

