import subprocess

from wine_quality.data_preparation import get_configurations

configs = get_configurations(filename="common_params")

spark.sql(
    f"DROP TABLE IF EXISTS {configs['database_name']}.{configs['feature_table_name']}"
)

subprocess.run(
    ["databricks", "fs", "rm", "-r", f"{configs['feature_table_path']}"]
)
