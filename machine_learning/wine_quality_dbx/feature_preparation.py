from wine_quality.data_preparation import get_dataframe, create_feature_table, get_configurations
from pyspark.sql import functions as F

red_wine_sdf = get_dataframe(
    spark, path="dbfs:/FileStore/datasets/wine_quality/winequality-red.csv"
)
white_wine_sdf = get_dataframe(
    spark, path="dbfs:/FileStore/datasets/wine_quality/winequality-white.csv"
)

# Create categories.
# 0 means bad quality and 1 means good quality
red_wine_sdf = red_wine_sdf.withColumn(
    "target", F.when(F.col("quality") < 6, 0).otherwise(1)
)
white_wine_sdf = white_wine_sdf.withColumn(
    "target", F.when(F.col("quality") < 6, 0).otherwise(1)
)

# let's use white wine dataset for training
def clean_raw_data(sdf):
    """
    This function performs the following operations:
    1. Drops the null values
    params: sdf: spark dataframe
    return: sdf: spark dataframe
    """
    sdf = sdf.dropna()
    sdf = sdf.withColumn("id", F.monotonically_increasing_id())
    for field in sdf.schema.fields:
        if " " in field.name:
            sdf = sdf.withColumnRenamed(field.name, field.name.replace(" ", "_"))
    return sdf


clean_data = clean_raw_data(white_wine_sdf)
configs = get_configurations(filename="common_params")
create_feature_table(
    sdf=clean_data,
    database_name=configs["database_name"],
    feature_table_name=configs["feature_table_name"],
    feature_table_path=configs["feature_table_path"],
    description=configs["description"],
)
