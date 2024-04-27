from typing import List
from typing import Tuple

from databricks.feature_store import FeatureLookup
from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.training_set import TrainingSet
from pyspark.sql import DataFrame


def get_train_testing_sets(
    database_name: str,
    table_name: str,
    feature_names: List[str],
    label: str,
    train_ids: DataFrame,
    test_ids: DataFrame,
) -> Tuple[TrainingSet, TrainingSet]:
    """
    Get training and testing sets from the feature store.

    :param database_name: Name of the database.
    :param table_name: Name of the table.
    :param feature_names: List of feature names.
    :param label: Label.
    :param train_ids: Training IDs.
    :param test_ids: Testing IDs.
    :return: Training and testing sets.
    """
    table = f"{database_name}.{table_name}"
    feature_lookup = FeatureLookup(
        table_name=table,
        feature_names=feature_names,
        lookup_key=["id"],
    )

    fs = FeatureStoreClient()
    training_set = fs.create_training_set(
        df=train_ids,
        feature_lookups=[feature_lookup],
        label=label,
        exclude_columns=["id"],
    )

    testing_set = fs.create_training_set(
        df=test_ids,
        feature_lookups=[feature_lookup],
        label=label,
        exclude_columns=["id"],
    )

    return training_set, testing_set


def get_train_test_ids(
    database_name: str, table_name: str, label: str
) -> Tuple[DataFrame, DataFrame]:
    """
    Get training and testing IDs from the feature store.

    :param database_name: Name of the database.
    :param table_name: Name of the table.
    :param label: Label.
    :return: Training and testing IDs.
    """

    fs = FeatureStoreClient()
    sdf = fs.read_table(f"{database_name}.{table_name}")
    sdf = sdf.select("id", label)
    train_ids, test_ids = sdf.randomSplit([0.8, 0.2], seed=42)
    return train_ids, test_ids
