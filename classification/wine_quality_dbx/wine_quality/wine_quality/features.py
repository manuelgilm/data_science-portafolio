from databricks.feature_store import FeatureLookup
from databricks.feature_store import FeatureStoreClient


def get_train_test_ids(configs: dict):
    """
    This function returns the train and test ids from the feature table.

    params: configs: dict
    return: train_ids: spark dataframe
    return: test_ids: spark dataframe
    """
    database_name = configs["database_name"]
    feature_table_name = configs["feature_table_name"]
    feature_table = f"{database_name}.{feature_table_name}"

    fs = FeatureStoreClient()
    feature_table = fs.read_table(feature_table)
    feature_names = [
        field.name
        for field in feature_table.schema.fields
        if field.name != "target"
        and field.name != "id"
        and field.name != "quality"
    ]
    train_ids, test_ids = feature_table.select("id", "target").randomSplit(
        [0.8, 0.2], seed=42
    )
    return train_ids, test_ids, feature_names


def get_training_testing_data(configs, feature_names, train_ids, test_ids):

    fs = FeatureStoreClient()

    database_name = configs["database_name"]
    feature_table_name = configs["feature_table_name"]
    feature_table = f"{database_name}.{feature_table_name}"

    feature_lookup = FeatureLookup(
        table_name=feature_table,
        feature_names=feature_names,
        lookup_key=["id"],
    )

    training_set = fs.create_training_set(
        train_ids,
        feature_lookups=[feature_lookup],
        label="target",
        exclude_columns=["id"],
    )

    testing_set = fs.create_training_set(
        test_ids,
        feature_lookups=[feature_lookup],
        label=None,
        exclude_columns=["id"],
    )

    return training_set, testing_set
