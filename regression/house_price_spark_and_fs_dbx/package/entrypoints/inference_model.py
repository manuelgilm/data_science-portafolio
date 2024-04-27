from databricks.feature_store import FeatureStoreClient
from package.configs.utils import get_configs
from package.training.retrieval import get_train_test_ids

if __name__ == "__main__":
    configs = get_configs()
    model_name = configs["model_name"]
    target = configs["target"]

    train_sdf, test_sdf = get_train_test_ids(
        database_name=configs["database_name"],
        table_name=configs["table_name"],
        label=target,
    )
    test_sdf_id = test_sdf.select("id", target)
    model_uri = f"models:/{model_name}/latest"
    fs = FeatureStoreClient()

    predictions = fs.score_batch(model_uri, test_sdf)
    print(predictions.select(target, "prediction").show(20))
