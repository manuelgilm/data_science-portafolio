from databricks.feature_store import FeatureStoreClient
from utils import util_functions as uf

if __name__ == "__main__":

    fs = FeatureStoreClient()
    config = uf.get_configurations(filename="feature_preparation")
    experiment_name = config["experiment_name"]
    categorical_columns, numerical_columns = uf.get_feature_names(
        config=config
    )
    categorical_features = [col + "_indexed" for col in categorical_columns]
    numerical_features = [col + "_imputed" for col in numerical_columns]

    _, test_ids = uf.get_train_test_ids(config=config, fs=fs)

    model_uri = "runs:/b4c6188e8fc343c79162f90b7e61f1ad/fs_cv_pipeline"

    # When using feature store we only need to pass the ids to the score_batch function.
    # The feature store will take care of the rest.
    predictions = fs.score_batch(
        model_uri=model_uri,
        df=test_ids.select("id"),
    )
    print(predictions.show())
