import mlflow

from pyspark.sql.functions import pandas_udf
from pyspark.sql import DataFrame
import pandas as pd 

def get_model(model_uri: str):
    """
    This function loads the model from the given model uri.

    params: model_uri: str
        The model uri.
    return: model: mlflow.spark.models.PythonModel()

    """
    return mlflow.pyfunc.spark_udf(model_uri=model_uri)


# @pandas_udf("double")
# def predict_udf(*args:DataFrame) -> DataFrame:
#     model_uri = 'runs:/ac8576dea85c45458cfb0ee7ddc9b000/pipeline'
#     model = get_model(model_uri=model_uri)
#     pdf = pd.concat(args, axis=1)
#     return pd.Series(model.predict(pdf))


if __name__=="__main__":

    fs = FeatureStoreClient()
    config = get_configurations(filename="feature_preparation")
    experiment_name = config["experiment_name"]
    categorical_columns, numerical_columns = get_feature_names(config=config)
    categorical_features = [col + "_indexed" for col in categorical_columns]
    numerical_features = [col + "_imputed" for col in numerical_columns]

    train_ids, test_ids = get_train_test_ids(config=config, fs=fs)
    training_set, testing_set = get_train_test_sets(
        config=config,
        fs=fs,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_names=categorical_columns + numerical_columns,
    )
    train_sdf = training_set.load_df()
    test_sdf = testing_set.load_df()

    model_uri =  'runs:/ac8576dea85c45458cfb0ee7ddc9b000/pipeline'
    model = get_model(model_uri=model_uri)

