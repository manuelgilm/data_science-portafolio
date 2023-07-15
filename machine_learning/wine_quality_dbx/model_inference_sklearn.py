import mlflow
import pandas as pd
from databricks.feature_store import FeatureStoreClient

from wine_quality.features import get_train_test_ids, get_training_testing_data
from wine_quality.data_preparation import get_configurations
from wine_quality.model_func import create_mlflow_experiment, get_pipeline

from pyspark.sql import functions as F


@F.pandas_udf("double")
def predict_udf(*cols: pd.Series) -> pd.Series:
    model_uri = "runs:/5d08a259e6f44d6abc65b905bab06ce1/model"
    model = mlflow.sklearn.load_model(model_uri)
    pdf = pd.concat(cols, axis=1)
    return pd.Series(model.predict(pdf))


if __name__ == "__main__":
    fs = FeatureStoreClient()
    configs = get_configurations(filename="common_params")
    train_ids, test_ids, feature_names = get_train_test_ids(configs=configs)
    train_set, test_set = get_training_testing_data(
        configs=configs,
        feature_names=feature_names,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    test_sdf = test_set.load_df()
    test_df = test_sdf.toPandas()

    print(test_sdf.columns)
    print("FEATURE NAMES")
    feature_names = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    print(test_sdf.select(feature_names).show())
    # test_sdf = test_sdf.withColumn(
    #     "prediction", predict_udf(*feature_names)
    # )

    # # print(test_sdf.select("prediction").show())
    # model_uri = "runs:/5d08a259e6f44d6abc65b905bab06ce1/model"
    # model = mlflow.sklearn.load_model(model_uri)
    # predictions = model.predict(test_df[feature_names])
    # print(predictions)
