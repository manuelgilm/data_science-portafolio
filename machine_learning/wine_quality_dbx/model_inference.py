import mlflow 
from databricks.feature_store import FeatureStoreClient

from wine_quality.features import get_train_test_ids, get_training_testing_data
from wine_quality.data_preparation import get_configurations

def get_model(model_uri:str):
    """
    This function loads the model from the specified model uri.
    params: model_uri: str
    return: model: spark model
    """
    return mlflow.spark.load_model(model_uri=model_uri)

if __name__=="__main__":

    model_uri_fs = 'runs:/83356f31d0ba4652879dbd19373a45aa/fs_pipeline'
    model_uri = 'runs:/83356f31d0ba4652879dbd19373a45aa/pipeline'
    configs = get_configurations(filename="common_params")
    train_ids, test_ids, feature_names = get_train_test_ids(configs=configs)

    # Inference with feature store
    fs = FeatureStoreClient()
    
    # print(test_ids.show())
    prediction = fs.score_batch(
        model_uri=model_uri_fs,
        df=test_ids,
    )

    print("INFERENCE WITH FEATURE STORE")
    print(prediction.show())

    train_set, test_set = get_training_testing_data(
        configs=configs,
        feature_names=feature_names,
        train_ids=train_ids,
        test_ids=test_ids,
    )

    # Inference with spark model
    model = get_model(model_uri=model_uri)
    prediction = model.transform(test_set.load_df())
    print("INFERENCE WITH SPARK MODEL")
    print(prediction.select("prediction","rawPrediction","probability","target").show())
    
    

