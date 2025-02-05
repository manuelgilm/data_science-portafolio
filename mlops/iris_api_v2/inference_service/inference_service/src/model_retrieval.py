import mlflow


def load_latest_model(model_name: str) -> mlflow.pyfunc.PyFunc:
    """
    Loads the latest model with alias "Champion"

    :return: The latest model with alias "Champion"
    """
    model_uri = f"models:/{model_name}@Champion"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model

    except Exception as e:
        print(f"Model not found at {model_uri}")
