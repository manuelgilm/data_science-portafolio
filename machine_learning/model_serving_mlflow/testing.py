from training_package.training import get_train_test_data
# from training_package.training import get_processing_pipeline
# from training_package.training import get_regression_metrics

# from training_package.mlflow_utils import create_or_set_experiment
# from training_package.mlflow_utils import get_custom_signature
from mlflow.models import infer_signature
from mlflow.models import ModelSignature
from mlflow.types import ParamSchema
from mlflow.types import ParamSpec

import pandas as pd

def get_custom_signature(
    x: pd.DataFrame, y: pd.DataFrame, params: dict
) -> ModelSignature:
    """
    Get signature for custom model.

    :param x: features
    :param y: target
    :param params: model parameters

    :return: signature
    """
    data_schema = infer_signature(x, y)
    
    type_map = {"str":"string", "int":"integer", "float":"float", "bool":"boolean"}

    params_spec = [
        ParamSpec(name=k, dtype=type_map[type(v).__name__], default=None)
        for k, v in params.items()
    ]
    params_schema = ParamSchema(params_spec)
    signature = ModelSignature(
        inputs=data_schema.inputs, outputs=data_schema.outputs, params=params_schema
    )

    return signature

if __name__=="__main__":
    x_train, x_test, y_train, y_test = get_train_test_data(n_features = 20)

    numerical_features = x_train.columns
    params = {"n_features": 20, "model_name":"regressor 1"}

    signature = get_custom_signature(x_train, y_train, params)
    print(signature)