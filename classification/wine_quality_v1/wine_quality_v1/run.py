from datetime import datetime

import mlflow
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from wine_quality_v1.data_preparation.data_preparation import get_wine_dataset_uci
from wine_quality_v1.data_preparation.data_preparation import read_config
from wine_quality_v1.data_preparation.data_preparation import remove_outlier
from wine_quality_v1.data_preparation.data_preparation import transform_to_binary

from wine_quality_v1.training.mlflow_utils import get_model 
from wine_quality_v1.training.mlflow_utils import get_or_create_experiment
from wine_quality_v1.training.optimization import optimize
from wine_quality_v1.training.pipelines import train
from wine_quality_v1.training.evaluation import get_classification_metrics
from wine_quality_v1.utils.utils import get_root_dir
from wine_quality_v1.utils.utils import read_pickle
from wine_quality_v1.utils.utils import read_set
import pickle
import pandas as pd

def data_processing():
    """
    Prepare the data for training.
    """
    config = read_config("data_preparation")
    df, metadata = get_wine_dataset_uci()
    df = remove_outlier(df=df, metadata=metadata)
    df = transform_to_binary(df=df, feature=metadata["target"])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)
    
    root_dir = get_root_dir()
    output_dir = root_dir / config["output_folder"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_dir / config["sets"]["train"], index=False)
    df_val.to_csv(output_dir / config["sets"]["val"], index=False)
    df_test.to_csv(output_dir / config["sets"]["test"], index=False)
    with open(output_dir / "dataset_info.pkl", "wb") as f:
        pickle.dump(metadata, f)


def main():
    config = read_config()
    metadata = read_pickle("output/dataset_info.pkl")
    train_df = read_set("train")
    val_df = read_set("val") 
    print(train_df.shape, val_df.shape)

    experiment_name = config["experiment_name"]
    exp_tags = config["tags"]
    alias = config["aliases"]["qa"]
    label = metadata["target"]
    registered_model_name = config["registered_model_name"]

    run_name = f"run-{str(datetime.now())}"

    feature_names = metadata["numerical_features"] + metadata["categorical_features"]

    x_train = train_df[feature_names]
    y_train = train_df[label]

    x_val = val_df[feature_names]
    y_val = val_df[label]

    experiment = get_or_create_experiment(experiment_name, exp_tags)
    best_params = optimize(
        experiment_id=experiment.experiment_id,
        numerical_feautres=metadata["numerical_features"],
        categorical_features=metadata["categorical_features"],
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
    )
    best_pipeline = train(
        params=best_params,
        numerical_features=metadata["numerical_features"],
        categorical_features=metadata["categorical_features"],
        x_train=x_train,
        y_train=y_train,
    )
    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment.experiment_id
    ) as run:
        print("Run ID:", run.info.run_id)
        mlflow.sklearn.log_model(best_pipeline, "best_model")
        model_version = mlflow.register_model(
            f"runs:/{run.info.run_id}/best_model",
            registered_model_name,
        )

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias,
            version=model_version.version,
        )

def evaluate():
    """
    Evaluate the model.
    """
    metadata = read_pickle("output/dataset_info.pkl")
    test_df = read_set("test")
    feature_names = metadata["numerical_features"] + metadata["categorical_features"]
    label = metadata["target"]

    x_test = test_df[feature_names]
    y_test = test_df[label]

    model = get_model(stage="qa")
    predictions = model.predict(x_test)

    metrics = get_classification_metrics(
        y_pred=predictions, y_true=y_test
    )

    print("METRICS")
    print(metrics)

    print(classification_report(y_test, predictions))
