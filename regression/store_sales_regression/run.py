import yaml
import pandas as pd

from store_sales_regression.utils.utils import read_source_data
from store_sales.regression.utils.utils import set_or_create_mlflow_experiment

from store_sales_regression.data.features import create_time_based_features
from store_sales_regression.data.features import aggregate_sales_data

from store_sales_regression.regression.model import train_model

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    with open("configs/data.yaml") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    df = read_source_data(config, "train")
    sales_data = aggregate_sales_data(df)
    features, feature_names = create_time_based_features(sales_data, "date")

    x_train, x_test, y_train, y_test = train_test_split(
        features[feature_names], features["sales"], test_size=0.2, random_state=42
    )

    experiment_id = set_or_create_mlflow_experiment("sales regression") 
    train_model(x_train, x_test, y_train, y_test, feature_names, experiment_id)
