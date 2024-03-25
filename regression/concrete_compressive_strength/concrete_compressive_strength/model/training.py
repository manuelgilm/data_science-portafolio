import pandas as pd
from concrete_compressive_strength.data.retrieval import get_dataset
from concrete_compressive_strength.data.retrieval import process_column_names
from concrete_compressive_strength.model.base import CustomPandasDataset
from concrete_compressive_strength.model.base import Regressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train():
    """
    Train the model and save it to disk.
    """
    # get the dataset
    df, metadata = get_dataset()

    target = metadata[metadata["role"] == "Target"]["name"].values[0]
    target = target.lower().replace(" ", "_")

    # process column names
    df = process_column_names(df)
    print(df.head())
    # get the numerical columns
    numerical_columns = df.columns.to_list()
    numerical_columns.remove(target)
    print(numerical_columns, target)
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), df[target], test_size=0.2, random_state=0
    )

    eval_data = pd.DataFrame(X_test)
    eval_data[target] = y_test

    # Training Baseline: Random Forest Regressor
    experiment_name = "concrete_compressive_strength"
    baseline_regressor = Regressor(
        regressor=RandomForestRegressor, experiment_name=experiment_name
    )
    baseline_regressor.fit(X_train, y_train, numerical_columns)
    baseline_regressor.evaluate(eval_data, target)
    baseline_model_uri = baseline_regressor.model_uri
    run_id = baseline_regressor.run_id

    dataset_name = "training_dataset"
    training_dataset_df = X_train.copy()
    training_dataset_df[target] = y_train
    training_dataset = CustomPandasDataset(
        df=training_dataset_df, source=None, targets=target, name=dataset_name
    )
    training_dataset.log_dataset(run_id=run_id)

    # Training other regressors
    regressors = [
        AdaBoostRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
        BaggingRegressor,
        LinearRegression,
    ]

    for regressor in regressors:
        regressor = Regressor(
            regressor=regressor, experiment_name=experiment_name
        )
        regressor.fit(X_train, y_train, numerical_columns)
        regressor.evaluate(
            eval_data, target, baseline_model_uri=baseline_model_uri
        )
