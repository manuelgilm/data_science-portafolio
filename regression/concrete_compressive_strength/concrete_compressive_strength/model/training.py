from concrete_compressive_strength.data.retrieval import get_dataset
from concrete_compressive_strength.data.retrieval import process_column_names
from concrete_compressive_strength.model.pipelines import get_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def train():
    """
    Train the model and save it to disk.
    """
    # get the dataset
    df, metadata = get_dataset()

    target = metadata[metadata["role"]=="Target"]["name"].values[0]
    target = target.lower().replace(" ", "_")

    # process column names
    df = process_column_names(df)
    print(df.head())
    # get the numerical columns
    numerical_columns = df.columns.to_list()
    numerical_columns.remove(target)
    print(numerical_columns, target)
    # get the pipeline
    pipeline = get_pipeline(numerical_columns)

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=0)

    # fit the model
    pipeline.fit(X_train, y_train)

    # make predictions
    y_pred = pipeline.predict(X_test)

    # evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R2: {r2}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")

    
    