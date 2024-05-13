from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow 


def get_train_dataset()->pd.DataFrame:
    """
    Gets the training dataset
    """

    x, y = make_classification(n_features=5)
    df = pd.DataFrame(x , columns=[f"feature_{n}" for n in range(x.shape[1])])
    df["target"] = y

    return df



if __name__ =="__main__":
    df = get_train_dataset()
    clf = RandomForestClassifier()
    feature_names = [f for f in df.columns if f != "target"]

    clf.fit(df[feature_names], df["target"])

    with mlflow.start_run(run_name="dummy_model") as run:
        mlflow.sklearn.log_model(sk_model=clf, artifact_path="classifier", registered_model_name="DummyModel")




