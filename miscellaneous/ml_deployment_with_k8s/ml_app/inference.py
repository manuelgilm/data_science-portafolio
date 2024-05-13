import json 
import requests 
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import classification_report

def get_train_dataset()->pd.DataFrame:
    """
    Gets the training dataset
    """

    x, y = make_classification(n_samples=1, n_features=5)
    df = pd.DataFrame(x , columns=[f"feature_{n}" for n in range(x.shape[1])])
    df["target"] = y

    return df

df = get_train_dataset()
features = [c for c in df.columns if c != "target"]
# print(df[features].to_dict(orient="split"))
data = {
    "dataframe_split":df[features].to_dict(orient="split")
}
print(data)
# headers = {"Content-Type":"application/json"}
# endpoint = "http://0.0.0.0:5000/invocations"

# response = requests.post(endpoint, data = json.dumps(data), headers=headers)

# print(response.status_code)
# print(type(response.status_code))
# print(response.json())

# predictions = response.json()["predictions"]
# report = classification_report(y_true = df["target"], y_pred=predictions, output_dict=True)
# print(report)