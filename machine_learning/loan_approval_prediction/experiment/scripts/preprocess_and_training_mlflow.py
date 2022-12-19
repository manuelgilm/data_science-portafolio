from azureml.core import Run
import azureml
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)

run = Run.get_context()

ws = run.experiment.workspace

df = ws.datasets.get("loan_data").to_pandas_dataframe()

### DATA PREPROCESSING
categorical_columns = ["Gender", "Married", "Education", "Dependents", "Property_Area"]
numerical_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
label = "Loan_Status"

categorical_data_prep = pd.get_dummies(df[categorical_columns])
numerical_columns_prep = df[numerical_columns].fillna(0)

df_prep = pd.concat([categorical_data_prep, numerical_columns_prep, df[label]], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df_prep, df[label], test_size=0.25)

lclf = LogisticRegression()

lclf.fit(x_train, y_train)

predictions = lclf.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Logging metrics with mlflow
mlflow_uri = azureml.mlflow.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.create_experiment("mlflow-experiment")
mlflow.set_experiment("mlflow-experiment")

with mlflow.start_run() as mlflow_run:
    mlflow_run.log_metrics(
        {
            "accuracy2": accuracy,
            "recall2": recall,
            "precision2": precision,
            "f1_score2": f1,
        }
    )

run.complete()
