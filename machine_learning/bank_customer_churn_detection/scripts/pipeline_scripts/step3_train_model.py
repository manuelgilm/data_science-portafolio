from multiprocessing.connection import Pipe
from random import Random
from azureml.core import Run
import os 
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


parser = argparse.ArgumentParser()
parser.add_argument('--features', dest='features')
parser.add_argument('--label', dest='label')

args = parser.parse_args()
features = [feature for feature in args.features.split(",")]
label = args.label

run = Run.get_context()
training_data = run.input_datasets["training_data"].to_pandas_dataframe()
print("BEFORE SPLIT")
print(training_data.head())
print(training_data.columns)
print(training_data.isnull().sum())
print("AFTER SPLIT")
x_train, x_test, y_train, y_test = train_test_split(training_data[features],training_data[label], test_size=0.25, random_state=1)
print(x_train.head())
print("X_TRAIN")
print(x_train.isnull().sum())

print("X_TEST")
print(x_test.isnull().sum())

print("Y_TRAIN")
print(y_train.isnull().sum())

print("Y_TEST")
print(y_test.isnull().sum())
# create pipeline
numeric_columns = features
numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0)
preprocessor = ColumnTransformer(
    transformers = [("num", numeric_imputer, numeric_columns)]
)

pipeline = Pipeline([
    ('preprocessing',preprocessor),
    ('classifier',RandomForestClassifier())
    ])
# training random forest classifier
pipeline.fit(x_train[features], y_train)

# getting predictions
predictions = pipeline.predict(x_test)

print(predictions)




