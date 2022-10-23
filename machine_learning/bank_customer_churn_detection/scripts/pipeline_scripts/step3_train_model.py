from azureml.core import Run
import os 
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--features', dest='features')
parser.add_argument('--label', dest='label')

args = parser.parse_args()
features = [feature for feature in args.features.split(",")]
label = args.label

run = Run.get_context()
training_data = run.input_datasets["training_data"].to_pandas_dataframe()

x_train, x_test, y_train, y_test = train_test_split(training_data[features],training_data[label], test_size=0.25, random_state=1)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

