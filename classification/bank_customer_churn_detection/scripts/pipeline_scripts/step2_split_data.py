from azureml.core import Run
import argparse
import pandas 
import os 
from azureml.data import OutputFileDatasetConfig
from sklearn.model_selection import train_test_split
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--filename',dest='dataset_name')
parser.add_argument('--output-training-data', dest="output_training_data")
parser.add_argument('--output-testing-data', dest="output_testing_data")
parser.add_argument('--features', dest='features')
parser.add_argument('--label', dest='label')

args = parser.parse_args()
run = Run.get_context()
print(args)
dataset = run.input_datasets["preprocessed_data"].to_pandas_dataframe()
features = [feature for feature in args.features.split(",")]
print("FEATURES")
print(features)
label = args.label

print("DATASET")
print(dataset.head())

x_train, x_test, y_train, y_test = train_test_split(dataset[features],dataset[label], test_size=0.25, random_state=1)

training_dataset = pd.concat([x_train, y_train], axis=1)
testing_dataset = pd.concat([x_test, y_test], axis=1)

if not (args.output_training_data is None):
    os.makedirs(args.output_training_data, exist_ok=True)
    training_output_path = os.path.join(args.output_training_data, "training_"+args.dataset_name+".csv")
    dataset.to_csv(training_output_path)

if not (args.output_testing_data is None):
    os.makedirs(args.output_testing_data, exist_ok=True)
    testing_output_path = os.path.join(args.output_testing_data, "testing_"+args.dataset_name+".csv")
    dataset.to_csv(testing_output_path)

