from azureml.core import Run
import argparse
import os 
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--filename',dest='dataset_name')
parser.add_argument('--output-folder',dest="folder")
parser.add_argument('--dataset', dest="dataset")

args = parser.parse_args()
run = Run.get_context()
dataset = run.input_datasets[args.dataset_name].to_pandas_dataframe()

print(dataset.head())


output_path = os.path.join(args.folder, "prepped_"+args.dataset_name)
dataset.to_csv(output_path)

