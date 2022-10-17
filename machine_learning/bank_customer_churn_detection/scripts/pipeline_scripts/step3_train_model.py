from azureml.core import Run
import os 
import argparse
import pandas as pd

run = Run.get_context()
training_data = run.input_datasets["training_data"].to_pandas_dataframe()

print(training_data.head())
# training_data = pd.read_csv(os.path.join(args.training_data,"training_"+args.filename))
# print(training_data.head())
# print(training_data.columns)