from azureml.core import Run
import os 
import argparse
import pandas as pd


run = Run.get_context()
testing_data = run.input_datasets["testing_data"].to_pandas_dataframe()
print(testing_data.head())

