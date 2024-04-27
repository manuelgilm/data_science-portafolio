import argparse
import os

import pandas as pd
from azureml.core import Run

run = Run.get_context()
testing_data = run.input_datasets["testing_data"].to_pandas_dataframe()
print(testing_data.head())
