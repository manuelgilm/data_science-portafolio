from azureml.core import Run
import argparse
import os 
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--filename',dest='dataset_name')
parser.add_argument('--preprocessed-data',dest="folder")
parser.add_argument('--dataset', dest="dataset")

args = parser.parse_args()
run = Run.get_context()
dataset = run.input_datasets[args.dataset_name].to_pandas_dataframe()

def transform_categorical_features(df, columns):
    '''Applies one hot encoder to categorical features'''
    df = pd.get_dummies(data=df, columns=columns)
    return df

categorical_columns =  ["gender"]
dataset = transform_categorical_features(dataset,categorical_columns)


if not (args.folder is None):
    os.makedirs(args.folder, exist_ok=True)
    output_path = os.path.join(args.folder, "prepped_"+args.dataset_name+".csv")
    print(args)
    print(output_path)
    dataset.to_csv(output_path)

