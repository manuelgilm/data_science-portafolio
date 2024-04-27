from azureml.core import Dataset, Workspace
from azureml.data.dataset_factory import FileDatasetFactory
from dotenv import load_dotenv
import os
load_dotenv(".env")

SUBSCRIPTION_ID = os.environ["SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ["RESOURCE_GROUP"]
WORKSPACE_NAME = os.environ["WORKSPACE_NAME"]

def register_dataset(workspace,data_storage, data_path, dataset_name):
    ''''''
    path = [(data_storage, data_path)]
    tabular_dataset = Dataset.Tabular.from_delimited_files(path=path)
    tabular_dataset.register(workspace=workspace, name=dataset_name)

def upload_file_to_datastore(datastore,local_path):
    target_path = (datastore, "data")
    dataset_factory = FileDatasetFactory()
    dataset_factory.upload_directory(local_path,target=target_path)

def get_workspace():
    ''''''
    ws = Workspace(
        subscription_id = SUBSCRIPTION_ID,
        resource_group = RESOURCE_GROUP,
        workspace_name = WORKSPACE_NAME,
        )
    return ws