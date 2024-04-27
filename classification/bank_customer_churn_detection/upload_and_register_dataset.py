import os

from dotenv import load_dotenv

load_dotenv(".env")

from scripts.utils.azure_utils import get_workspace
from scripts.utils.azure_utils import register_dataset
from scripts.utils.azure_utils import upload_file_to_datastore

if __name__ == "__main__":
    ws = get_workspace()
    default_datastore = ws.get_default_datastore()
    dataset_name = "bank_customer_churn"
    data_path = "data/raw/"
    remote_path = "data"
    upload_file_to_datastore(datastore=default_datastore, local_path=data_path)
    register_dataset(
        workspace=ws,
        data_storage=default_datastore,
        data_path=remote_path,
        dataset_name=dataset_name,
    )
