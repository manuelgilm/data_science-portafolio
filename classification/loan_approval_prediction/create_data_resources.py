from resource_management.workspace_management import create_workspace
from resource_management.data_management import (
    create_blob_datastore,
    upload_folder,
    create_tabular_dataset,
    create_blob_container,
    create_storage_account,
)
from azure.mgmt.storage import StorageManagementClient
from azure.identity import AzureCliCredential

from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()

    ws_name = os.environ["WORKSPACE_NAME"]
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    resource_group = os.environ["RESOURCE_GROUP"]
    datastore_name = os.environ["DATASTORE_NAME"]
    location = os.environ["LOCATION"]
    storage_account_name = os.environ["STORAGE_ACCOUNT"]
    container_name = os.environ["CONTAINER_NAME"]

    credentials = AzureCliCredential()
    storage_client = StorageManagementClient(
        credential=credentials, subscription_id=subscription_id
    )

    # create storage account
    create_storage_account(
        storage_client=storage_client,
        storage_account_name=storage_account_name,
        resource_group_name=resource_group,
        location=location,
    )

    # create container
    create_blob_container(
        storage_client=storage_client,
        storage_account_name=storage_account_name,
        resource_group_name=resource_group,
        container_name=container_name
    )

    # create and get ml workspace
    ws = create_workspace(
        workspace_name=ws_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
    )

    # Retrieve the account's primary access key and generate a connection string.
    keys = storage_client.storage_accounts.list_keys(
        resource_group, storage_account_name
    )
    
    # create datastore
    datastore = create_blob_datastore(
        workspace=ws,
        datastore_name=datastore_name,
        storage_account_name=storage_account_name,
        blob_container_name=container_name,
        account_key=keys.keys[0].value,
    )

    # upload data to storage account
    upload_folder(datastore=datastore, source_directory="./Loan_Data", target=".")

    # create dataset
    create_tabular_dataset(
        workspace=ws,
        datastore=datastore,
        paths="Loan_Approval_Prediction.csv",
        dataset_name="loan_data",
    )
