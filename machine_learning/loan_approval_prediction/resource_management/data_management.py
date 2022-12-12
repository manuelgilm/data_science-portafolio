from azureml.core import Datastore

def create_blob_datastore(
    workspace, datastore_name, storage_account_name, blob_container_name, account_key
):
    """Creates a datastore"""
    az_datastore = Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=datastore_name,
        account_name=storage_account_name,
        container_name=blob_container_name,
        account_key=account_key,
    )
    return az_datastore