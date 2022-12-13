from azureml.core import Datastore, Dataset

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

def create_tabular_dataset(workspace, datastore, paths, dataset_name=None, dataset_description=None):
    """Creates a dataset object and register it."""
    filepath = [(datastore, paths)]
    dataset = Dataset.Tabular.from_delimited_files(path=filepath)
    dataset.register(workspace=workspace, name=dataset_name, description=dataset_description, create_new_version=True)


def upload_files(datastore, path_list, target, relative_local_path):
    """Uploads files to a datastore.
    
    datastore: Datastore object.
    path_list: list of paths to each one of the files.
    target: Remote path (in datastore)
    relative_local_paht: the relative local path    
    """

    datastore.upload_files(
        files = path_list,
        target_path = target,
        relative_root = relative_local_path,
        overwrite=True
    )
    

def upload_folder(datastore, source_directory, target):
    """Upload a folder to a datastore.
    
    source_directory: local directory to be uploaded.
    target: Remote path (in datastore)    
    """
    datastore.upload(
        src_dir = source_directory,
        target_path=target,
        overwrite=True
    )