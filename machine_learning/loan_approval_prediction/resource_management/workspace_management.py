from azureml.core import Workspace


def create_workspace(workspace_name, subscription_id, resource_group):
    """Creates a Azure Machine Learning Workspace"""
    try:
        if resource_group:
            ws = Workspace.create(
                name=workspace_name,
                subscription_id=subscription_id,
                resource_group=resource_group,
            )
        else:
            ws = Workspace.create(
                name=workspace_name,
                subscription_id=subscription_id,
                create_resource_group=True,
            )

        ws.write_config("./configs")
    except Exception as e:
        ws = Workspace.from_config(path="./configs")
    return ws


def get_workspace_from_config(config_path):
    """Gets a Azuren Machine Learning Workspace from config file."""
    try:
        ws = Workspace.from_config(config_path)
    except Exception as e:
        raise e
    return ws



