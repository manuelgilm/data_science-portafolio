from azureml.core import Workspace
from dotenv import load_dotenv
import os

load_dotenv()

ws_name = os.environ["WORKSPACE_NAME"]
subscription_id = os.environ["SUBSCRIPTION_ID"]
resource_group = os.environ["RESOURCE_GROUP"]

try:
    ws = Workspace.create(
        name=ws_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
    )
    ws.write_config("./configs")
except Exception as e:
    raise e


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
        raise e
    return ws


def get_workspace_from_config(config_path):
    """Gets a Azuren Machine Learning Workspace from config file."""
    try:
        ws = Workspace.from_config(config_path)
    except Exception as e:
        raise e

    return ws

