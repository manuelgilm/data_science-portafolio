import os

from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from dotenv import load_dotenv
from resource_management.azure_environment import create_resource_group

if __name__ == "__main__":
    load_dotenv()

    # ENVIRONMENT VARIABLES
    resource_group_name = os.environ["RESOURCE_GROUP"]
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    location = os.environ["LOCATION"]

    credential = AzureCliCredential()
    resource_client = ResourceManagementClient(credential, subscription_id)

    # create resource group
    create_resource_group(
        resource_client=resource_client,
        resource_group_name=resource_group_name,
        location=location,
    )
