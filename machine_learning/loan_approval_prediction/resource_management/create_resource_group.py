import os
from dotenv import load_dotenv

load_dotenv()
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient

# ENVIRONMENT VARIABLES
# resource_group_name
resource_group_name = os.environ["RESOURCE_GROUP"]
# storage account name
storage_account_name = os.environ["STORAGE_ACCOUNT"]
# Retrieve subscription ID from environment variable.
subscription_id = os.environ["SUBSCRIPTION_ID"]
# location
location = os.environ["LOCATION"]

# Acquire a credential object using CLI-based authentication
credential = AzureCliCredential()

# Obtain the management object for resources.
resource_client = ResourceManagementClient(credential, subscription_id)

# storage client
storage_client = StorageManagementClient(credential, subscription_id)

# Provision the resource group.
rg_names = [rg.name for rg in resource_client.resource_groups.list()]
if resource_group_name in rg_names:
    print("RESOURCE GROUP ALREADY EXIST")
else:
    rg_result = resource_client.resource_groups.create_or_update(
        resource_group_name, {"location":location }
    )
    print(
        f"Provisioned resource group {rg_result.name} in \
            the {rg_result.location} region"
    )

# STORAGE ACCOUNT
# create storage account within the resource group, then create a container.
# Check if the account name is available. Storage account names must be unique across
# Azure because they're used in URLs.
availability_result = storage_client.storage_accounts.check_name_availability(
    { "name": storage_account_name }
)

if availability_result.name_available:
    # The name is available, so provision the account
    poller = storage_client.storage_accounts.begin_create(resource_group_name, storage_account_name,
        {
            "location" : location,
            "kind": "StorageV2",
            "sku": {"name": "Standard_LRS"}
        }
    )
    # Long-running operations return a poller object; calling poller.result()
    # waits for completion.
    account_result = poller.result()
    print(f"Provisioned storage account {account_result.name}")
else:
    print("STORAGE ACCOUNT NAME ALREADY TAKEN")


# Retrieve the account's primary access key and generate a connection string.
keys = storage_client.storage_accounts.list_keys(resource_group_name, storage_account_name)

print(f"Primary key for storage account: {keys.keys[0].value}")

conn_string = f"DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName={storage_account_name};AccountKey={keys.keys[0].value}"

print(f"Connection string: {conn_string}")

# Provision the blob container in the account (this call is synchronous)
CONTAINER_NAME = os.environ["CONTAINER_NAME"]
container = storage_client.blob_containers.create(resource_group_name, storage_account_name, CONTAINER_NAME, {})

print(f"Provisioned blob container {container.name}")