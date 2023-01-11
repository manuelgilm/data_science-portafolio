from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from dotenv import load_dotenv
import os 
load_dotenv()

subscription_id = os.environ["SUBSCRIPTION_ID"]
resource_group_name = os.environ["RESOURCE_GROUP"]
credentials = AzureCliCredential()
client = ResourceManagementClient(credential=credentials, subscription_id=subscription_id)
delete_async_operation = client.resource_groups.begin_delete(resource_group_name)
delete_async_operation.wait()