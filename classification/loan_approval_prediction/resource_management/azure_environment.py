def create_resource_group(resource_client, resource_group_name, location):
    """Creates a resource group"""
    resource_group_names = [rg.name for rg in resource_client.resource_groups.list()]
    if resource_group_name in resource_group_names:
        print("RESOURCE GROUP ALREADY EXIST")
    else:
        rg_result = resource_client.resource_groups.create_or_update(
            resource_group_name, {"location": location}
        )
        print(
            f"Provisioned resource group {rg_result.name} in \
                the {rg_result.location} region"
        )
