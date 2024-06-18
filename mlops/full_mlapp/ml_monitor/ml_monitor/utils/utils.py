import os

from typing import Union


def get_url(service_name: str) -> Union[str, None]:
    """
    Get the URL of the service

    :param service_name: name of the service
    :return: URL of the service
    """

    host = os.environ.get(f"{service_name}_SERVICE_HOST", None)
    port = os.environ.get(f"{service_name}_SERVICE_PORT", None)

    if host is None or port is None:
        return None
    return f"http://{host}:{port}"
