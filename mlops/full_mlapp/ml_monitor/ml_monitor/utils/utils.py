import os

from typing import Union
from typing import Dict
from typing import Any
from pathlib import Path

def get_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the record from the payload

    :param payload: payload
    :return: record
    """
    record = {
        feature: value
        for feature, value in zip(
            payload["dataframe_split"]["columns"], payload["dataframe_split"]["data"][0]
        )
    }
    record.update({"prediction": None, "label": None})

    return record


def get_root_path() -> Path:
    """
    Get the root path of the project

    :return: root path of the project
    """
    return Path(__file__).parent.parent.parent


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


def get_payload(
    sepal_length: float, sepal_width: float, petal_length: float, petal_width: float
) -> Dict[str, Any]:
    """
    Get the payload to send to the model

    :param sepal_length: sepal length
    :param sepal_width: sepal width
    :param petal_length: petal length
    :param petal_width: petal width
    :return: payload
    """
    payload = {
        "dataframe_split": {
            "columns": [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
            "data": [[sepal_length, sepal_width, petal_length, petal_width]],
        }
    }
    return payload
