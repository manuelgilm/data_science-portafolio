import json
import requests
from ml_monitor.utils.utils import get_url
import os

from typing import Dict
from typing import Any


def get_model_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the model prediction from the target service.

    :param payload: dict, the input payload for the model.
    :return: dict, the model prediction.
    """
    target_service = os.environ.get("TARGET_SERVICE", None)
    if target_service is None:
        raise ValueError("TARGET_SERVICE environment variable is not set.")
    url = get_url(target_service)

    headers = {
        "Content-Type": "application/json",
    }
    endpoint = url + "/invocations"
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
    return response.json()
