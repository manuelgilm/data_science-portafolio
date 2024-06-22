import json
import requests
from ml_monitor.utils.utils import get_url
import os


def get_model_prediction(payload):
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
