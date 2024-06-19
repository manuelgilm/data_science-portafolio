import json 
import requests
from ml_monitor.utils.utils import get_url

def get_model_prediction(payload):
    config = {
        "service_name": "iris",
    }
    service_name = config["service_name"]
    url = get_url(service_name)

    headers = {
        "Content-Type": "application/json",
    }
    endpoint = url + "/invocations"
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
    return response.json()