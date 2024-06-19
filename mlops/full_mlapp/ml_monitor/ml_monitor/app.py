from flask import Flask
from flask import request
import os
import json
import requests

from ml_monitor.utils.utils import get_url

app = Flask(__name__)

tasks = {
    "Hi": "Hi, I am a task inside the app",
}


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


@app.route("/prediction", methods=["POST"])
def get_prediction():
    # get the data from the request
    data_json = request.get_json()
    prediction = get_model_prediction(data_json)
    prediction.update({"status": "success"})
    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
