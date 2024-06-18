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

    # payload = {"dataframe_split":
    #     {"columns":[
    #         "sepal length (cm)",
    #         "sepal width (cm)",
    #         "petal length (cm)",
    #         "petal width (cm)"
    #         ]
    #     ,"data":[
    #         [5.1,3.5,1.4,0.2]]
    #     }
    # }


# #post method
# @app.route("/resource", methods=["POST"])
# def post():

#     # get the data from the request
#     data_json = request.get_json()
#     keys = list(data_json.keys())
#     if len(keys) == 0:
#         return "No data found"

#     if len(keys) > 1:
#         return "Only one key is allowed"
#     value = data_json[keys[0]]
#     task = tasks.get(keys[0], None)

#     if task is None:
#         return "Task not found"

#     return task + " " + value


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
