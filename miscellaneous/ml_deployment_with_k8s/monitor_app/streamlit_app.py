from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import requests
import json
"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)", "petal width (cm)", "petal length (cm)"]
    
def __is_valid_json(input):
    try:
        json.loads(input)
        return True
    except ValueError:
        print("Not Valid Json")
    return False

def __is_valid_schema(input):
    features = json.loads(input)
    feature_names = list(features.keys())
    if feature_names == FEATURE_NAMES:
        return True 
    else:
        return False
    
def get_payload(input):

    if __is_valid_json(input) and __is_valid_schema(input):
        features = json.loads(input)
        columns = list(features.keys())
        data = list(features.values())
        payload =  {"dataframe_split": {
            "data":[data],
            "columns":columns
        }}
        return payload

def make_request(payload):

    headers = {"Content-Type":"application/json"}
    endpoint = "http://irisclassifier:5000/invocations"

    response = requests.post(endpoint, data = json.dumps(payload), headers=headers)

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        return predictions
    else:
        return response.text
    
def check_status():
    headers = {"Content-Type":"application/json"}
    endpoint = "http://irisclassifier:5000/ping"

    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return response.text


with st.echo(code_location='below'):
   
    input = st.text_area(label="features")

    st.write(f"You wrote {len(input)} characters.")
    st.write(input)
    payload = get_payload(input)
    st.write("Your input is")
    st.write(payload)
    if payload:
        print(payload)
        
        prediction = make_request(payload=payload)
        st.write("Your Prediction is")
        st.write(prediction)

        
    else:
        st.write("Payload not Provided!")


