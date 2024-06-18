import requests
import json 

payload = {"dataframe_split":
    {"columns":[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
        ]
    ,"data":[
        [5.1,3.5,7.0,3]]
    }
}

headers = {
    'Content-Type': 'application/json',
}
# response = requests.post('http://localhost:5000/invocations', headers=headers, data=json.dumps(payload))
response = requests.post('http://172.175.208.114:8080/prediction', headers=headers, data=json.dumps(payload))

print(response.json())
