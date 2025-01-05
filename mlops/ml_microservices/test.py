import requests

def test_gateway():
    response = requests.get("http://localhost:8002/ml1/ml1/predict")
    print(response.json())

if __name__=="__main__":
    test_gateway()