import requests


def test_gateway(endpoint):
    response = requests.get(endpoint)
    print(response.json())


if __name__ == "__main__":
    endpoint = "http://localhost:8002/ml1/ml1/predict"
    test_gateway(endpoint)
