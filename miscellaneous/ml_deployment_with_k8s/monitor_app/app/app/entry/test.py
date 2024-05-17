from app.storage.base import Prediction


def main():
    prediction = Prediction(model_name="dummy_classifier")
    for val in range(10):

        prediction.save_value(val)

    