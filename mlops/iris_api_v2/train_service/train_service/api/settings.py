from fastapi import FastAPI
from train_service.src.train import Trainer
import mlflow


def lifespan(app: FastAPI):
    mlflow.set_tracking_uri("http://tracking_service:5000")
    train_service = Trainer()
    x_train, x_test, y_train, y_test = train_service.get_train_test_data()
    print(x_train.head())
    yield
