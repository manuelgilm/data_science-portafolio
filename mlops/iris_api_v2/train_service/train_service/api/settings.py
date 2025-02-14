from fastapi import FastAPI
import mlflow


def lifespan(app: FastAPI):
    mlflow.set_tracking_uri("http://tracking_service:5000")
    yield
