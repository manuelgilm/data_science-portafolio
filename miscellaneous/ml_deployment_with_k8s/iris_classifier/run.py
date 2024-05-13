import os
import subprocess

import mlflow


def run_mlflow_ui():
    """Run the mlflow ui."""
    subprocess.run(
        [
            "poetry",
            "run",
            "mlflow",
            "ui",
            "--port",
            "5000",
            "--host",
            "0.0.0.0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_streamlit_app():
    """Run the streamlit app."""
    subprocess.run(
        [
            "poetry",
            "run",
            "streamlit",
            "run",
            "iris_classifier/eda/vz_app.py",
            "--server.port",
            "5000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def deploy_model(name:str):
    """
    Deploy the model.

    :param name: Name Of the registered model.
    """
    subprocess.run(
        [
            "poetry",
            "run",
            "mlflow",
            "models",
            "serve",
            "--model-uri",
            f"models:/{name}/latest",
            "--port",
            "5000",
            "--host",
            "0.0.0.0",
            "--no-conda",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def train_model():
    """Train the model."""
    subprocess.run(
        [
            "poetry",
            "run",
            "train",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def train():
    """
    Train the model and run the mlflow ui.
    """
    train_model()
    run_mlflow_ui()


def train_and_deploy_model():
    """
    Train the model and deploy the model.
    """
    name = "Iris-Classifier"
    train_model()
    print("Model trained")
    deploy_model(name=name)
    print("Model deployed")


def run_mode(mode: str):
    """
    Run the mode.

    :param mode: The mode to run.
    :return: The result of the mode.
    """
    modes = {
        "EDA": run_streamlit_app,
        "TRAIN": train,
        "SCORE": train_and_deploy_model,
    }
    return modes[mode]()


if __name__ == "__main__":
    mode = os.environ.get("MODE", "EDA")
    if mode in ["EDA", "TRAIN", "SCORE"]:
        run_mode(mode)
    else:
        raise ValueError(f"Invalid mode: {mode}")
