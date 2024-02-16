import os
import subprocess


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


def score_dataset():
    """Score the dataset."""
    subprocess.run(
        [
            "poetry",
            "run",
            "predict",
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


def run_mode(mode: str):
    """
    Run the mode.

    :param mode: The mode to run.
    :return: The result of the mode.
    """
    modes = {
        "EDA": run_streamlit_app,
        "MLFLOW": run_mlflow_ui,
        "TRAIN": train_model,
        "SCORE": score_dataset,
    }
    return modes[mode]()


if __name__ == "__main__":
    mode = os.environ.get("MODE", "EDA")
    if mode in ["EDA", "MLFLOW", "TRAIN", "SCORE"]:
        run_mode(mode)
    else:
        raise ValueError(f"Invalid mode: {mode}")
