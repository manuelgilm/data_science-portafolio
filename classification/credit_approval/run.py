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


def deploy_model(run_id: str, model_artifact: str):
    """
    Deploy the model.

    :param run_id: The run id of the model to deploy.
    :param model_artifact: The model artifact to deploy.
    """
    subprocess.run(
        [
            "poetry",
            "run",
            "mlflow",
            "models",
            "serve",
            "--model-uri",
            f"runs:/{run_id}/{model_artifact}",
            "--port",
            "5000",
            "--host",
            "0.0.0.0",
            "--no-conda",
        ]
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
    name = "credit_approval"
    train_model()
    print("Model trained")
    experiment = mlflow.get_experiment_by_name(name)
    latest_run = mlflow.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["start_time desc"],
        max_results=1,
    )
    run_id = latest_run.iloc[0]["run_id"]
    deploy_model(run_id=run_id, model_artifact=name)
    print("Model deployed")


def run_mode(mode: str):
    """
    Run the mode.

    :param mode: The mode to run.
    :return: The result of the mode.
    """
    modes = {
        "TRAIN": train,
        "SCORE": train_and_deploy_model,
    }
    return modes[mode]()


if __name__ == "__main__":
    mode = os.environ.get("MODE", "TRAIN")
    if mode in ["TRAIN", "SCORE"]:
        run_mode(mode)
    else:
        raise ValueError(f"Invalid mode: {mode}")
