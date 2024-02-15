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


if __name__ == "__main__":
    mode = os.environ.get("MODE", "EDA")
    if mode == "EDA":
        run_streamlit_app()
    elif mode == "MLFLOW":
        run_mlflow_ui()
    else:
        raise ValueError(f"Invalid mode: {mode}. Use EDA or MLFLOW.")
