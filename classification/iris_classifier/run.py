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
        ]
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
            "5001",
        ]
    )


if __name__ == "__main__":
    run_mlflow_ui()
    run_streamlit_app()
