import subprocess


def run_streamlit_app():
    """Run the streamlit app."""

    subprocess.run(["streamlit", "run", "iris_classifier/eda/vz_app.py"])
