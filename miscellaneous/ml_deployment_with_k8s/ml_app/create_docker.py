import mlflow

mlflow.models.build_docker(
    model_uri=f"models:/DummyModel/latest",
    name="dummy-classifier",
    enable_mlserver=True,
)
