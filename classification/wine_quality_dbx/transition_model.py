import mlflow
from mlflow.tracking import MlflowClient
from wine_quality.data_preparation import get_configurations
from wine_quality.model_func import create_mlflow_experiment

if __name__ == "__main__":

    configs = get_configurations(filename="common_params")
    experiment_name = configs["experiment_name"]
    create_mlflow_experiment(experiment_name=experiment_name)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)
    run_id = runs["run_id"].iloc[0]
    model_artifact_uri = runs["artifact_uri"].iloc[0]
    print(model_artifact_uri)
    model_name = "wine_quality_model"
    mlflow.register_model(
        model_uri="runs:/{}/pipeline".format(run_id), name=model_name
    )
    client.transition_model_version_stage(
        name=model_name, version=1, stage="Production"
    )
