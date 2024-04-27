from resource_management.workspace_management import get_workspace_from_config

from azureml.core import ScriptRunConfig
from azureml.core import Environment, Experiment
from azureml.core.conda_dependencies import CondaDependencies

ws = get_workspace_from_config(config_path="./configs")

# creating packages
conda_packages = CondaDependencies.create(conda_packages=["pandas","scikit-learn", "matplotlib","mlflow"])

# creating and registering environment
env = Environment("loanEnvExperiment")
env.python.conda_dependencies = conda_packages
env.register(workspace=ws)

# creating script configuration
script_config = ScriptRunConfig(
    source_directory="./experiments/scripts",
    script="preprocess_and_training_mlflow.py",
    environment=env
)

# create experiment
experiment = Experiment(workspace=ws, name="loanStatusPrediction")
experiment = experiment.submit(config=script_config)
experiment.wait_for_completion(show_output=True)





