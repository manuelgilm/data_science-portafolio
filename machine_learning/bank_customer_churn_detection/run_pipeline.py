from azureml.core import Dataset, Experiment
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from scripts.utils.azure_utils import get_workspace
from azureml.data import OutputFileDatasetConfig
# creating environment for pipeline scripts
packages = CondaDependencies.create(
    conda_packages=['scikit-learn==1.1.1','pip', 'pandas','matplotlib'],
    pip_packages=['azureml-defaults','joblib']
)

run_config = RunConfiguration()
run_config.environment.python.conda_dependencies = packages

# get workspace
ws = get_workspace()

#get dataset 
dataset = Dataset.get_by_name(ws,"bank_customer_churn")

# create link between pipelines
data_link = OutputFileDatasetConfig("datalink")
dataset_name = "bank_customer_churn_detection"

step1 = PythonScriptStep(
    name = "Data Preprocessing",
    source_directory = "scripts\pipeline_scripts",
    script_name="processing_data.py",
    compute_target="ml-testing2",
    runconfig = run_config,
    arguments = [
        '--filename',dataset_name,
        '--dataset', dataset.as_named_input(dataset_name),
        '--output-folder', data_link
    ]
)

pipeline = Pipeline(workspace=ws, steps=[step1], description="Loading data for preprocessing")

experiment = Experiment(workspace=ws, name="bank_custumer_churn_detection")
experiment.submit(pipeline)