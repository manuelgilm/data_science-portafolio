from azureml.core import Dataset, Experiment
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.core import Pipeline, PipelineData
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

#get datastore
default_datastore = ws.get_default_datastore()

#get dataset 
dataset = Dataset.get_by_name(ws,"bank_customer_churn")

# create link between pipelines
dataset_name = "bank_customer_churn_detection"

# create an output dataset
prepped_data = PipelineData(name="preprocessed_data", datastore=default_datastore).as_dataset()
training_data = PipelineData(name="training_data", datastore=default_datastore).as_dataset()
testing_data = PipelineData(name="testing_data", datastore=default_datastore).as_dataset()

# parameters
label = PipelineParameter(name="label",default_value="active_member")

features_list = [
    "balance",
    "products_number",
    "credit_card",
    "tenure",
    "age",
    "gender_Female",
    "gender_Male",
    "credit_score"
]
features = PipelineParameter(name="features",default_value=",".join(features_list))


step1 = PythonScriptStep(
    name = "Data Preprocessing",
    source_directory = "scripts\pipeline_scripts",
    script_name="step1_processing_data.py",
    compute_target="ml-testing2",
    runconfig = run_config,
    outputs=[prepped_data],
    arguments = [
        '--filename',dataset_name,
        '--dataset', dataset.as_named_input(dataset_name),
        '--preprocessed-data', prepped_data
    ]
)

step2 = PythonScriptStep(
    name="Split data",
    source_directory="scripts\pipeline_scripts",
    script_name="step2_split_data.py",
    compute_target="ml-testing2",
    runconfig=run_config,
    inputs=[prepped_data.parse_delimited_files()],
    outputs = [training_data, testing_data],
    arguments=[
        '--filename', dataset_name,
        '--output-training-data',training_data,
        '--output-testing-data', testing_data,
        '--features', features,
        '--label', label
    ]
)

step3 = PythonScriptStep(
    name="Model training",
    source_directory = "scripts\pipeline_scripts",
    script_name = "step3_train_model.py",
    compute_target="ml-testing2",
    runconfig=run_config,
    inputs=[training_data.parse_delimited_files()],
    arguments=[        
        '--features', features,
        '--label', label
        ]
)

step4 = PythonScriptStep(
    name = "Model Testing",
    source_directory = "scripts\pipeline_scripts",
    script_name = "step4_score_model.py",
    compute_target="ml-testing2",
    runconfig=run_config,
    inputs=[testing_data.parse_delimited_files()],
)
pipeline = Pipeline(workspace=ws, steps=[step1, step2,step3, step4], description="Loading data for preprocessing")

experiment = Experiment(workspace=ws, name="bank_custumer_churn_detection")
experiment.submit(pipeline)