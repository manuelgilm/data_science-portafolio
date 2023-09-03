import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Imputer, VectorAssembler

def create_mlflow_experiment(experiment_name: str) -> None:
    """
    This function creates a mlflow experiment.

    params: experiment_name: str

    """
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

def get_pipeline(columns: list) -> Pipeline:
    """
    This function creates a pipeline for training the model.

    params: columns: list
    return: pipeline: pyspark pipeline
    """

    imputed_columns = [col + "_imputed" for col in columns]
    numerical_imputer = Imputer(
        inputCols=columns, outputCols=imputed_columns, strategy="mean"
    )
    assembler = VectorAssembler(inputCols=imputed_columns, outputCol="features")
    rf = RandomForestClassifier(featuresCol="features", labelCol="target", numTrees=10)
    pipeline = Pipeline(stages=[numerical_imputer, assembler, rf])

    return pipeline        