import mlflow
import yaml
from databricks.feature_store import FeatureLookup
from databricks.feature_store import FeatureStoreClient
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType


def read_data(config: dict) -> DataFrame:
    """Reads the data from dbfs and returns a spark dataframe.

    params: config: dict
    return: sdf: spark dataframe
    """
    spark = SparkSession.getActiveSession()
    data_path = config["source_data"]
    print(f"Reading the data from dbfs: {data_path}")
    sdf = spark.read.csv(
        data_path, header=True, inferSchema=True, multiLine="true", escape='"'
    )
    return sdf


def get_configurations(filename: str) -> dict:
    """Reads the configuration file from dbfs and returns a dictionary.

    params: filename: str
    return: config: dict
    """
    with open(f"/dbfs/FileStore/configs/{filename}.yaml", "rb") as f:
        print(f"Reading the configuration file from dbfs: {filename}.yaml")
        config = yaml.load(f)
    return config


def get_imputer(numerical_columns: list) -> Imputer:
    """
    This function returns an Imputer for the given numerical columns.

    params:numerical_columns:list
        A list of numerical columns.
    return:Imputer
    """
    numerical_imputed = [col + "_imputed" for col in numerical_columns]
    numerical_imputer = Imputer(
        inputCols=numerical_columns,
        outputCols=numerical_imputed,
        strategy="median",
    )
    return numerical_imputer


def get_string_indexer(categorical_columns: list) -> StringIndexer:
    """
    This function returns a StringIndexer for the given categorical columns.

    params:categorical_columns:list
        A list of categorical columns.
    return:StringIndexer
    """
    indexed_columns = [col + "_indexed" for col in categorical_columns]
    string_indexer = StringIndexer(
        inputCols=categorical_columns,
        outputCols=indexed_columns,
        handleInvalid="skip",
    )
    return string_indexer


def get_pipeline(
    config: dict, categorical_columns: list, numerical_columns: list
):
    """
    This function returns a pipeline for the given configuration and spark dataframe.

    params:config:dict
        A dictionary containing the configuration for the pipeline.
    params:sdf:DataFrame
        A spark dataframe containing the data to be used for training the pipeline.
    return:Pipeline
    """

    label = config["label"]

    numerical_imputer = get_imputer(numerical_columns=numerical_columns)
    string_indexer = get_string_indexer(
        categorical_columns=categorical_columns
    )
    feature_names = (
        numerical_imputer.getOutputCols() + string_indexer.getOutputCols()
    )
    vector_assembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features",
        handleInvalid="keep",
    )

    dtr = DecisionTreeRegressor(labelCol=label, featuresCol="features")
    dtr.setMaxBins(50)

    pipeline = Pipeline(
        stages=[numerical_imputer, string_indexer, vector_assembler, dtr]
    )

    return pipeline


def get_feature_names(config: dict) -> list:
    """
    This function returns the feature names for the given configuration.
    """
    columns_to_keep = config["columns_to_keep"]
    columns_to_keep.remove(config["label"])
    sdf = read_data(config=config)
    categorical_columns = [
        field.name
        for field in sdf.select(columns_to_keep).schema.fields
        if isinstance(field.dataType, StringType)
    ]
    numerical_columns = [
        field.name
        for field in sdf.select(columns_to_keep).schema.fields
        if isinstance(field.dataType, DoubleType)
    ]
    return categorical_columns, numerical_columns


def set_or_create_mlflow_experiment(experiment_name: str) -> None:
    """
    This function sets or creates a new mlflow experiment.
    """
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)


def get_train_test_ids(config: dict, fs: FeatureStoreClient):
    """
    This function returns the training and test ids for the given configuration.
    """
    feature_table = config["feature_table_name"]
    database_name = config["database_name"]
    sdf = fs.read_table(name=f"{database_name}.{feature_table}")
    ids_and_labels = sdf.select("id", "price")
    train_ids, test_ids = ids_and_labels.randomSplit([0.8, 0.2], seed=42)
    return train_ids, test_ids


def get_train_test_sets(config, fs, train_ids, test_ids, feature_names):
    """
    This function returns the train and test data for the given configuration and feature store client.
    """

    label = config["label"]
    table_name = config["feature_table_name"]
    database_name = config["database_name"]
    feature_table_name = f"{database_name}.{table_name}"

    feature_lookups = FeatureLookup(
        table_name=feature_table_name,
        feature_names=feature_names,
        lookup_key="id",
    )
    training_set = fs.create_training_set(
        train_ids,
        feature_lookups=[feature_lookups],
        label=label,
        exclude_columns=["id"],
    )
    testing_set = fs.create_training_set(
        test_ids,
        feature_lookups=[feature_lookups],
        label=None,
        exclude_columns=["id"],
    )

    return training_set, testing_set
