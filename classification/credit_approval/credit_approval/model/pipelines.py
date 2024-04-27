from typing import List

from skdag import DAG
from skdag import DAGBuilder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_sk_pipeline(
    categorical_columns: List[str], numerical_columns: List[str]
) -> Pipeline:
    """
    Get the pipeline for the credit approval dataset.

    :param categorical_columns: List of categorical columns.
    :param numerical_columns: List of numerical columns.
    :return: The pipeline.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numerical_imputer",
                SimpleImputer(strategy="median"),
                numerical_columns,
            ),
            ("encoder", OneHotEncoder(), categorical_columns),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", RandomForestClassifier()),
        ]
    )
    return pipeline


def get_dag_pipeline(
    categorical_columns: List[str], numerical_columns: List[str]
) -> DAG:
    """
    Get the pipeline for the credit approval dataset.

    :param categorical_columns: List of categorical columns.
    :param numerical_columns: List of numerical columns.
    :return: The pipeline.
    """
    dag = (
        DAGBuilder(infer_dataframe=True)
        .add_step("input", "passthrough")
        .add_step(
            "categorical_imputer",
            SimpleImputer(strategy="most_frequent"),
            deps={"input": categorical_columns},
        )
        .add_step(
            "numerical_imputer",
            SimpleImputer(strategy="median"),
            deps={"input": numerical_columns},
            dataframe_columns=numerical_columns,
        )
        .add_step(
            "encoder",
            OneHotEncoder(),
            deps={"categorical_imputer": categorical_columns},
            dataframe_columns=categorical_columns,
        )
        .add_step(
            "model",
            RandomForestClassifier(),
            deps=["encoder", "numerical_imputer"],
        )
        .make_dag(n_jobs=2, verbose=True)
    )
    return dag
