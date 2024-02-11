from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ClassifierPipeline:
    """
    Define the pipeline for the classifier.

    Methods:
    ----------
    get_pipeline:
        Get the pipeline for the classifier.
    """

    def __get_categorical_columns(self, df):
        """
        Get the categorical columns of the dataframe.

        :param df: The dataframe to parse.
        :return: A list of strings representing the categorical
        columns of the dataframe.
        """
        return list(df.select_dtypes(include=["object"]).columns)

    def __get_numerical_columns(self, df):
        """
        Get the numerical columns of the dataframe.

        :param df: The dataframe to parse.
        :return: A list of strings representing the numerical
        columns of the dataframe.
        """
        return list(df.select_dtypes(include=["float64", "int64"]).columns)

    def __get_pipeline(self, df):
        """
        Get the pipeline for the classifier.

        :param df: The dataframe to parse.
        :return: A pipeline for the classifier.
        """
        numerical_features = self.__get_numerical_columns(df)
        categorical_features = self.__get_categorical_columns(df)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(), numerical_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier()),
            ]
        )
        return pipeline

    def get_pipeline(self, df):
        """
        Get the pipeline for the classifier.

        :param df: The dataframe to parse.
        :return: A pipeline for the classifier.
        """
        return self.__get_pipeline(df)
