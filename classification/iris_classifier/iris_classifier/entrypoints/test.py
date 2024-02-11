from iris_classifier.data.retrieval import get_dataset
from iris_classifier.train.pipelines import ClassifierPipeline


def test_get_dataset():
    """
    Test the get_dataset method.
    """
    df = get_dataset()
    print(df.head())


def test_get_pipeline():
    """
    Test the get_pipeline method.
    """
    df = get_dataset()
    clf_pipeline = ClassifierPipeline()
    print(clf_pipeline.get_pipeline(df))
