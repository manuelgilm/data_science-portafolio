from iris_classifier.data.retrieval import get_dataset
from iris_classifier.train.pipelines import ClassifierPipeline
from sklearn.pipeline import Pipeline


def test_get_pipeline():
    """
    Test the get_pipeline method.
    """
    df = get_dataset()
    clf_pipeline = ClassifierPipeline()
    pipeline = clf_pipeline.get_pipeline(df)
    assert isinstance(pipeline, Pipeline)


def test_pipeline_steps():
    """
    Test the number of steps in the pipeline.
    """
    df = get_dataset()
    clf_pipeline = ClassifierPipeline()
    pipeline = clf_pipeline.get_pipeline(df)
    steps = pipeline.steps
    assert len(steps) == 2
