from sklearn.datasets import fetch_kddcup99


def get_dataset():
    """
    Get the KDD Cup 1999 dataset from the sklearn.datasets module.
    """
    data = fetch_kddcup99(as_frame=True)
    df = data.frame

    return df
