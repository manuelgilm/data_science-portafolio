import pandas as pd
from wine_quality_v1.data_preparation.data_preparation import get_wine_dataset_uci
from wine_quality_v1.data_preparation.data_preparation import read_config


def test_get_dataset():

    df, feature_names = get_wine_dataset_uci()
    assert type(df) == type(pd.DataFrame())
    assert len(feature_names) == 11


def test_reading_config_file():
    """
    Test reading config file.
    """

    config = read_config()
    assert type(config) == type(dict())
