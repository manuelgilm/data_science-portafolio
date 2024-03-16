from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Tuple 

def get_dataset()->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the concrete compressive strength dataset. 
    along with the metadata.

    :return: Tuple of pandas dataframes (df, metadata)
    """
    # fetch dataset 
    concrete_compressive_strength = fetch_ucirepo(id=165) 

    # get metadata
    metadata = concrete_compressive_strength.variables

    # data (as pandas dataframes) 
    X = concrete_compressive_strength.data.features 
    y = concrete_compressive_strength.data.targets 
    df = pd.concat([X, y], axis=1)
    return df, metadata


def process_column_names(df):
    """
    Process column names to remove spaces and special characters.

    :param df: pandas dataframe
    :return: pandas dataframe
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df