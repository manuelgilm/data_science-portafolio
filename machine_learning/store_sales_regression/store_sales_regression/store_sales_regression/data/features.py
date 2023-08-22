import pandas as pd
from typing import Tuple

def create_time_based_features(df:pd.DataFrame, date_col:str)->Tuple[pd.DataFrame, list[str]]:
    """
    Creates features based on time.

    :param df: pandas DataFrame
    :param date_col: date column
    :return: pandas DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["quarter"] = df[date_col].dt.quarter
    df["dayofyear"] = df[date_col].dt.dayofyear
    feature_names = ["year", "month", "day", "dayofweek", "quarter", "dayofyear"]
    return df, feature_names


def aggregate_sales_data(df:pd.DataFrame)->pd.DataFrame:
    """
    Aggregates sales data.

    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df = df.copy()

    # Aggregate all sales per store
    df_agg = df.groupby(["date","store"], as_index=False).agg({"sales": "sum"})
    return df_agg