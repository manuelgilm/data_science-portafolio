import pandas as pd
from ucimlrepo import fetch_ucirepo


def get_data() -> pd.DataFrame:
    """
    Get the credit approval dataset.
    """
    credit_approval = fetch_ucirepo(id=27)
    X = credit_approval.data.features
    y = credit_approval.data.targets
    target = credit_approval["metadata"]["target_col"][0]
    df = pd.concat([X, y], axis=1)
    # map target to 0 and 1
    df[target] = df[target].map(lambda x: 1 if x == "+" else 0)

    return df


def get_data_info() -> pd.DataFrame:
    """
    Get the data information of the credit approval dataset.
    """
    credit_approval = fetch_ucirepo(id=27)
    data_info = credit_approval["variables"]
    return data_info
