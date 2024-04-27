from pathlib import Path
from typing import List
from typing import Union

import pandas as pd
from bike_sharing.utils.utils import get_project_root


def save_dataset(
    df: pd.DataFrame, column_names: List[str], path: Union[str, Path]
) -> None:
    """
    Save the training dataset.

    :param df: Dataframe
    :param column_names: List of column names
    :param path: Path to save the dataset (referent to the project root)

    """
    # save the training dataset
    if isinstance(path, str):
        path = Path(path)

    project_dir = get_project_root()
    folder = path if not path.suffix else path.parent
    folder_path = project_dir / folder
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    filename = path.name if path.suffix else "dataset.csv"
    filepath = folder_path / filename
    df[column_names].to_csv(filepath.as_posix(), index=False)
