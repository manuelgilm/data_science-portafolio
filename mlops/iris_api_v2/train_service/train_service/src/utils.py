import pickle
from pathlib import Path 
import os 
def load_pickle(path: str):
    """
    Load a pickle file from the given path.

    :param path: The path to the pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_pickle(obj, path: str):
    """
    Save a pickle file to the given path.

    :param obj: The object to save
    :param path: The path to save the object
    """
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)