from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from typing import Dict 
from typing import Any

def get_processing_pipeline(config:Dict[str, Any])->Pipeline:
    """
    Get the processing pipeline for the data
    """
    pass