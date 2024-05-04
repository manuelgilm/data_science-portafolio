from factory_pattern_for_ml_comparison.models.base import CustomModel
from sklearn.ensemble import RandomForestClassifier

import pandas as pd  
import numpy as np

from typing import Optional
from typing import Union 
from typing import List
import logger

class CustomRFC(CustomModel):
    
    def __init__(self, model_name: str, feature_names: Optional[List[str]] = None):
        super().__init__(model_name)
        self.model = RandomForestClassifier()
        self.feature_names

    def fit(self, x, y):

        feature_names = x.columns
        targets = y.unique()

    def predict(self, X):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get_params(self):
        pass

    def set_params(self, **params):
        pass    

    def check_signature(self, x:Union[np.ndarray, pd.DataFrame], y:Union[np.ndarray, pd.Series]):

        if self.feature_names is None:
            if isinstance(x, pd.DataFrame):
                self.feature_names = x.columns
            else:
                logger.error("Feature Names not provided Generate them using generic names")
                self.feature_names = [f"f_{i}" for i in range(x.shape[1])]
        
        signature = {"feature_names": self.feature_names, "targets": y.unique()}

    def __get_model_signature(self):
        pass