from factory_pattern_for_ml_comparison.models.base import CustomModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np

from typing import Optional
from typing import Union
from typing import List


def logger(text):
    print(text)


class CustomRFC(CustomModel):

    def __init__(self, model_name: str, feature_names: Optional[List[str]] = None):
        super().__init__(model_name)
        self.model = RandomForestClassifier()
        self.feature_names = feature_names

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get_params(self):
        pass

    def set_params(self, **params):
        pass


class CustomDecisionTree(CustomModel):

    def __init__(self, model_name: str, feature_names: Optional[List[str]] = None):
        super().__init__(model_name)
        self.model = DecisionTreeClassifier()
        self.feature_names = feature_names

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get_params(self):
        pass

    def set_params(self, **params):
        pass


class CustomAdaBoost(CustomModel):

    def __init__(self, model_name: str, feature_names: Optional[List[str]] = None):
        super().__init__(model_name)
        self.model = AdaBoostClassifier()
        self.feature_names = feature_names

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get_params(self):
        pass

    def set_params(self, **params):
        pass
