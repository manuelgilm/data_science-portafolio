from inspect import getmembers
from inspect import isclass
from inspect import isabstract

from typing import List
from typing import Optional
from typing import Dict
from typing import Any

from factory_pattern_for_ml_comparison.models import classifiers

MODEL_TYPES = ["classifiers", "regressors"]


class ModelFactory:

    def get_model_list(self, model_type: Optional[str] = None) -> List[str]:
        """
        Get a list of all available models

        :param model_type: Type of model to get

        """

        if model_type not in MODEL_TYPES and model_type is not None:
            raise ValueError(f"Invalid model type: {model_type}")

        if model_type is None:
            # return all models
            return self.__all_models()
        else:
            return self.__all_models()[model_type]

    def __all_models(self) -> Dict[str, Any]:
        """
        Get all models
        """

        return {
            "classifiers": self.__get_classifiers(),
        }

    def __get_classifiers(self) -> List[str]:
        """
        Get a list of all available models

        """
        models = getmembers(
            classifiers,
            lambda o: (
                o
                if isclass(o)
                and not isabstract(o)
                and o.__module__ == classifiers.__name__
                else None
            ),
        )

        return models
