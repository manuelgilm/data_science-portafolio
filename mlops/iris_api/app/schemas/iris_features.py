from pydantic import BaseModel
from pydantic import field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import Optional


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ModelResponse(BaseModel):
    model_config = {"validate_default": True}
    message: str
    prediction: float
    species: Optional[str] = ""

    @field_validator("species", mode="before")
    def set_species(cls, values: object, info: FieldValidationInfo):
        prediction = info.data["prediction"]
        species = cls.get_species(prediction)
        return species

    @classmethod
    def get_species(cls, prediction: float):
        if prediction == 0:
            return "Iris-setosa"
        elif prediction == 1:
            return "Iris-versicolor"
        else:
            return "Iris-virginica"
