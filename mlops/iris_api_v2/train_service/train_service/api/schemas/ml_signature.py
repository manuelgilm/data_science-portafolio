from pydantic import BaseModel


class ModelInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ModelOutput(BaseModel):
    prediction: str
