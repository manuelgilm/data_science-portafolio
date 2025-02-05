from pydantic import BaseModel
from pydantic import Optional


class ModelInput(BaseModel):
    petal_length: float
    petal_width: float
    sepal_length: float
    sepal_width: float


class InferenceData(BaseModel):
    inputs: ModelInput
    label: Optional[str] = None



class ModelSignature(BaseModel):
    pass     