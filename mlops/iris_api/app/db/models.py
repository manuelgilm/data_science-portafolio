from sqlmodel import SQLModel
from sqlmodel import Field
import uuid
from datetime import datetime
from typing import Optional


class Prediction(SQLModel, table=True):
    id: uuid.UUID = Field(default=uuid.uuid4(), primary_key=True, index=True)
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    ground_truth: Optional[int] = None
    prediction: int
    created_at: datetime = Field(default=datetime.now())
