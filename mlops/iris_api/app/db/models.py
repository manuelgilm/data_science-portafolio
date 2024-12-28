from sqlmodel import SQLModel
from sqlmodel import Field
import uuid
from datetime import datetime
from typing import Optional
from typing import List


class Prediction(SQLModel, table=True):
    id: uuid.UUID = Field(default=uuid.uuid4(), primary_key=True, index=True)
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    ground_truth: Optional[int] = None
    prediction: int
    score: float
    created_at: datetime = Field(default=datetime.now())


class Drift(SQLModel, table=True):
    id: uuid.UUID = Field(default=uuid.uuid4(), primary_key=True, index=True)
    prediction_id: uuid.UUID = Field(foreign_key="prediction.id")
    drift_type: str
    threshold: float
    p_value_feature_0: float
    p_value_feature_1: float
    p_value_feature_2: float
    p_value_feature_3: float
    distance_feature_0: float
    distance_feature_1: float
    distance_feature_2: float
    distance_feature_3: float
    created_at: datetime = Field(default=datetime.now())
