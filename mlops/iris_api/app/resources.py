from sqlmodel import Session
from sqlmodel import select
from app.db.models import Prediction
from typing import Any
from typing import Dict
import uuid


class Tracker:

    def save_prediction(
        self, iris_data: Dict[str, Any], session: Session
    ) -> Prediction:

        prediction = Prediction(**iris_data)
        prediction.id = uuid.uuid4()

        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction

    def get_model_predictions(self, session: Session) -> list[Prediction]:
        statement = select(Prediction)
        return session.exec(statement).all()
