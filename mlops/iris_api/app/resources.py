from sqlmodel import Session
from sqlmodel import select
from app.db.models import Prediction
from app.db.models import Drift
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

    def save_drift(self, drift_data: Dict[str, Any], session: Session) -> Drift:
        drift = Drift(**drift_data)
        drift.id = uuid.uuid4()

        session.add(drift)
        session.commit()
        session.refresh(drift)
        return drift

    def get_drift_info(self, session: Session) -> list[Drift]:
        statement = select(Drift)
        return session.exec(statement).all()
