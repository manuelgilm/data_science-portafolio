from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Depends
from fastapi import status
from sqlmodel import Session
from app.routers.utils import process_features
from app.schemas.iris_features import IrisFeatures
from app.schemas.iris_features import ModelResponse
from app.resources import Tracker
from app.db.db import get_session
from app import ml_models
import numpy as np
iris_router = APIRouter()

@iris_router.post(
    "/predict", status_code=status.HTTP_200_OK, response_model=ModelResponse
)
async def get_prediction(
    iris_features: IrisFeatures,
    session: Session = Depends(get_session),
    manager: Tracker = Depends(Tracker)
):

    model = ml_models["model"]
    if model is None:
        return {"message": "Model not found"}
    
    detector = ml_models["detector"]
    if detector is None:
        return {"message": "Detector not found"}
    
    features = process_features(iris_features)
    prediction = model.predict(features)
    proba = model.predict_proba(features)
    score = max(proba[0])

    # detecting drift
    print("Detecting drift")
    print("input")
    print(np.array(features))
    drift = detector.predict(np.array(features), drift_type="batch", return_p_val=True, return_distance=True)
    print(drift)
    # save prediction
    iris_prediction = {
        **iris_features.model_dump(),
        "prediction": int(prediction[0]),
        "score": score,
        "ground_truth": iris_features.label.value,
    }
    manager.save_prediction(iris_prediction, session)
    # create response
    response = ModelResponse(
        message="Prediction successful", prediction=int(prediction[0])
    )
    return response


@iris_router.get("/predictions", status_code=status.HTTP_200_OK)
async def get_predictions(
    session: Session = Depends(get_session), manager: Tracker = Depends(Tracker)
):
    predictions = manager.get_model_predictions(session)
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No predictions found"
        )
    return predictions
