from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Depends
from fastapi import status
from sqlmodel import Session
from app.schemas.iris_features import IrisFeatures
from app.schemas.iris_features import ModelResponse
from app.resources import Tracker
from app.db.db import get_session
from app import ml_models

iris_router = APIRouter()


@iris_router.post(
    "/predict", status_code=status.HTTP_200_OK, response_model=ModelResponse
)
async def get_prediction(
    iris_features: IrisFeatures,
    session: Session = Depends(get_session),
    manager: Tracker = Depends(Tracker),
):

    model = ml_models["model"]
    if model is None:
        return {"message": "Model not found"}
    process_features = [
        iris_features.sepal_length,
        iris_features.sepal_width,
        iris_features.petal_length,
        iris_features.petal_width,
    ]
    prediction = model.predict([process_features])
    proba = model.predict_proba([process_features])
    score = max(proba[0])

    # save prediction
    iris_prediction = {**iris_features.model_dump(), "prediction": int(prediction[0]), "score": score}
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
