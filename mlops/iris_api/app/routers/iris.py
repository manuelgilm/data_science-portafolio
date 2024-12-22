from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Depends
from fastapi import status
from app.schemas.iris_features import IrisFeatures
from app import ml_models

iris_router = APIRouter()


@iris_router.post("/predict", status_code=status.HTTP_200_OK)
async def get_prediction(iris_features: IrisFeatures):

    model = ml_models['model']
    if model is None:
        return {"message": "Model not found"}
    process_features = [iris_features.sepal_length, iris_features.sepal_width, iris_features.petal_length, iris_features.petal_width]
    print(process_features)
    prediction = model.predict([process_features])
    print(prediction)
    return {"message": "Predicting Iris species", "prediction": float(prediction[0])}
