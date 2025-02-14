from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
import mlflow
from train_service.src.train import Trainer
from train_service.src.tracking import get_or_create_experiment
train_router = APIRouter()


@train_router.post("/train")
async def train_model():
    """
    Request to train a model.

    """
    experiment = get_or_create_experiment("train_model")
    try:
        train_service = Trainer()
        train_service.fit_model()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Model trained successfully"})


@train_router.get("/models")
async def get_models():
    try:
        models = mlflow.search_model_versions()
        return JSONResponse(status_code=status.HTTP_200_OK, content=models.to_json())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@train_router.post("/experiment")
async def create_experiment(experiment_name: str):
    try:
        mlflow.create_experiment(experiment_name)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": "Experiment created successfully"},
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
