from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
import mlflow

train_router = APIRouter()


@train_router.post("/train")
async def train_model():
    """
    Request to train a model.

    """
    # check if there is new data to train model

    return {"message": "Model trained successfully"}


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
