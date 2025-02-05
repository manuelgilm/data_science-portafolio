from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
import mlflow 
train_router = APIRouter()

@train_router.post("/train")
async def train_model():
    # Logic to train the model
    # steps:
    # 1. Get the last date the model was trained.
    # 2. Verify if the there is new data to train the model.
    # 2.1 If there is no new data, return a message saying that the model is already up to date.
    # 3. Get the test and train data.
    # 4. Train the model.
    return {"message": "Model trained successfully"}

@train_router.post("/experiment")
async def create_experiment(experiment_name: str):
    try:
        mlflow.create_experiment(experiment_name)
        return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message": "Experiment created successfully"})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))