from fastapi import FastAPI 
from fastapi import APIRouter 

inference_router = APIRouter()

@inference_router.post("/inference")
async def inference():
    return {"message": "Inference successful"}
