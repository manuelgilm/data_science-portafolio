from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

monitor_router = APIRouter()


@monitor_router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}
