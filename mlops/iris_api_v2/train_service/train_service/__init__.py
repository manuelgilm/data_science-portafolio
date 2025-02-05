from fastapi import FastAPI

from train_service.api.routes.train_route import train_router
from train_service.api.settings import lifespan
app = FastAPI(lifespan=lifespan)

app.include_router(train_router, tags=["train"])



