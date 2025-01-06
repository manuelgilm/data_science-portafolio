from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
import httpx

version = "v1"
description = "API for the ML Microservice"
title = "ML Microservice"
app = FastAPI(title=title, description=description, version=version)


@app.get("/ml2/predict")
async def root():
    return {"message": "Hello World from ML2"}
