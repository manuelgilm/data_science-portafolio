from fastapi import FastAPI

version = "v1"
description = "API for the ML Microservice"
title = "ML Microservice"
app = FastAPI(title=title, description=description, version=version)


@app.get("/ml1/predict")
async def root():
    return {"message": "Hello World from ML1"}
