from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Depends
from ml_service1.resources.roles import RoleChecker

version = "v1"
description = "API for the ML Microservice"
title = "ML Microservice"
app = FastAPI(title=title, description=description, version=version)


@app.get("/ml1/predict")
async def root(_: RoleChecker = Depends(RoleChecker(["admin", "user"]))):
    return {"message": "Hello World from ML1"}
