from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
import httpx
from ml_service2.resources.roles import RoleChecker

version = "v1"
description = "API for the ML Microservice"
title = "ML Microservice"
app = FastAPI(title=title, description=description, version=version)


@app.get("/ml2/predict")
async def root(_: RoleChecker = RoleChecker(["admin", "user"])):
    return {"message": "Hello World from ML2"}
