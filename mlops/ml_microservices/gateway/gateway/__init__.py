from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
import httpx
from fastapi.middleware.cors import CORSMiddleware
from gateway.db import init_db
from contextlib import asynccontextmanager
from dotenv import load_dotenv


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

services = {"ml1": "http://localhost:8000", "ml2": "http://localhost:8001"}


async def forward_request(
    service_url: str, method: str, path: str, body=None, headers=None
):
    async with httpx.AsyncClient() as client:
        url = f"{service_url}{path}"
        response = await client.request(method, url, json=body, headers=headers)
        return response


@app.api_route(
    "/{service}/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"]
)
async def gateway(service: str, path: str, request: Request):
    service_url = services.get(service)
    print("service_url", service_url)
    if not service_url:
        raise HTTPException(status_code=404, detail="Service not found")
    method = request.method

    body = await request.json() if method in ["POST", "PUT", "PATCH"] else {}
    headers = dict(request.headers)
    response = await forward_request(service_url, method, f"/{path}", body, headers)
    return JSONResponse(status_code=200, content={"content": response.json()})
