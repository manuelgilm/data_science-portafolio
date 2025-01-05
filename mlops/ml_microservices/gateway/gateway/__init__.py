from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
import httpx 

app = FastAPI()

services = {
    "ml1": "http://localhost:8000/ml1",
    "ml2": "http://localhost:8001/ml2"

}

async def forward_request(service_url: str, method: str, path:str, body=None, headers=None):
    async with httpx.AsyncClient() as client:
        url = f"{service_url}{path}"
        response = await client.request(method, url, json=body, headers=headers)
        return response


@app.api_route("/{service}/{path}", methods=["GET", "POST", "PUT", "PATCH"])
async def gateway(service: str, path: str, request: Request):
    service_url = services.get(service)
    if not service_url:
        raise HTTPException(status_code=404, detail="Service not found")
    method = request.method
    body = await request.json() if request.method in ["POST", "PUT", "PATCH"] else None
    headers = dict(request.headers)
    response = await forward_request(service_url, method, f"/{path}", body, headers)
    return JSONResponse(status_code=response.status_code, content=response.json())