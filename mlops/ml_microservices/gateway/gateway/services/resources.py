import httpx
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse

services = {
    "ml1": "http://ml_service1:8001",
    "ml2": "http://ml_service2:8002",
    "auth": "http://gateway:8000",
}


async def forward_request(
    service_url: str, method: str, path: str, body=None, headers=None
):
    async with httpx.AsyncClient() as client:
        url = f"{service_url}{path}"
        response = await client.request(method, url, json=body, headers=headers)
        return response


async def handle_request(service: str, path: str, request: Request):
    service_url = services.get(service)
    print("service_url", service_url)
    if not service_url:
        raise HTTPException(status_code=404, detail="Service not found")
    method = request.method

    body = await request.json() if method in ["POST", "PUT", "PATCH"] else {}
    headers = dict(request.headers)
    response = await forward_request(service_url, method, f"/{path}", body, headers)
    return JSONResponse(status_code=200, content={"content": response.json()})
