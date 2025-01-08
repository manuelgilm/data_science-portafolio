from fastapi import APIRouter
from fastapi import Request
from fastapi import Depends
from gateway.services.resources import handle_request
from gateway.auth.dependencies import AccessTokenBearer

router = APIRouter()


@router.get("/{service}/{path:path}")
async def gateway(
    service: str,
    path: str,
    request: Request,
    user_details: AccessTokenBearer = Depends(AccessTokenBearer()),
):
    print("user_details", user_details)
    user_data = {
        "user_id": str(user_details["data"]["id"]),
        "user_role": user_details["data"]["role"],
    }
    
    return await handle_request(service, path, request, user_data)
