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
    return await handle_request(service, path, request)


# @router.post("/{service}/{path:path}")
# async def gateway(service: str, path: str):
#     return {"service": service, "path": path}

# @router.delete("/{service}/{path:path}")
# async def gateway(service: str, path: str):
#     return {"service": service, "path": path}

# @router.put("/{service}/{path:path}")
# async def gateway(service: str, path: str):
#     return {"service": service, "path": path}

# @router.patch("/{service}/{path:path}")
# async def gateway(service: str, path: str):
#     return {"service": service, "path": path}

# @router.options("/{service}/{path:path}")
# async def gateway(service: str, path: str):
#     return {"service": service, "path": path}
