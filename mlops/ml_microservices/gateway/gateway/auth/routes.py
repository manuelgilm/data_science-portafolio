from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from fastapi.responses import JSONResponse
from sqlmodel import Session
from gateway.auth.models import User
from gateway.auth.schemas import CreateUser
from gateway.auth.schemas import LoginUser
from gateway.db import get_session
from gateway.auth.resources import UserManager
from gateway.auth.jwt_utils import create_token
from gateway.auth.jwt_utils import verify_password
from datetime import timedelta
from gateway.auth.dependencies import AccessTokenBearer
from gateway.db import add_jti_to_blacklist


REFRESH_TOKEN_EXPIRY = ""
router = APIRouter()
access_token_bearer = AccessTokenBearer()


@router.post("/signup", status_code=status.HTTP_201_CREATED, response_model=User)
async def signup(
    user_data: CreateUser,
    session: Session = Depends(get_session),
    manager: UserManager = Depends(UserManager),
):

    user = manager.get_user_by_email(user_data.email, session)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists"
        )

    user = manager.create_user(user_data=user_data, session=session)
    return user


@router.post("/login", response_model=User)
async def login(
    user_login: LoginUser,
    session: Session = Depends(get_session),
    manager: UserManager = Depends(UserManager),
):

    user = manager.get_user_by_email(user_login.email, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if not verify_password(user_login.password, user_login.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password"
        )

    token = create_token(
        data={"email": user.email, "id": str(user.id_), "role": "user"}
    )

    refresh_token = create_token(
        data={
            "email": user.email,
            "id": str(user.id_),
            "role": "user",
        },
        refresh=True,
        expiry=timedelta(days=REFRESH_TOKEN_EXPIRY),
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"access_token": token, "refresh_token": refresh_token},
    )


@router.post("/logout")
async def revoke_token(user_details: AccessTokenBearer = Depends(access_token_bearer)):
    jti = user_details["jti"]

    await add_jti_to_blacklist(jti)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Token revoked successfully"},
    )
