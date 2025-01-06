from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from sqlmodel import Session
from gateway.auth.models import User
from gateway.auth.schemas import CreateUser
from gateway.db import get_session
from gateway.auth.resources import UserManager


router = APIRouter()


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
