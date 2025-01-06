from sqlmodel import SQLModel
from sqlmodel import Field
from typing import Optional
import uuid


class User(SQLModel, table=True):
    id_: uuid.UUID = Field(default=uuid.uuid4(), primary_key=True, index=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    role: str
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False
