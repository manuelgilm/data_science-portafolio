from pydantic import BaseModel


class CreateUser(BaseModel):
    username: str
    password: str
    email: str


class LoginUser(BaseModel):
    email: str
    password: str
