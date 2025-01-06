from datetime import datetime
from datetime import timedelta

import jwt
import uuid
from passlib.context import CryptContext
from typing import Dict
import os


password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_password_hash(password: str) -> str:
    """
    Generate a password hash

    :param password: str: The password to hash
    :return: str: The hashed password
    """
    return password_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hashed password

    :param password: Password to be verified
    :param hashed_password: Hashed password to be compared against
    :return: Boolean indicating if the password is valid
    """
    return password_context.verify(password, hashed_password)


def create_token(data: Dict, expiry: timedelta = None, refresh: bool = False) -> str:
    """
    Create a JWT token

    :param data: Data to be stored in the token
    :param expiry: Expiry time for the token
    :param refresh: Boolean indicating if the token is a refresh token
    :return: JWT token
    """
    payload = {}
    payload["data"] = data
    payload["exp"] = datetime.now() + (
        expiry
        if expiry is not None
        else timedelta(seconds=int(os.environ["ACCESS_TOKEN_EXPIRY"]))
    )
    payload["jti"] = str(uuid.uuid4())
    payload["refresh"] = refresh
    token = jwt.encode(
        payload=payload,
        key=os.environ["JWT_SECRET"],
        algorithm=os.environ["JWT_ALGORITHM"],
    )

    return token


def decode_token(token: str) -> Dict:
    """
    Decode a JWT token

    :param token: JWT token to be decoded
    :return: Decoded token
    """
    try:
        token_data = jwt.decode(
            jwt=token,
            key=os.environ["JWT_SECRET"],
            algorithms=[os.environ["JWT_ALGORITHM"]],
        )
        return token_data
    except jwt.PyJWKError as e:
        raise e
