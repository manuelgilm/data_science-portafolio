from fastapi.security import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.exceptions import HTTPException
from fastapi import Depends
from fastapi import Request
from fastapi import status
from typing import Optional
from typing import Dict
from typing import Any
from gateway.auth.jwt_utils import decode_token


class TokenBearer(HTTPBearer):

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(
        self, request: Request
    ) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        token = credentials.credentials
        token_data = decode_token(token)
        if token and not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        return token_data


class AccessTokenBearer(TokenBearer):

    def verify_token_data(self, token_data: Dict[str, Any]):
        """
        Verify the token data

        :param token_data: Token data to be verified

        """

        if token_data and token_data["refresh"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )


class RefreshTokenBearer(TokenBearer):

    def verify_token_data(self, token_data: Dict[str, Any]):
        """
        Verify the token data

        :param token_data: Token data to be verified

        """

        if token_data and not token_data["refresh"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
