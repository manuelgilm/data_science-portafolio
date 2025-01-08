from typing import List 
from fastapi import Request 
from fastapi import HTTPException
from fastapi import status 

class RoleChecker:

    def __init__(self, allowed_roles:List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, request: Request):
        role = request.headers.get("user_role", None)
        if role is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Role header missing")
        
        if role not in self.allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return True