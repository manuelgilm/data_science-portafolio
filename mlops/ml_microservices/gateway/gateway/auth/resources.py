from gateway.auth.models import User
from gateway.auth.schemas import CreateUser
from gateway.auth.jwt_utils import generate_password_hash
import uuid
from sqlmodel import Session
from sqlmodel import select
from typing import Dict


class UserManager:

    def create_user(self, user_data: CreateUser, session: Session):
        user_data_dict = user_data.model_dump()
        user_data_dict["hashed_password"] = generate_password_hash(
            user_data_dict["password"]
        )
        user_data_dict["id_"] = uuid.uuid4()
        user_data_dict["role"] = "user"

        user = User(**user_data_dict)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    def get_user(self, user_id: uuid.UUID, session: Session) -> User:
        statement = select(User).where(User.id_ == user_id)
        result = session.exec(statement).first()
        return result

    def get_user_by_email(self, email: str, session: Session) -> User:
        statement = select(User).where(User.email == email)
        result = session.exec(statement).first()
        return result

    def udate_user(self, user: User, user_data: CreateUser, session: Session) -> User:
        user_data_dict = user_data.model_dump()
        user_data_dict["hashed_password"] = generate_password_hash(
            user_data_dict["password"]
        )
        user_data_dict["id_"] = uuid.uuid4()

        updated_user = User(**user_data_dict)
        session.add(updated_user)
        session.commit()
        session.refresh(updated_user)
        return updated_user

    def delete_user(self, user: User, session: Session) -> Dict[str, str]:
        session.delete(user)
        session.commit()
        return {"message": "User deleted successfully"}
