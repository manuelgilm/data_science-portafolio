from sqlmodel import SQLModel
from sqlmodel import create_engine
from sqlmodel import Session

DB_URL = "sqlite:///./user.db"
engine = create_engine(DB_URL, echo=True)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
