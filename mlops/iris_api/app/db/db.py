from sqlmodel import SQLModel 
from sqlmodel import create_engine
from sqlmodel import Session

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session