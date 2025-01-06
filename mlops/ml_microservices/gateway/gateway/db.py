from sqlmodel import SQLModel
from sqlmodel import create_engine
from sqlmodel import Session
import os

DB_URL = "sqlite:///./user.db"
engine = create_engine(DB_URL, echo=True)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


import redis.asyncio as aioredis

token_blocklist = aioredis.from_url("redis://localhost:6379/0")


async def add_jti_to_blacklist(jti: str) -> None:
    await token_blocklist.set(name=jti, value="", ex=os.environ["JTI_EXPIRY"])


async def token_in_blacklist(jti: str) -> bool:
    jti = await token_blocklist.get(jti)
    return jti is not None
