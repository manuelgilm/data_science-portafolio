from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gateway.db import init_db
from gateway.auth.routes import router as auth_router
from gateway.services.routes import router as services_router
from contextlib import asynccontextmanager
from dotenv import load_dotenv


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(services_router, prefix="/services", tags=["services"])
