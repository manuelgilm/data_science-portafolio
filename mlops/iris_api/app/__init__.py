from fastapi import FastAPI
from app.db.db import init_db
from app.ml_core.model import load_model
from app.ml_core.model import train_model
from contextlib import asynccontextmanager

ml_models = {}
from app.routers.iris import iris_router
from app.routers.monitoring import monitor_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = load_model("app/ml_core/model.pkl")
    if ml_models["model"] is None:
        train_model()
        ml_models["model"] = load_model("app/ml_core/model.pkl")
    init_db()
    yield
    # clean up code goes here
    ml_models.clear()


version = "v1"
app = FastAPI(
    lifespan=lifespan,
    version=version,
    title="Basic Iris API",
    description="A simple API to demonstrate FastAPI with SQLModel",
)


app.include_router(iris_router, tags=["Iris"], prefix=f"/{version}/iris")
app.include_router(monitor_router, tags=["Monitoring"], prefix=f"/{version}/monitor")
