from contextlib import asynccontextmanager
from api.dependencies import build_app_state
from api.routes import router
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app:FastAPI):
    build_app_state(app)
    yield

app = FastAPI(
    title="RAG Experiment Lab API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)