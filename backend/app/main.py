from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.generation import initialize_llm
from app.services.retrieval import initialize_retriever
from app.utils.logging_config import setup_logging


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources once when the API starts."""

    initialize_retriever()
    initialize_llm()

    yield


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(query_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }