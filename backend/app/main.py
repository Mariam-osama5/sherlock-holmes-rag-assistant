from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.utils.logging_config import setup_logging
from app.core.config import settings
from app.api.routes.query import router as query_router


setup_logging()


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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