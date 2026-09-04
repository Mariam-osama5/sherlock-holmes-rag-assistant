from pathlib import Path
from pydantic_settings import BaseSettings


# Project root:
# RAG_Assistant_Project/
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Ollama
    ollama_model: str = "qwen3:1.7b"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chroma
    vector_store_path: Path = (
        PROJECT_ROOT / "data" / "vector_store"
    )
    collection_name: str = "sherlock_holmes"

    # Retrieval
    top_k: int = 5

    # API
    api_title: str = "Sherlock Holmes RAG Assistant"
    api_version: str = "1.0.0"


settings = Settings()