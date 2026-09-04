from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"


class Settings(BaseSettings):
    ollama_model: str = "qwen3:1.7b"
    ollama_host: str = "http://127.0.0.1:11434"

    embedding_model: str = "all-MiniLM-L6-v2"

    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    collection_name: str = "sherlock_holmes"

    top_k: int = 5
    candidate_k: int = 20

    semantic_weight: float = 0.4
    keyword_weight: float = 0.6

    frontend_origin: str = "http://localhost:8501"

    api_title: str = "Sherlock Holmes RAG Assistant"
    api_version: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        extra="ignore"
    )


settings = Settings()