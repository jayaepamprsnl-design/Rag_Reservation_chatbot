from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM (OpenAI-compatible endpoint)
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_timeout: float = 30.0
    llm_max_retries: int = 3

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "parking_static"

    # SQLite / SQLAlchemy
    database_url: str = "sqlite:///./parking.db"

    # App server
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # RAG
    retrieval_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
