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

    # Admin API
    admin_api_base_url: str = "http://localhost:8000"
    admin_api_key: str = ""

    # Email channel (opt-in via ADMIN_EMAIL_ENABLED=true)
    admin_email_enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_user: str = ""
    smtp_password: str = ""
    admin_email: str = "admin@citypark.example.com"
    email_from: str = "no-reply@citypark.example.com"
    # Secret for signing single-use action tokens — must be overridden in prod
    email_token_secret: str = "change-me-in-production"
    email_token_max_age_seconds: int = 86400          # 24 h

    # MCP reservations server (Stage 3). The export is active only when
    # MCP_API_KEY is set (same shared secret as the server) and the flag is on.
    mcp_export_enabled: bool = True
    mcp_server_url: str = "http://127.0.0.1:8765/mcp"
    mcp_api_key: str = ""
    mcp_client_timeout: float = 5.0                   # seconds per attempt
    mcp_client_max_attempts: int = 3                  # background/retry worker
    mcp_client_backoff_seconds: float = 0.5           # doubles each retry
    mcp_retry_interval_seconds: int = 60              # API background worker

    # LangGraph orchestration (Stage 4)
    checkpoint_db: str = "./checkpoints.db"           # SQLite checkpointer; ":memory:" for tests
    approval_ttl_hours: int = 24                      # no admin reply after this -> EXPIRED
    approval_expiry_enabled: bool = True
    approval_expiry_check_seconds: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
