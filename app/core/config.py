"""
NexusInfer — Application Configuration

Centralizes all configuration using pydantic-settings.
Reads from environment variables and .env files for 12-factor compliance.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Application ---
    APP_NAME: str = "NexusInfer"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Redis (Message Broker) ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./nexusinfer.db"
    # For PostgreSQL (Day 4+):
    # DATABASE_URL: str = "postgresql://nexus:nexus@localhost:5432/nexusinfer"

    # --- ML Models ---
    DEFAULT_MODEL: str = "sentiment"
    MODEL_CACHE_DIR: str = "./models_cache"
    # Maximum input text length (characters)
    MAX_INPUT_LENGTH: int = 5000

    # --- Worker ---
    WORKER_CONCURRENCY: int = 4
    TASK_TIMEOUT: int = 300  # seconds

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton of the application settings.
    Using lru_cache ensures the .env file is read only once.
    """
    return Settings()
