"""Application configuration via environment variables."""

import os
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://localhost/birdwatch",
        alias="DATABASE_URL",
    )
    db_pool_min_size: int = Field(default=2, alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=10, alias="DB_POOL_MAX_SIZE")

    # Paths
    media_dir: Path = Field(default=Path("media"), alias="MEDIA_DIR")

    # LLM/Embeddings
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")

    # App settings
    debug: bool = Field(default=False, alias="DEBUG")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
