"""
Application settings via pydantic-settings.

All values can be overridden through environment variables or a .env file.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    DATABASE_URL: str = "postgresql://ragops:ragops@localhost:5432/ragops"

    # LLM
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval tuning
    TOP_K_DENSE: int = 20
    TOP_K_SPARSE: int = 20
    TOP_K_RERANK: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
