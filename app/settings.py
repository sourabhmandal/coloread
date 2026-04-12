"""Application settings (.env, global configs) management using Pydantic."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local or remote OpenAI-compatible server settings.
    openai_base_url: str = "http://127.0.0.1:8080/v1"
    openai_api_key: Optional[SecretStr] = None

    openai_model: str = "gemma-4-E2B-it-GGUF"
    max_upload_size_mb: int = Field(default=20, ge=1, le=200)

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, v: str) -> str:
        # Keep URL normalized for clients that are sensitive to trailing slashes.
        return v.rstrip("/")

    @property
    def resolved_api_key(self) -> str:
        """Return a usable API key string for OpenAI-compatible backends."""
        if self.openai_api_key and self.openai_api_key.get_secret_value().strip():
            return self.openai_api_key.get_secret_value().strip()
        # Most local servers do not enforce auth, but the client expects a value.
        return "not-needed"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # BaseSettings pulls required fields from env/.env at runtime.
    return Settings()  # pyright: ignore[reportCallIssue]

__all__ = ["Settings", "get_settings"]
