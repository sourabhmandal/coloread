from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required secret
    github_token: SecretStr

    # Constrained values
    openai_model: Literal["gpt-4o-mini", "gpt-4.1-mini"] = "gpt-4o-mini"
    max_upload_size_mb: int = Field(default=20, ge=1, le=200)

    @field_validator("github_token")
    @classmethod
    def validate_github_token(cls, v: SecretStr) -> SecretStr:
        token = v.get_secret_value().strip()
        if not token:
            raise ValueError("GITHUB_TOKEN cannot be empty")
        if not (token.startswith("ghp_") or token.startswith("github_pat_")):
            raise ValueError("GITHUB_TOKEN format looks invalid")
        return v

    @model_validator(mode="after")
    def validate_settings_combination(self):
        # Example cross-field rule
        if self.openai_model == "gpt-4.1-mini" and self.max_upload_size_mb > 100:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be <= 100 for gpt-4.1-mini")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # BaseSettings pulls required fields from env/.env at runtime.
    return Settings()  # pyright: ignore[reportCallIssue]