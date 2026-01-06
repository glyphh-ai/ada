from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database URL should include ?sslmode=require on Heroku
    database_url: str = Field(
        default="postgresql+psycopg://localhost:5432/glyph_ai",
        validation_alias=AliasChoices("GLYPH_DATABASE_URL", "DATABASE_URL"),
    )
    app_name: str = "glyph-ai-service"
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    app_base_url: str = "http://localhost:5173"
    sendgrid_api_key: str | None = None
    sendgrid_from_email: str = "no-reply@example.com"
    log_level: str = "INFO"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    origins_allow: str = "*"
    platform_api_base: str | None = None
    runtime_token: str | None = None
    runtime_version: str = "dev"
    usage_metrics_enabled: bool = False
    usage_metrics_flush_seconds: int = 60

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_prefix = "GLYPH_"
        extra = "ignore"

    def get_allowed_origins(self) -> list[str]:
        raw = self.origins_allow or ""
        origins = [token.strip() for token in raw.split(",") if token.strip()]
        return origins or ["*"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
