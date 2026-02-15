"""
Runtime configuration management using Pydantic settings.

All configuration is read from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration settings"""
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # Deployment mode
    deployment_mode: Literal["local", "self-hosted", "cloud"] = Field(
        default="local",
        description="Deployment mode: local (no auth), self-hosted, or cloud"
    )
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/glyphh_runtime",
        description="PostgreSQL connection URL"
    )
    
    # JWT Authentication
    jwt_secret_key: Optional[str] = Field(
        default=None,
        description="JWT signing key (required for self-hosted/cloud)"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    
    # Licensing
    license_key: Optional[str] = Field(
        default=None,
        description="License key for self-hosted deployments"
    )
    platform_api_url: str = Field(
        default="https://platform.glyphh.com/api",
        description="Platform API URL for license validation"
    )
    license_grace_period_days: int = Field(
        default=7,
        description="Grace period in days when license validation fails"
    )
    
    # NL Query (enabled by default for studio chat functionality)
    enable_nl_query: bool = Field(
        default=True,
        description="Enable natural language query interface"
    )
    nl_model: str = Field(
        default="microsoft/Phi-3.5-mini-instruct",
        description="Model to use for NL query translation"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(
        default=60,
        description="Default rate limit per minute"
    )
    
    # Resource quotas
    default_model_memory_mb: int = Field(
        default=1024,
        description="Default memory quota per model in MB"
    )
    default_model_storage_gb: int = Field(
        default=10,
        description="Default storage quota per model in GB"
    )
    
    # Local mode limits (development)
    local_mode_max_models: int = Field(
        default=10,
        description="Maximum models in local mode (development limit)"
    )
    local_mode_max_glyphs: int = Field(
        default=1000,
        description="Maximum glyphs per model in local mode (development limit)"
    )
    
    # Vector dimension limit
    max_vector_dimension: int = Field(
        default=2000,
        description="Maximum vector dimension for models. Cloud default is 2000 (pgvector HNSW index limit). "
                    "Local installs can increase this but will lose index-based similarity search."
    )
    
    # Model storage
    model_storage_path: str = Field(
        default="/data/models",
        description="Persistent directory for deployed .glyphh model files"
    )
    
    # Edge generation
    edge_generation_strategy: Literal["eager", "lazy", "on-demand"] = Field(
        default="lazy",
        description="Edge generation strategy"
    )
    edge_ttl_hours: int = Field(
        default=24,
        description="Edge cache TTL in hours"
    )
    
    # Beam search defaults
    default_beam_width: int = Field(default=5, description="Default beam width")
    default_max_tree_depth: int = Field(default=3, description="Default max tree depth")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    
    # Telemetry
    telemetry_enabled: bool = Field(default=True, description="Enable telemetry")
    telemetry_endpoint: Optional[str] = Field(
        default=None,
        description="Telemetry endpoint URL"
    )
    
    # CORS — in local mode, allow all origins for dev flexibility
    # In cloud/self-hosted mode, use explicit production origins
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:4321",
            "http://localhost:5173",
        ],
        description="Allowed CORS origins (ignored in local mode where * is used)"
    )
    cors_origins_production: List[str] = Field(
        default=[
            "https://studio.glyphh.com",
            "https://app.glyphh.com",
            "https://*.glyphh.com",
        ],
        description="Allowed CORS origins for production deployments"
    )
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: Optional[str], info) -> Optional[str]:
        """Require JWT secret for non-local deployments"""
        # Note: This validation happens at startup
        # We'll check deployment_mode in the application
        return v
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars not defined in Settings


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def validate_settings() -> None:
    """Validate settings on startup - raises if invalid"""
    settings = get_settings()
    
    if settings.deployment_mode != "local":
        if not settings.jwt_secret_key:
            raise ValueError(
                f"JWT_SECRET_KEY is required for {settings.deployment_mode} deployment mode"
            )
        
        if settings.deployment_mode == "self-hosted" and not settings.license_key:
            raise ValueError(
                "LICENSE_KEY is required for self-hosted deployment mode"
            )
