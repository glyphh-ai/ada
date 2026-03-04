"""
Runtime configuration management using Pydantic settings.

All configuration is read from environment variables with sensible defaults.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


@dataclass
class TierConfig:
    """Resolved plan limits for the authenticated user/deployment."""

    tier: str                   # "free" | "developer" | "team" | "enterprise"
    max_models: int             # -1 = unlimited
    max_glyphs_per_model: int   # -1 = unlimited
    rate_limit_per_minute: int  # -1 = unlimited
    allow_commercial: bool
    allow_external_db: bool

    @classmethod
    def free(cls) -> "TierConfig":
        return cls(
            tier="free", max_models=3, max_glyphs_per_model=10_000,
            rate_limit_per_minute=60, allow_commercial=False, allow_external_db=False,
        )

    @classmethod
    def developer(cls) -> "TierConfig":
        return cls(
            tier="developer", max_models=10, max_glyphs_per_model=250_000,
            rate_limit_per_minute=300, allow_commercial=True, allow_external_db=True,
        )

    @classmethod
    def team(cls) -> "TierConfig":
        return cls(
            tier="team", max_models=-1, max_glyphs_per_model=-1,
            rate_limit_per_minute=1_000, allow_commercial=True, allow_external_db=True,
        )

    @classmethod
    def enterprise(cls) -> "TierConfig":
        return cls(
            tier="enterprise", max_models=-1, max_glyphs_per_model=-1,
            rate_limit_per_minute=-1, allow_commercial=True, allow_external_db=True,
        )

    @classmethod
    def from_jwt_claims(cls, payload: Dict[str, Any]) -> "TierConfig":
        """Resolve tier from JWT claims. Named tier sets the baseline; individual
        claim overrides allow custom plans without new tier names."""
        tier = payload.get("tier", "free")
        if tier == "developer":
            base = cls.developer()
        elif tier == "team":
            base = cls.team()
        elif tier == "enterprise":
            base = cls.enterprise()
        else:
            base = cls.free()
        return cls(
            tier=base.tier,
            max_models=payload.get("max_models", base.max_models),
            max_glyphs_per_model=payload.get("max_glyphs_per_model", base.max_glyphs_per_model),
            rate_limit_per_minute=payload.get("rate_limit_per_minute", base.rate_limit_per_minute),
            allow_commercial=payload.get("allow_commercial", base.allow_commercial),
            allow_external_db=payload.get("allow_external_db", base.allow_external_db),
        )

    def check_model_limit(self, current_count: int) -> bool:
        """True if another model can be created."""
        return self.max_models == -1 or current_count < self.max_models

    def check_glyph_limit(self, current_count: int) -> bool:
        """True if another glyph can be written."""
        return self.max_glyphs_per_model == -1 or current_count < self.max_glyphs_per_model

    def glyph_warning_threshold(self) -> Optional[int]:
        """Return the 90% warning count, or None if unlimited."""
        if self.max_glyphs_per_model == -1:
            return None
        return int(self.max_glyphs_per_model * 0.9)


class Settings(BaseSettings):
    """Runtime configuration settings"""

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8002, description="Server port")

    # Deployment mode
    deployment_mode: Literal["local", "self-hosted", "cloud"] = Field(
        default="local",
        description="Deployment mode: local (no auth), self-hosted, or cloud"
    )

    # Database — optional; auto-defaults to SQLite in local mode
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL. Auto-defaults to SQLite in local mode if not set."
    )

    # Storage backend
    storage_backend: Literal["auto", "pgvector", "sqlite", "memory"] = Field(
        default="auto",
        description=(
            "Storage backend for glyph persistence. "
            "'auto' detects from database_url (sqlite:// → sqlite, postgresql:// → pgvector). "
            "'pgvector' requires DATABASE_URL pointing to PostgreSQL + pgvector. "
            "'sqlite' uses SQLite with Python cosine similarity (dev/small datasets). "
            "'memory' reserved for future in-process backends (Pinecone, Qdrant, etc.)."
        ),
    )

    @property
    def resolved_database_url(self) -> str:
        """Return the effective database URL, auto-defaulting to SQLite in local mode."""
        if self.database_url:
            return self.database_url
        if self.deployment_mode == "local":
            return "sqlite+aiosqlite:///./glyphh_dev.db"
        raise ValueError(
            "DATABASE_URL is required for non-local deployment modes. "
            "Set it with: export DATABASE_URL=postgresql+asyncpg://..."
        )

    @property
    def resolved_storage_backend(self) -> str:
        """Return the effective storage backend, auto-detecting from URL if 'auto'."""
        if self.storage_backend != "auto":
            return self.storage_backend
        url = self.resolved_database_url
        if "sqlite" in url:
            return "sqlite"
        if "postgresql" in url or "postgres" in url:
            return "pgvector"
        return "sqlite"  # safe fallback
    
    # JWT Authentication — shared secret with Platform for web UI sessions.
    # Database tokens are the primary auth for CLI/API. Platform JWTs are
    # accepted for browser-based dashboard sessions (POST /auth/login on Platform).
    jwt_secret_key: Optional[str] = Field(
        default=None,
        description="Platform JWT secret (HS256). Set to same value as Platform's JWT_SECRET_KEY."
    )
    
    # NL Query (enabled by default for studio chat functionality)
    enable_nl_query: bool = Field(
        default=True,
        description="Enable natural language query interface"
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
    
    # Vector dimension limit
    max_vector_dimension: int = Field(
        default=2000,
        description="Maximum vector dimension for models. Cloud default is 2000 (pgvector HNSW index limit). "
                    "Local installs can increase this but will lose index-based similarity search."
    )

    @property
    def resolved_max_vector_dimension(self) -> int:
        """Return effective dimension limit based on storage backend.

        SQLite stores vectors as JSON text — no size constraint applies.
        pgvector HNSW index supports up to 2000 dims; brute-force up to 16384.
        Use max_vector_dimension (env: MAX_VECTOR_DIMENSION) to override for
        pgvector deployments that don't need the HNSW index.
        """
        if self.resolved_storage_backend == "sqlite":
            return 65536  # No practical limit for SQLite JSON storage
        return self.max_vector_dimension
    
    # Model storage
    model_storage_path: str = Field(
        default="/data/models",
        description="Persistent directory for model data"
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
    """Validate settings on startup - raises if invalid."""
    settings = get_settings()

    if settings.deployment_mode != "local":
        if not settings.database_url:
            raise ValueError(
                "DATABASE_URL is required for non-local deployment modes"
            )
