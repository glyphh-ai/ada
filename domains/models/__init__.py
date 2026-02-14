"""
Models domain - Model management and glyph storage.

This domain handles:
- Loading and unloading .glyphh models
- Namespace isolation for multi-model support
- Glyph CRUD operations
- Model configuration management
"""

from domains.models.manager import ModelManager, LoadedModel
from domains.models.storage import GlyphStorage
from domains.models.db_models import Glyph, Edge, ModelConfig, Token
from domains.models.schemas import (
    # Glyph schemas
    CreateGlyphRequest,
    CreateGlyphResponse,
    BatchCreateGlyphRequest,
    BatchCreateGlyphResponse,
    GlyphResponse,
    # Query schemas
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    ScoredGlyph,
    FactTreeRequest,
    TemporalPredictRequest,
    TemporalPredictResponse,
    # Model config schemas
    ModelConfigUpdate,
    ModelConfigResponse,
    ReEncodeRequest,
    ReEncodeResponse,
    ReEncodeStatusResponse,
    ClearDataResponse,
    # Deployment schemas
    DeploymentResponse,
    RuntimeStatusResponse,
    ModelInfoResponse,
    ModelsListResponse,
)

__all__ = [
    # Manager
    "ModelManager",
    "LoadedModel",
    # Storage
    "GlyphStorage",
    # DB Models
    "Glyph",
    "Edge",
    "ModelConfig",
    "Token",
    # Schemas
    "CreateGlyphRequest",
    "CreateGlyphResponse",
    "BatchCreateGlyphRequest",
    "BatchCreateGlyphResponse",
    "GlyphResponse",
    "SimilaritySearchRequest",
    "SimilaritySearchResponse",
    "ScoredGlyph",
    "FactTreeRequest",
    "TemporalPredictRequest",
    "TemporalPredictResponse",
    "ModelConfigUpdate",
    "ModelConfigResponse",
    "ReEncodeRequest",
    "ReEncodeResponse",
    "ReEncodeStatusResponse",
    "ClearDataResponse",
    "DeploymentResponse",
    "RuntimeStatusResponse",
    "ModelInfoResponse",
    "ModelsListResponse",
]
