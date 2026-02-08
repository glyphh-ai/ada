"""
Pydantic schemas for API request/response validation.

These schemas define the contract between the Runtime API and clients.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Glyph Schemas
# ============================================================================

class GlyphBase(BaseModel):
    """Base glyph schema"""
    concept_text: str = Field(..., description="Concept text to encode", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class CreateGlyphRequest(GlyphBase):
    """Request to create a single glyph"""
    pass


class BatchCreateGlyphRequest(BaseModel):
    """Request to create multiple glyphs"""
    concepts: List[str] = Field(..., description="List of concept texts to encode", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Shared metadata for all glyphs")


class GlyphResponse(BaseModel):
    """Response containing a glyph"""
    id: UUID
    org_id: str
    model_id: str
    concept_text: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CreateGlyphResponse(BaseModel):
    """Response after creating a glyph"""
    glyph_id: UUID
    org_id: str
    model_id: str
    created_at: datetime


class BatchCreateGlyphResponse(BaseModel):
    """Response after creating multiple glyphs"""
    created: List[CreateGlyphResponse]
    failed: List[Dict[str, Any]] = Field(default_factory=list)
    total: int
    success_count: int
    failure_count: int


# ============================================================================
# Query Schemas
# ============================================================================

class SimilaritySearchRequest(BaseModel):
    """Request for similarity search"""
    query: str = Field(..., description="Query text to search for", min_length=1)
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")
    include_embeddings: bool = Field(default=False, description="Include embeddings in response")


class ScoredGlyph(BaseModel):
    """Glyph with similarity score"""
    glyph: GlyphResponse
    similarity_score: float = Field(..., ge=0, le=1)
    security_weight: float = Field(default=1.0, ge=0, le=1)
    final_score: float = Field(..., ge=0, le=1)


class SimilaritySearchResponse(BaseModel):
    """Response from similarity search"""
    results: List[ScoredGlyph]
    total_count: int
    query_time_ms: float


class FactTreeRequest(BaseModel):
    """Request for fact tree generation"""
    claim: str = Field(..., description="Claim to verify", min_length=1)
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum tree depth")
    branching_factor: int = Field(default=5, ge=1, le=20, description="Maximum children per node")


class FactTreeNode(BaseModel):
    """Node in a fact tree"""
    id: str
    claim: str
    supporting_glyphs: List[UUID]
    confidence: float = Field(..., ge=0, le=1)
    children: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    """Citation to a source glyph"""
    glyph_id: UUID
    concept_text: str
    relevance_score: float


class FactTreeResponse(BaseModel):
    """Response from fact tree generation"""
    root_claim: str
    nodes: List[FactTreeNode]
    confidence: float
    citations: List[Citation]
    generation_time_ms: float


class TemporalPredictRequest(BaseModel):
    """Request for temporal prediction"""
    current_state: List[str] = Field(..., description="Current state concepts", min_length=1)
    steps_ahead: int = Field(default=1, ge=1, le=10, description="Steps to predict ahead")
    beam_width: int = Field(default=5, ge=1, le=20, description="Beam search width")
    direction: str = Field(default="forward", description="Prediction direction: forward or backward")
    
    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")
        return v


class Delta(BaseModel):
    """Change between states"""
    glyph_id: UUID
    change_type: str  # added, removed, modified
    magnitude: float


class PredictedState(BaseModel):
    """Predicted future/past state"""
    state_glyphs: List[UUID]
    confidence: float = Field(..., ge=0, le=1)
    path_score: float
    deltas: List[Delta]


class TemporalPredictResponse(BaseModel):
    """Response from temporal prediction"""
    predictions: List[PredictedState]
    prediction_time_ms: float


class NLQueryRequest(BaseModel):
    """Request for natural language query"""
    query: str = Field(..., description="Natural language query", min_length=1)
    debug: bool = Field(default=False, description="Include translation debug info")


class NLQueryResponse(BaseModel):
    """Response from natural language query"""
    result: Any  # Can be SimilaritySearchResponse, FactTreeResponse, or TemporalPredictResponse
    query_type: str  # similarity, fact_tree, temporal
    translated_query: Optional[Dict[str, Any]] = None  # Only if debug=True
    query_time_ms: float


# ============================================================================
# Deployment Schemas (CLI-facing)
# ============================================================================

class DeploymentResponse(BaseModel):
    """Response after deploying a model"""
    org_id: str
    model_id: str
    version: Optional[str] = None
    endpoints: Dict[str, str] = Field(default_factory=dict)
    webhook_token: Optional[str] = None


class RuntimeStatusResponse(BaseModel):
    """Runtime status response"""
    version: str
    models_loaded: int
    uptime: str
    deployment_mode: str


class ModelInfoResponse(BaseModel):
    """Information about a deployed model"""
    org_id: str
    model_id: str
    name: Optional[str] = None
    version: Optional[str] = None
    deployed_at: Optional[datetime] = None
    status: str = "Active"


class ModelsListResponse(BaseModel):
    """List of deployed models"""
    models: List[ModelInfoResponse]


class TokenInfoResponse(BaseModel):
    """Information about a token"""
    id: str
    model: Optional[str] = None
    created_at: Optional[datetime] = None
    status: str = "Active"


class TokensListResponse(BaseModel):
    """List of tokens"""
    tokens: List[TokenInfoResponse]


class LogsResponse(BaseModel):
    """Runtime logs response"""
    logs: List[str]


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: ErrorDetail


# ============================================================================
# Model Configuration Schemas
# ============================================================================

class SimilarityWeightsUpdate(BaseModel):
    """Similarity weights for edge types"""
    similarity: Optional[float] = Field(None, ge=0, le=1)
    contrast: Optional[float] = Field(None, ge=0, le=1)
    analogy: Optional[float] = Field(None, ge=0, le=1)
    composition: Optional[float] = Field(None, ge=0, le=1)
    precedes: Optional[float] = Field(None, ge=0, le=1)
    follows: Optional[float] = Field(None, ge=0, le=1)
    causes: Optional[float] = Field(None, ge=0, le=1)
    prevents: Optional[float] = Field(None, ge=0, le=1)


class ModelConfigUpdate(BaseModel):
    """Request to update model configuration (no re-encode needed)"""
    similarity_weights: Optional[SimilarityWeightsUpdate] = None
    beam_width: Optional[int] = Field(None, ge=1, le=50)
    max_tree_depth: Optional[int] = Field(None, ge=1, le=20)


class ModelConfigResponse(BaseModel):
    """Current model configuration"""
    org_id: str
    model_id: str
    similarity_weights: Dict[str, float]
    beam_width: int
    max_tree_depth: int
    resource_quotas: Dict[str, Any]
    resource_usage: Dict[str, Any]
    updated_at: datetime


class ReEncodeRequest(BaseModel):
    """Request to re-encode all glyphs"""
    regenerate_edges: bool = Field(default=True, description="Regenerate edges after re-encoding")
    background: bool = Field(default=True, description="Run as background job for large models")


class ReEncodeResponse(BaseModel):
    """Response from re-encode operation"""
    status: str  # started, completed
    job_id: Optional[str] = None  # If background=True
    glyphs_processed: Optional[int] = None
    edges_regenerated: Optional[int] = None
    duration_ms: Optional[float] = None


class ReEncodeStatusResponse(BaseModel):
    """Status of a re-encode job"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(..., ge=0, le=1)
    glyphs_total: int
    glyphs_processed: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ClearDataResponse(BaseModel):
    """Response from clear data operation"""
    org_id: str
    model_id: str
    glyphs_deleted: int
    edges_deleted: int


# ============================================================================
# Model Metadata Schemas (for Marketplace Display)
# ============================================================================

class ModelMetadata(BaseModel):
    """Model metadata for marketplace display."""
    meta_name: str = Field(..., description="Display name for the model")
    short_description: str = Field(
        default="",
        max_length=200,
        description="Brief description for marketplace cards"
    )
    long_description: str = Field(
        default="",
        description="Detailed description for model pages (markdown supported)"
    )


class ModelMetadataResponse(BaseModel):
    """API response for model metadata."""
    org_id: str
    model_id: str
    meta_name: str
    short_description: str
    long_description: str
    model_version: Optional[str] = None
    sdk_version: Optional[str] = None
