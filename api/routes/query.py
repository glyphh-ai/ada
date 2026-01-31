"""
Query API Routes for Glyphh Runtime.

Endpoints for similarity search, fact tree generation, and temporal prediction.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from domains.auth.service import AuthService, User
from domains.models.storage import GlyphStorage
from domains.query.service import QueryService
from infrastructure.config import get_settings
from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/{namespace}", tags=["query"])
settings = get_settings()


# Request/Response Models
class SimilaritySearchRequest(BaseModel):
    """Similarity search request."""
    query: str = Field(..., description="Query text to search for")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class SimilaritySearchResult(BaseModel):
    """Single search result."""
    glyph_id: str
    concept_text: str
    similarity_score: float
    final_score: float
    metadata: Dict[str, Any]


class SimilaritySearchResponse(BaseModel):
    """Similarity search response."""
    results: List[SimilaritySearchResult]
    total_count: int
    query_time_ms: float


class FactTreeRequest(BaseModel):
    """Fact tree generation request."""
    claim: str = Field(..., description="Claim to verify")
    max_depth: int = Field(3, ge=1, le=10, description="Maximum tree depth")


class FactTreeResponse(BaseModel):
    """Fact tree response."""
    root_claim: str
    confidence: float
    nodes: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    generation_time_ms: float


class TemporalPredictRequest(BaseModel):
    """Temporal prediction request."""
    current_state: List[str] = Field(..., description="Current state concepts")
    steps_ahead: int = Field(1, ge=1, le=10, description="Steps to predict")
    beam_width: int = Field(5, ge=1, le=20, description="Beam width")


class PredictionResult(BaseModel):
    """Single prediction result."""
    state: List[str]
    confidence: float
    path_score: float


class TemporalPredictResponse(BaseModel):
    """Temporal prediction response."""
    predictions: List[PredictionResult]
    prediction_time_ms: float


# Dependency injection
async def get_query_service(
    namespace: str,
    db: AsyncSession = Depends(get_db),
) -> QueryService:
    """Get query service for namespace."""
    from main import model_manager
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    storage = GlyphStorage(db)
    return QueryService(storage, model_manager)


async def get_current_user() -> Optional[User]:
    """Get current user (placeholder)."""
    if settings.deployment_mode == "local":
        return None
    return None


# Endpoints
@router.post("/search", response_model=SimilaritySearchResponse)
async def similarity_search(
    namespace: str,
    request: SimilaritySearchRequest,
    query_service: QueryService = Depends(get_query_service),
    user: Optional[User] = Depends(get_current_user),
) -> SimilaritySearchResponse:
    """
    Search for similar glyphs.
    
    Encodes the query text and finds the most similar glyphs
    using weighted similarity scoring.
    """
    import time
    start = time.time()
    
    try:
        results = await query_service.similarity_search(
            namespace=namespace,
            query=request.query,
            top_k=request.top_k,
            user_permissions=user,
            filters=request.filters,
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
        return SimilaritySearchResponse(
            results=[
                SimilaritySearchResult(
                    glyph_id=str(r.glyph.id),
                    concept_text=r.glyph.concept_text,
                    similarity_score=r.similarity_score,
                    final_score=r.final_score,
                    metadata=r.glyph.metadata or {},
                )
                for r in results
            ],
            total_count=len(results),
            query_time_ms=elapsed_ms,
        )
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fact-tree", response_model=FactTreeResponse)
async def generate_fact_tree(
    namespace: str,
    request: FactTreeRequest,
    query_service: QueryService = Depends(get_query_service),
    user: Optional[User] = Depends(get_current_user),
) -> FactTreeResponse:
    """
    Generate a fact tree for claim verification.
    
    Builds a hierarchical verification report with citations
    to source glyphs.
    """
    import time
    start = time.time()
    
    try:
        fact_tree = await query_service.generate_fact_tree(
            namespace=namespace,
            claim=request.claim,
            max_depth=request.max_depth,
            user_permissions=user,
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
        # Convert fact tree to response format
        if hasattr(fact_tree, 'to_dict'):
            tree_dict = fact_tree.to_dict()
        else:
            tree_dict = fact_tree if isinstance(fact_tree, dict) else {}
        
        return FactTreeResponse(
            root_claim=request.claim,
            confidence=tree_dict.get("confidence", 0.0),
            nodes=tree_dict.get("nodes", []),
            citations=tree_dict.get("citations", []),
            generation_time_ms=elapsed_ms,
        )
    except Exception as e:
        logger.error(f"Fact tree generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=TemporalPredictResponse)
async def temporal_predict(
    namespace: str,
    request: TemporalPredictRequest,
    query_service: QueryService = Depends(get_query_service),
    user: Optional[User] = Depends(get_current_user),
) -> TemporalPredictResponse:
    """
    Predict future states using temporal edges.
    
    Uses beam search to find the most likely future states
    based on temporal patterns in the data.
    """
    import time
    start = time.time()
    
    try:
        predictions = await query_service.predict_temporal(
            namespace=namespace,
            current_state=request.current_state,
            steps_ahead=request.steps_ahead,
            beam_width=request.beam_width,
            user_permissions=user,
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
        return TemporalPredictResponse(
            predictions=[
                PredictionResult(
                    state=p.state if hasattr(p, 'state') else [],
                    confidence=p.confidence if hasattr(p, 'confidence') else 0.0,
                    path_score=p.path_score if hasattr(p, 'path_score') else 0.0,
                )
                for p in predictions
            ],
            prediction_time_ms=elapsed_ms,
        )
    except Exception as e:
        logger.error(f"Temporal prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# NL Query endpoint (if enabled)
@router.post("/query")
async def nl_query(
    namespace: str,
    query: str = Query(..., description="Natural language query"),
    query_service: QueryService = Depends(get_query_service),
    user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Execute a natural language query.
    
    Translates the query to a structured query using rules-based
    matching with optional LLM fallback.
    """
    if not settings.enable_nl_query:
        raise HTTPException(
            status_code=501,
            detail="Natural language query interface is not enabled"
        )
    
    # TODO: Implement NL query service integration
    raise HTTPException(
        status_code=501,
        detail="NL query not yet implemented"
    )


@router.get("/intents")
async def list_intents(namespace: str) -> Dict[str, Any]:
    """
    List available intent patterns for NL queries.
    
    Returns the intent patterns loaded from the model.
    """
    if not settings.enable_nl_query:
        raise HTTPException(
            status_code=501,
            detail="Natural language query interface is not enabled"
        )
    
    # TODO: Return intent patterns from model
    return {"intents": []}
