"""
Query API Routes for Glyphh Runtime.

Endpoints for similarity search, fact tree generation, and temporal prediction.
All routes scoped by /{org_id}/{model_id}/...
"""

import logging
import time
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
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["query"])
settings = get_settings()


# Request/Response Models
class SimilaritySearchRequest(BaseModel):
    query: str = Field(..., description="Query text to search for")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class SimilaritySearchResult(BaseModel):
    glyph_id: str
    concept_text: str
    similarity_score: float
    final_score: float
    metadata: Dict[str, Any]


class SimilaritySearchResponse(BaseModel):
    results: List[SimilaritySearchResult]
    total_count: int
    query_time_ms: float


class FactTreeRequest(BaseModel):
    claim: str = Field(..., description="Claim to verify")
    max_depth: int = Field(3, ge=1, le=10, description="Maximum tree depth")


class FactTreeResponse(BaseModel):
    root_claim: str
    confidence: float
    nodes: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    generation_time_ms: float


class TemporalPredictRequest(BaseModel):
    current_state: List[str] = Field(..., description="Current state concepts")
    steps_ahead: int = Field(1, ge=1, le=10, description="Steps to predict")
    beam_width: int = Field(5, ge=1, le=20, description="Beam width")


class PredictionResult(BaseModel):
    state: List[str]
    confidence: float
    path_score: float


class TemporalPredictResponse(BaseModel):
    predictions: List[PredictionResult]
    prediction_time_ms: float


# Dependency injection
async def get_query_service(
    org_id: str,
    model_id: str,
    db: AsyncSession = Depends(get_db),
) -> QueryService:
    from main import model_manager
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    storage = GlyphStorage(db)
    return QueryService(storage, model_manager)


async def get_current_user() -> Optional[User]:
    if settings.deployment_mode == "local":
        return None
    return None


# Endpoints
@router.post("/search", response_model=SimilaritySearchResponse)
async def similarity_search(
    org_id: str,
    model_id: str,
    request: SimilaritySearchRequest,
    query_service: QueryService = Depends(get_query_service),
    user: Optional[User] = Depends(get_current_user),
) -> SimilaritySearchResponse:
    """Search for similar glyphs."""
    start = time.time()
    
    try:
        results = await query_service.similarity_search(
            org_id=org_id,
            model_id=model_id,
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
    org_id: str,
    model_id: str,
    request: FactTreeRequest,
    query_service: QueryService = Depends(get_query_service),
    user: Optional[User] = Depends(get_current_user),
) -> FactTreeResponse:
    """Generate a fact tree for claim verification."""
    start = time.time()
    
    try:
        fact_tree = await query_service.generate_fact_tree(
            org_id=org_id,
            model_id=model_id,
            claim=request.claim,
            max_depth=request.max_depth,
            user_permissions=user,
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
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
    org_id: str,
    model_id: str,
    request: TemporalPredictRequest,
    query_service: QueryService = Depends(get_query_service),
    user: Optional[User] = Depends(get_current_user),
) -> TemporalPredictResponse:
    """Predict future states using temporal edges."""
    start = time.time()
    
    try:
        predictions = await query_service.predict_temporal(
            org_id=org_id,
            model_id=model_id,
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
