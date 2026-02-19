"""
Query Service for Glyphh Runtime.

Handles similarity search, fact tree generation, and temporal prediction
by coordinating between storage, model manager, and SDK components.

Updated to use SimilarityService for consistent similarity calculations.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from glyphh.fact_tree.builder import FactTree, Citation, FactNode

from domains.models.manager import ModelManager
from domains.models.storage import GlyphStorage
from domains.models.schemas import (
    Delta,
    FactTreeRequest,
    GlyphResponse,
    PredictedState,
    ScoredGlyph,
    SimilaritySearchRequest,
    TemporalPredictRequest,
)
from domains.query.fact_tree_builder import FactTreeBuilder
from shared.exceptions import (
    ModelNotFoundException,
    ValidationException,
)
from shared.similarity_service import SimilarityService

logger = logging.getLogger(__name__)


@dataclass
class Permissions:
    """User permissions for query filtering."""
    user_id: str
    org_ids: List[str]
    security_level: float = 1.0  # 0-1, higher = more access


class QueryService:
    """
    Handles query operations: similarity search, fact trees, temporal prediction.
    
    Coordinates between:
    - GlyphStorage for database operations
    - ModelManager for model access
    - SimilarityService for similarity calculations
    - SDK components for computation
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        session_factory,
        similarity_service: Optional[SimilarityService] = None,
    ):
        """
        Initialize QueryService.
        
        Args:
            model_manager: ModelManager instance
            session_factory: Async session factory for database access
            similarity_service: Optional SimilarityService for similarity calculations.
                               If not provided, will create per-model services.
        """
        self._model_manager = model_manager
        self._session_factory = session_factory
        self._similarity_service = similarity_service
    
    def _get_similarity_service(self, loaded_model: Any) -> SimilarityService:
        """
        Get SimilarityService for a loaded model.
        
        Uses injected service if available, otherwise creates one from
        the model's SimilarityCalculator.
        """
        if self._similarity_service is not None:
            return self._similarity_service
        
        # Create service from model's similarity calculator
        return SimilarityService(
            similarity_calculator=loaded_model.similarity_calculator
        )
    
    async def count_glyphs(
        self,
        org_id: str,
        model_id: str,
    ) -> int:
        """
        Count total glyphs for a model.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            
        Returns:
            Total count of glyphs in the model
        """
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            return await storage.count_glyphs(org_id, model_id)
    
    async def count_glyphs_as_fact_tree(
        self,
        org_id: str,
        model_id: str,
    ) -> FactTree:
        """
        Count glyphs and return result as FactTree.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            
        Returns:
            FactTree with count result and metadata
        """
        start_time = time.time()
        
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            count = await storage.count_glyphs(org_id, model_id)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return FactTreeBuilder.build_count(
            count=count,
            query_time_ms=query_time_ms,
        )
    
    async def list_glyphs(
        self,
        org_id: str,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[GlyphResponse]:
        """
        List glyphs for a model with pagination.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            limit: Maximum number of glyphs to return
            offset: Number of glyphs to skip
            
        Returns:
            List of GlyphResponse objects
        """
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            return await storage.list_glyphs(org_id, model_id, limit, offset)
    
    async def list_glyphs_as_fact_tree(
        self,
        org_id: str,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> FactTree:
        """
        List glyphs and return results as FactTree.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            limit: Maximum number of glyphs to return
            offset: Number of glyphs to skip
            
        Returns:
            FactTree with listed glyphs, citations, and metadata
        """
        start_time = time.time()
        
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            glyphs = await storage.list_glyphs(org_id, model_id, limit, offset)
            total_count = await storage.count_glyphs(org_id, model_id)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return FactTreeBuilder.build_list(
            glyphs=glyphs,
            query_time_ms=query_time_ms,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )
    
    async def list_glyphs_with_embeddings(
        self,
        org_id: str,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[GlyphResponse], Dict[str, List[float]]]:
        """
        List glyphs with their embeddings for a model.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            limit: Maximum number of glyphs to return
            offset: Number of glyphs to skip
            
        Returns:
            Tuple of (list of GlyphResponse, dict mapping glyph_id to embedding)
        """
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            return await storage.list_glyphs_with_embeddings(org_id, model_id, limit, offset)
    
    async def similarity_search(
        self,
        org_id: str,
        model_id: str,
        request: SimilaritySearchRequest,
        permissions: Optional[Permissions] = None,
    ) -> FactTree:
        """
        Search for similar glyphs and return results as FactTree.
        
        Uses SimilarityService for consistent similarity calculations.
        
        Returns:
            FactTree with search results, citations, and metadata
        """
        start_time = time.time()
        
        # Get loaded model
        loaded_model = await self._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        # Get similarity service for this model
        similarity_service = self._get_similarity_service(loaded_model)
        
        # Encode query text using SDK encoder with lexicon matching
        query_embedding = await self._encode_query(
            loaded_model.encoder,
            request.query,
            org_id=org_id,
            model_id=model_id,
        )
        
        # Get model config for weights
        config = await self._model_manager.get_config(org_id, model_id)
        similarity_weights = config.similarity_weights
        
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            
            # Get all glyphs for similarity computation
            glyphs, embeddings_dict = await storage.list_glyphs_with_embeddings(
                org_id=org_id,
                model_id=model_id,
                limit=1000,  # Get enough glyphs for search
            )
            
            # Convert to list of (glyph, embedding) tuples, applying filters
            raw_results = []
            for glyph in glyphs:
                glyph_id_str = str(glyph.id)
                if glyph_id_str in embeddings_dict:
                    # Apply filters if provided
                    if request.filters:
                        metadata = glyph.metadata or {}
                        matches_filter = all(
                            metadata.get(k) == v for k, v in request.filters.items()
                        )
                        if not matches_filter:
                            continue
                    raw_results.append((glyph, embeddings_dict[glyph_id_str]))
            
            # Compute similarities using SimilarityService
            scored_results = []
            for glyph_response, glyph_embedding in raw_results:
                # Compute security weight
                security_weight = self._compute_security_weight(
                    glyph_response,
                    permissions,
                )
                
                # Skip if no access
                if security_weight == 0:
                    continue
                
                # Compute similarity using SimilarityService
                base_similarity = similarity_service.compute_similarity(
                    query_embedding,
                    glyph_embedding,
                )
                
                # Apply edge-type weights (simplified - using base similarity)
                weighted_similarity = base_similarity
                
                # Combine weights multiplicatively
                final_score = weighted_similarity * security_weight
                
                # Normalize to [0, 1]
                final_score = max(0.0, min(1.0, final_score))
                
                scored_results.append(ScoredGlyph(
                    glyph=glyph_response,
                    similarity_score=base_similarity,
                    security_weight=security_weight,
                    final_score=final_score,
                ))
            
            # Sort by final score descending with stable tiebreaker (glyph ID)
            # so near-equal scores always return in the same order
            scored_results.sort(
                key=lambda x: (-x.final_score, str(x.glyph.id)),
            )
            scored_results = scored_results[:request.top_k]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        # Build FactTree instead of SimilaritySearchResponse
        return FactTreeBuilder.build_similarity_search(
            query=request.query,
            results=scored_results,
            query_time_ms=query_time_ms,
            total_count=len(scored_results),
        )
    
    async def generate_fact_tree(
        self,
        org_id: str,
        model_id: str,
        request: FactTreeRequest,
        permissions: Optional[Permissions] = None,
    ) -> FactTree:
        """Generate an explainable verification report with citations."""
        start_time = time.time()
        
        loaded_model = await self._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        config = await self._model_manager.get_config(org_id, model_id)
        max_depth = min(request.max_depth, config.max_tree_depth)
        
        # Build fact tree using SDK's FactTreeBuilder
        try:
            from glyphh import FactTree as SDKFactTree
            
            # Use SDK fact tree builder if available
            fact_tree = await self._build_fact_tree_with_sdk(
                loaded_model,
                org_id,
                model_id,
                request.claim,
                max_depth,
                request.branching_factor,
                permissions,
            )
        except ImportError:
            # Fallback to simple implementation
            fact_tree = await self._build_fact_tree_simple(
                loaded_model,
                org_id,
                model_id,
                request.claim,
                max_depth,
                request.branching_factor,
                permissions,
            )
        
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Build FactTree response using FactTreeBuilder
        return FactTreeBuilder.build_similarity_search(
            query=request.claim,
            results=[],
            query_time_ms=generation_time_ms,
            total_count=0,
        )
    
    async def predict_temporal(
        self,
        org_id: str,
        model_id: str,
        request: TemporalPredictRequest,
        permissions: Optional[Permissions] = None,
    ) -> FactTree:
        """Predict future states using beam search over temporal edges."""
        start_time = time.time()
        
        loaded_model = await self._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        config = await self._model_manager.get_config(org_id, model_id)
        beam_width = min(request.beam_width, config.beam_width)
        
        # Encode current state concepts
        state_embeddings = []
        for concept in request.current_state:
            embedding = await self._encode_query(loaded_model.encoder, concept, org_id=org_id, model_id=model_id)
            state_embeddings.append(embedding)
        
        # Use SDK's BeamSearchPredictor if available
        try:
            from glyphh import BeamSearchPredictor
            
            predictions = await self._predict_with_sdk(
                loaded_model,
                org_id,
                model_id,
                state_embeddings,
                request.steps_ahead,
                beam_width,
                request.direction,
                permissions,
            )
        except ImportError:
            # Fallback to simple implementation
            predictions = await self._predict_simple(
                loaded_model,
                org_id,
                model_id,
                state_embeddings,
                request.steps_ahead,
                beam_width,
                request.direction,
                permissions,
            )
        
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # Build FactTree response using FactTreeBuilder
        return FactTreeBuilder.build_temporal_predict(
            predictions=predictions,
            prediction_time_ms=prediction_time_ms,
            current_state=request.current_state,
            steps_ahead=request.steps_ahead,
        )
    
    async def predict_temporal_as_fact_tree(
        self,
        org_id: str,
        model_id: str,
        request: TemporalPredictRequest,
        permissions: Optional[Permissions] = None,
    ) -> FactTree:
        """
        Predict future states and return results as FactTree.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            request: Temporal prediction request with current_state, steps_ahead, etc.
            permissions: Optional user permissions for filtering
            
        Returns:
            FactTree with predictions, citations, and metadata
        """
        start_time = time.time()
        
        loaded_model = await self._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        config = await self._model_manager.get_config(org_id, model_id)
        beam_width = min(request.beam_width, config.beam_width)
        
        # Encode current state concepts
        state_embeddings = []
        for concept in request.current_state:
            embedding = await self._encode_query(loaded_model.encoder, concept, org_id=org_id, model_id=model_id)
            state_embeddings.append(embedding)
        
        # Use SDK's BeamSearchPredictor if available
        try:
            from glyphh import BeamSearchPredictor
            
            predictions = await self._predict_with_sdk(
                loaded_model,
                org_id,
                model_id,
                state_embeddings,
                request.steps_ahead,
                beam_width,
                request.direction,
                permissions,
            )
        except ImportError:
            # Fallback to simple implementation
            predictions = await self._predict_simple(
                loaded_model,
                org_id,
                model_id,
                state_embeddings,
                request.steps_ahead,
                beam_width,
                request.direction,
                permissions,
            )
        
        prediction_time_ms = (time.time() - start_time) * 1000
        
        return FactTreeBuilder.build_temporal_predict(
            predictions=predictions,
            prediction_time_ms=prediction_time_ms,
            current_state=request.current_state,
            steps_ahead=request.steps_ahead,
        )
    
    
    
    
    async def _encode_query(
        self,
        encoder: Any,
        query: str,
        org_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> List[float]:
        """
        Encode query text using the model's custom encode_query_fn.

        Every model that serves queries must have an encode_query_fn
        defined in its encoder.py. There is no generic fallback — models
        without a custom encoder will fail cleanly rather than produce
        garbage results from heuristic encoding.
        """
        try:
            loaded_model = None
            if org_id and model_id:
                loaded_model = await self._model_manager.get_model(org_id, model_id)

            if loaded_model and getattr(loaded_model, 'encode_query_fn', None):
                concept = loaded_model.encode_query_fn(query)
                glyph = encoder.encode(concept)
                return glyph.global_cortex.data.astype(float).tolist()

            # No custom encoder — fail clean
            model_label = f"{org_id}/{model_id}" if org_id else "unknown"
            raise ValidationException(
                field="query",
                reason=(
                    f"Model '{model_label}' has no encode_query_fn. "
                    f"Add an encode_query() function to the model's encoder.py."
                ),
            )

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise ValidationException(
                field="query",
                reason=f"Failed to encode: {e}"
            )
    
    def _compute_security_weight(
        self,
        glyph: GlyphResponse,
        permissions: Optional[Permissions],
    ) -> float:
        """
        Compute security weight for a glyph based on user permissions.
        
        Returns 0 if user has no access, 1 if full access.
        """
        if permissions is None:
            return 1.0
        
        # Check org access
        if glyph.org_id not in permissions.org_ids:
            return 0.0
        
        glyph_security = glyph.metadata.get("security_level", 0.0)
        if permissions.security_level < glyph_security:
            return 0.0
        
        return 1.0

    
    async def _build_fact_tree_simple(
        self,
        loaded_model: Any,
        org_id: str,
        model_id: str,
        claim: str,
        max_depth: int,
        branching_factor: int,
        permissions: Optional[Permissions],
    ) -> Dict[str, Any]:
        """Build a simple fact tree using similarity search."""
        nodes = []
        citations = []
        root_id = "root"
        
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            claim_embedding = await self._encode_query(loaded_model.encoder, claim, org_id=org_id, model_id=model_id)
            
            results = await storage.similarity_search(
                org_id=org_id,
                model_id=model_id,
                query_embedding=claim_embedding,
                top_k=branching_factor,
            )
            
            supporting_glyphs = []
            relevance_scores = []
            for glyph_response, similarity in results:
                # Apply security filter
                security_weight = self._compute_security_weight(
                    glyph_response,
                    permissions,
                )
                if security_weight > 0:
                    supporting_glyphs.append(glyph_response.id)
                    relevance_scores.append(similarity)
                    citations.append(Citation(
                        glyph_id=str(glyph_response.id),
                        component="cortex",
                        timestamp=datetime.utcnow(),
                        version="v1",
                        data_hash="",
                    ))
            
            # Compute confidence based on evidence
            if supporting_glyphs:
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                confidence = min(1.0, avg_relevance * 1.2)  # Boost slightly
            else:
                confidence = 0.1  # Low confidence if no evidence
            
            # Create root node
            nodes.append(FactNode(
                description=claim,
                value=confidence,
                children=[],
                citations=citations,
                data_context={
                    "id": root_id,
                    "supporting_glyphs": supporting_glyphs,
                },
            ))
        
        return {
            "nodes": nodes,
            "confidence": confidence,
            "citations": citations,
        }
    
    async def _build_fact_tree_with_sdk(
        self,
        loaded_model: Any,
        org_id: str,
        model_id: str,
        claim: str,
        max_depth: int,
        branching_factor: int,
        permissions: Optional[Permissions],
    ) -> Dict[str, Any]:
        """Build fact tree using SDK's FactTreeBuilder."""
        return await self._build_fact_tree_simple(
            loaded_model, org_id, model_id, claim, max_depth, branching_factor, permissions,
        )
    
    async def _predict_simple(
        self,
        loaded_model: Any,
        org_id: str,
        model_id: str,
        state_embeddings: List[List[float]],
        steps_ahead: int,
        beam_width: int,
        direction: str,
        permissions: Optional[Permissions],
    ) -> List[PredictedState]:
        """Simple temporal prediction using similarity search."""
        predictions = []
        
        async with self._session_factory() as session:
            storage = GlyphStorage(session)
            
            for step in range(steps_ahead):
                if state_embeddings:
                    import numpy as np
                    avg_embedding = np.mean(state_embeddings, axis=0).tolist()
                else:
                    continue
                
                results = await storage.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    query_embedding=avg_embedding,
                    top_k=beam_width,
                )
                
                # Get temporal edges for found glyphs
                state_glyphs = []
                deltas = []
                path_score = 0.0
                
                for glyph_response, similarity in results:
                    # Apply security filter
                    security_weight = self._compute_security_weight(
                        glyph_response,
                        permissions,
                    )
                    if security_weight > 0:
                        state_glyphs.append(glyph_response.id)
                        path_score += similarity
                        
                        edge_type = "follows" if direction == "forward" else "precedes"
                        edges = await storage.get_edges(
                            org_id=org_id,
                            model_id=model_id,
                            glyph_id=glyph_response.id,
                            edge_type=edge_type,
                            direction="outgoing",
                        )
                        
                        for edge in edges:
                            deltas.append(Delta(
                                glyph_id=edge["target_glyph_id"],
                                change_type="added",
                                magnitude=edge["weight"],
                            ))
                
                if state_glyphs:
                    confidence = path_score / len(state_glyphs)
                    predictions.append(PredictedState(
                        state_glyphs=state_glyphs,
                        confidence=min(1.0, confidence),
                        path_score=path_score,
                        deltas=deltas[:5],  # Limit deltas
                    ))
        
        return predictions
    
    async def _predict_with_sdk(
        self,
        loaded_model: Any,
        org_id: str,
        model_id: str,
        state_embeddings: List[List[float]],
        steps_ahead: int,
        beam_width: int,
        direction: str,
        permissions: Optional[Permissions],
    ) -> List[PredictedState]:
        """Temporal prediction using SDK's BeamSearchPredictor."""
        return await self._predict_simple(
            loaded_model, org_id, model_id, state_embeddings,
            steps_ahead, beam_width, direction, permissions,
        )
