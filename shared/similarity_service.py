"""
Similarity Service for Glyphh Runtime.

Centralizes similarity calculations using the SDK's SimilarityCalculator
when available, with fallback to cosine similarity.
"""

import logging
import math
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    Centralized similarity calculation service.
    
    Uses SDK SimilarityCalculator when available for consistent similarity
    scoring with the SDK implementation. Falls back to cosine similarity
    with a logged warning when SimilarityCalculator is unavailable.
    
    Requirements:
    - 3.1: QueryService uses SimilarityCalculator for similarity search
    - 3.2: EdgeGeneratorService uses SimilarityCalculator for edge weights
    - 3.4: Fallback to cosine similarity with warning when unavailable
    """
    
    def __init__(
        self,
        similarity_calculator: Optional[Any] = None,
    ):
        """
        Initialize with optional SDK SimilarityCalculator.
        
        Args:
            similarity_calculator: SDK SimilarityCalculator instance, or None
                                   to use fallback cosine similarity
        """
        self._calculator = similarity_calculator
        self._fallback_warned = False
    
    @property
    def has_calculator(self) -> bool:
        """Check if SDK SimilarityCalculator is available."""
        return self._calculator is not None
    
    def _warn_fallback(self) -> None:
        """Log a warning about using fallback similarity (once)."""
        if not self._fallback_warned:
            logger.warning(
                "SimilarityCalculator unavailable, using fallback cosine similarity. "
                "Results may differ from SDK implementation."
            )
            self._fallback_warned = True
    
    def _cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Fallback implementation when SDK SimilarityCalculator is unavailable.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score in range [-1, 1]
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions must match: {len(embedding1)} != {len(embedding2)}"
            )
        
        if len(embedding1) == 0:
            return 0.0
        
        # Compute dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))
        
        # Handle zero vectors
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Uses SDK SimilarityCalculator if available, otherwise falls back
        to cosine similarity with a logged warning.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (typically in range [0, 1] or [-1, 1])
        """
        if self._calculator is not None:
            try:
                # Use SDK SimilarityCalculator
                return self._calculator.compute(embedding1, embedding2)
            except Exception as e:
                logger.error(f"SimilarityCalculator.compute() failed: {e}")
                self._warn_fallback()
                return self._cosine_similarity(embedding1, embedding2)
        else:
            self._warn_fallback()
            return self._cosine_similarity(embedding1, embedding2)
    
    def compute_batch_similarity(
        self,
        query_embedding: List[float],
        target_embeddings: List[List[float]],
    ) -> List[float]:
        """
        Compute similarity for multiple targets efficiently.
        
        Uses SDK SimilarityCalculator batch method if available,
        otherwise computes similarities individually.
        
        Args:
            query_embedding: Query embedding vector
            target_embeddings: List of target embedding vectors
            
        Returns:
            List of similarity scores, one per target
        """
        if not target_embeddings:
            return []
        
        if self._calculator is not None:
            try:
                # Try batch method if available
                if hasattr(self._calculator, 'compute_batch'):
                    return self._calculator.compute_batch(
                        query_embedding, target_embeddings
                    )
                # Fall back to individual computations via calculator
                return [
                    self._calculator.compute(query_embedding, target)
                    for target in target_embeddings
                ]
            except Exception as e:
                logger.error(f"SimilarityCalculator batch computation failed: {e}")
                self._warn_fallback()
        else:
            self._warn_fallback()
        
        # Fallback: compute individually using cosine similarity
        return [
            self._cosine_similarity(query_embedding, target)
            for target in target_embeddings
        ]
    
    def compute_top_k(
        self,
        query_embedding: List[float],
        target_embeddings: List[List[float]],
        k: int,
    ) -> List[tuple]:
        """
        Find top-k most similar targets.
        
        Args:
            query_embedding: Query embedding vector
            target_embeddings: List of target embedding vectors
            k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        if not target_embeddings or k <= 0:
            return []
        
        similarities = self.compute_batch_similarity(query_embedding, target_embeddings)
        
        # Create (index, score) pairs and sort by score descending
        indexed_scores = list(enumerate(similarities))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:k]


# Factory function for creating SimilarityService with SDK calculator
def create_similarity_service(
    encoder_config: Optional[Any] = None,
) -> SimilarityService:
    """
    Create a SimilarityService with SDK SimilarityCalculator if available.
    
    Args:
        encoder_config: EncoderConfig to initialize SimilarityCalculator with.
                        If None, creates service with fallback only.
    
    Returns:
        SimilarityService instance
    """
    from shared.sdk_adapter import get_sdk_adapter
    
    adapter = get_sdk_adapter()
    calculator = adapter.create_similarity_calculator()
    
    if calculator is None:
        logger.info("Creating SimilarityService with fallback cosine similarity")
    else:
        logger.info("Creating SimilarityService with SDK SimilarityCalculator")
    
    return SimilarityService(similarity_calculator=calculator)
