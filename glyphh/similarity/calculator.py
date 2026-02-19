"""
Weighted Similarity Calculator for the Glyphh SDK.

This module implements the SimilarityCalculator class that computes weighted similarity
between glyphs with dual weighting:
- Similarity weights (importance): Control how much each hierarchy level contributes
- Security weights (visibility): Control access based on user clearance

The calculator supports two similarity metrics:
- Cosine similarity (default): Range [-1, 1], preserves direction
- Hamming similarity: Range [0, 1], intuitive percentage agreement

Key Features:
- Dual weighting system (similarity + security)
- Multiple similarity metrics (cosine, hamming)
- Metric conversion functions
- Edge-based similarity computation
- Support for all 8 edge types (4 spatial + 4 temporal)

Example:
    >>> from glyphh.similarity import SimilarityCalculator
    >>> from glyphh.core.types import Glyph
    >>> 
    >>> calculator = SimilarityCalculator(threshold=0.5)
    >>> result = calculator.compute_similarity(
    ...     glyph1,
    ...     glyph2,
    ...     edge_type="neural_cortex",
    ...     user_clearance=1.0,
    ...     metric="cosine"
    ... )
    >>> print(f"Score: {result.score}, Visible: {result.visible}")
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np
import hashlib

from glyphh.core.types import Glyph, Edge, Vector
from glyphh.core.ops import cosine_similarity, hamming_similarity
from glyphh.fact_tree import FactTree, Citation


@dataclass
class SimilarityResult:
    """
    Result of similarity computation.
    
    Contains the computed similarity score, visibility decision, and metadata
    about the computation.
    
    Attributes:
        glyph1: First glyph in comparison
        glyph2: Second glyph in comparison
        edge_type: Type of edge used for comparison
        score: Final weighted similarity score
        visible: Whether result is visible to user (based on clearance)
        raw_score: Raw similarity before weighting
        similarity_weight: Applied similarity weight
        security_weight: Applied security weight
        metric: Similarity metric used ("cosine" or "hamming")
        fact_tree: Comprehensive explanation of the computation
        timestamp: When computation was performed
        metadata: Additional metadata about the computation
    """
    glyph1: Glyph
    glyph2: Glyph
    edge_type: str
    score: float
    visible: bool
    raw_score: float
    similarity_weight: float
    security_weight: float
    metric: str
    fact_tree: FactTree
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimilarityCalculator:
    """
    Weighted similarity calculator with dual weighting system.
    
    The calculator computes similarity between glyphs using:
    1. Raw similarity computation (cosine or hamming)
    2. Similarity weighting (importance)
    3. Security weighting (access control)
    4. Visibility decision (threshold + clearance check)
    
    Similarity Metrics:
    - Cosine (default): Normalized dot product, range [-1, 1]
      - 1.0: Identical vectors
      - 0.0: Orthogonal (50% agreement)
      - -1.0: Opposite vectors
    
    - Hamming: Proportion of agreeing dimensions, range [0, 1]
      - 1.0: Identical vectors (100% agreement)
      - 0.5: Orthogonal (50% agreement)
      - 0.0: Opposite vectors (0% agreement)
    
    Relationship: hamming = (cosine + 1) / 2
    
    Attributes:
        threshold: Minimum similarity score for visibility (default: 0.5)
        default_metric: Default similarity metric ("cosine" or "hamming")
    
    Example:
        >>> calculator = SimilarityCalculator(threshold=0.7)
        >>> result = calculator.compute_similarity(
        ...     glyph1,
        ...     glyph2,
        ...     edge_type="neural_cortex",
        ...     user_clearance=1.0,
        ...     metric="cosine"
        ... )
        >>> 
        >>> # Convert between metrics
        >>> hamming_score = calculator.convert_cosine_to_hamming(result.raw_score)
        >>> cosine_score = calculator.convert_hamming_to_cosine(hamming_score)
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        default_metric: str = "cosine"
    ):
        """
        Initialize the similarity calculator.
        
        Args:
            threshold: Minimum similarity score for visibility (default: 0.5)
            default_metric: Default similarity metric ("cosine" or "hamming")
        
        Raises:
            ValueError: If threshold is not in [0, 1] or metric is invalid
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        if default_metric not in ["cosine", "hamming"]:
            raise ValueError(
                f"Invalid metric: {default_metric}. Must be 'cosine' or 'hamming'"
            )
        
        self.threshold = threshold
        self.default_metric = default_metric
    
    def compute_similarity(
        self,
        glyph1: Glyph,
        glyph2: Glyph,
        edge_type: str,
        user_clearance: float = 1.0,
        metric: Optional[str] = None
    ) -> SimilarityResult:
        """
        Compute weighted similarity between two glyphs.
        
        This method performs the complete similarity computation:
        1. Extract relevant vectors based on edge type
        2. Compute raw similarity (cosine or hamming)
        3. Apply similarity weights (importance)
        4. Apply security weights (access control)
        5. Make visibility decision (threshold + clearance)
        6. Build comprehensive fact tree explaining the computation
        
        Args:
            glyph1: First glyph
            glyph2: Second glyph
            edge_type: Type of edge for comparison (e.g., "neural_cortex")
            user_clearance: User's security clearance level (0.0-1.0)
            metric: Similarity metric ("cosine" or "hamming"), uses default if None
        
        Returns:
            SimilarityResult with score, visibility, fact tree, and metadata
        
        Raises:
            ValueError: If glyphs are from different vector spaces
            ValueError: If edge_type is invalid
            ValueError: If metric is invalid
            ValueError: If user_clearance is not in [0, 1]
        
        Example:
            >>> calculator = SimilarityCalculator(threshold=0.5)
            >>> result = calculator.compute_similarity(
            ...     glyph1,
            ...     glyph2,
            ...     edge_type="neural_cortex",
            ...     user_clearance=1.0,
            ...     metric="cosine"
            ... )
            >>> print(f"Score: {result.score:.3f}")
            >>> print(f"Visible: {result.visible}")
            >>> print(result.fact_tree.to_text())
        """
        import time
        start_time = time.time()
        
        # Validate inputs
        if glyph1.space_id != glyph2.space_id:
            raise ValueError(
                f"Cannot compare glyphs from different vector spaces: "
                f"{glyph1.space_id} vs {glyph2.space_id}"
            )
        
        if not 0.0 <= user_clearance <= 1.0:
            raise ValueError(
                f"User clearance must be in [0, 1], got {user_clearance}"
            )
        
        # Use default metric if not specified
        if metric is None:
            metric = self.default_metric
        
        if metric not in ["cosine", "hamming"]:
            raise ValueError(
                f"Invalid metric: {metric}. Must be 'cosine' or 'hamming'"
            )
        
        # Get relevant vectors based on edge type
        vector1, weights1, security_level1 = self._extract_edge_data(glyph1, edge_type)
        vector2, weights2, security_level2 = self._extract_edge_data(glyph2, edge_type)
        
        # Compute raw similarity
        if metric == "cosine":
            raw_score = self._cosine_similarity(vector1, vector2)
        elif metric == "hamming":
            raw_score = self._hamming_similarity(vector1, vector2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Apply similarity weights
        similarity_weight = self._compute_similarity_weight(weights1, weights2)
        weighted_score = raw_score * similarity_weight
        
        # Apply security weights
        security_weight = self._compute_security_weight(
            security_level1,
            security_level2,
            user_clearance
        )
        final_score = weighted_score * security_weight
        
        # Determine visibility
        required_clearance = max(security_level1, security_level2)
        visible = (final_score >= self.threshold) and (user_clearance >= required_clearance)
        
        # Compute elapsed time
        computation_time = time.time() - start_time
        
        # Build fact tree (includes computation time in data context)
        fact_tree = self._build_fact_tree(
            glyph1, glyph2, edge_type, user_clearance, metric, computation_time
        )
        
        # Create result
        return SimilarityResult(
            glyph1=glyph1,
            glyph2=glyph2,
            edge_type=edge_type,
            score=final_score,
            visible=visible,
            raw_score=raw_score,
            similarity_weight=similarity_weight,
            security_weight=security_weight,
            metric=metric,
            fact_tree=fact_tree,
            timestamp=datetime.now(),
            metadata={
                "user_clearance": user_clearance,
                "required_clearance": required_clearance,
                "threshold": self.threshold,
                "dimensions": len(vector1.data),
                "computation_time": computation_time
            }
        )
    
    def _extract_edge_data(
        self,
        glyph: Glyph,
        edge_type: str
    ) -> tuple[Vector, Dict[str, float], float]:
        """
        Extract vector, weights, and security level for a given edge type.
        
        Args:
            glyph: Glyph to extract data from
            edge_type: Type of edge (e.g., "neural_cortex", "neural_layer")
        
        Returns:
            Tuple of (vector, weights, security_level)
        
        Raises:
            ValueError: If edge_type is invalid or data is missing
        """
        # Validate edge type
        valid_types = [
            "neural_cortex", "neural_layer", "neural_segment", "neural_role",
            "temporal_cortex", "temporal_layer", "temporal_segment", "temporal_role"
        ]
        if edge_type not in valid_types:
            raise ValueError(
                f"Invalid edge type: {edge_type}. Must be one of: {valid_types}"
            )
        
        # Extract data based on edge type
        if edge_type == "neural_cortex" or edge_type == "temporal_cortex":
            # Global cortex level
            vector = glyph.global_cortex
            weights = {"cortex": 1.0}
            security_level = glyph.security_levels.get("cortex", 0.0)
        
        elif edge_type == "neural_layer" or edge_type == "temporal_layer":
            # Layer level - use first layer for now
            # TODO: Support specifying which layer
            if not glyph.layers:
                raise ValueError(f"Glyph {glyph.identifier} has no layers")
            
            first_layer = next(iter(glyph.layers.values()))
            vector = first_layer.cortex
            weights = first_layer.weights.copy()
            weights.setdefault("layer", 1.0)
            security_level = glyph.security_levels.get("layer", 0.0)
        
        elif edge_type == "neural_segment" or edge_type == "temporal_segment":
            # Segment level - use first segment of first layer for now
            # TODO: Support specifying which layer and segment
            if not glyph.layers:
                raise ValueError(f"Glyph {glyph.identifier} has no layers")
            
            first_layer = next(iter(glyph.layers.values()))
            if not first_layer.segments:
                raise ValueError(
                    f"Layer {first_layer.name} in glyph {glyph.identifier} has no segments"
                )
            
            first_segment = next(iter(first_layer.segments.values()))
            vector = first_segment.cortex
            weights = first_segment.weights.copy()
            weights.setdefault("segment", 1.0)
            security_level = glyph.security_levels.get("segment", 0.0)
        
        elif edge_type == "neural_role" or edge_type == "temporal_role":
            # Role level - use first role of first segment of first layer for now
            # TODO: Support specifying which layer, segment, and role
            if not glyph.layers:
                raise ValueError(f"Glyph {glyph.identifier} has no layers")
            
            first_layer = next(iter(glyph.layers.values()))
            if not first_layer.segments:
                raise ValueError(
                    f"Layer {first_layer.name} in glyph {glyph.identifier} has no segments"
                )
            
            first_segment = next(iter(first_layer.segments.values()))
            if not first_segment.roles:
                raise ValueError(
                    f"Segment {first_segment.name} in glyph {glyph.identifier} has no roles"
                )
            
            first_role_name = next(iter(first_segment.roles.keys()))
            vector = first_segment.roles[first_role_name]
            weights = {"role": first_segment.weights.get(first_role_name, 1.0)}
            security_level = glyph.security_levels.get("role", 0.0)
        
        else:
            raise ValueError(f"Unsupported edge type: {edge_type}")
        
        return vector, weights, security_level
    
    def _cosine_similarity(self, v1: Vector, v2: Vector) -> float:
        """
        Compute cosine similarity between two vectors.
        
        For bipolar vectors {-1, +1}:
        - Range: [-1, 1]
        - 1.0: Identical vectors
        - 0.0: Orthogonal (50% agreement)
        - -1.0: Opposite vectors (0% agreement)
        
        Args:
            v1: First vector
            v2: Second vector
        
        Returns:
            Cosine similarity score
        
        Raises:
            ValueError: If vectors have different dimensions
        """
        if v1.dimension != v2.dimension:
            raise ValueError(
                f"Dimension mismatch: v1 has {v1.dimension} dimensions, "
                f"v2 has {v2.dimension} dimensions"
            )
        
        return cosine_similarity(v1.data, v2.data)
    
    def _hamming_similarity(self, v1: Vector, v2: Vector) -> float:
        """
        Compute hamming similarity between two vectors.
        
        For bipolar vectors {-1, +1}:
        - Range: [0, 1]
        - 1.0: Identical vectors (100% agreement)
        - 0.5: Orthogonal (50% agreement)
        - 0.0: Opposite vectors (0% agreement)
        
        Relationship: hamming = (cosine + 1) / 2
        
        Args:
            v1: First vector
            v2: Second vector
        
        Returns:
            Hamming similarity score
        
        Raises:
            ValueError: If vectors have different dimensions
        """
        if v1.dimension != v2.dimension:
            raise ValueError(
                f"Dimension mismatch: v1 has {v1.dimension} dimensions, "
                f"v2 has {v2.dimension} dimensions"
            )
        
        return hamming_similarity(v1.data, v2.data)
    
    def _compute_similarity_weight(
        self,
        weights1: Dict[str, float],
        weights2: Dict[str, float]
    ) -> float:
        """
        Compute combined similarity weight from two weight dictionaries.
        
        The similarity weight represents the importance of this comparison.
        It's computed by averaging the weights from both glyphs and multiplying
        across hierarchy levels.
        
        Args:
            weights1: Weights from first glyph
            weights2: Weights from second glyph
        
        Returns:
            Combined similarity weight (0.0-1.0)
        """
        # Get all unique weight keys
        all_keys = set(weights1.keys()) | set(weights2.keys())
        
        # Average weights from both glyphs
        combined_weight = 1.0
        for key in all_keys:
            w1 = weights1.get(key, 1.0)
            w2 = weights2.get(key, 1.0)
            avg_weight = (w1 + w2) / 2.0
            combined_weight *= avg_weight
        
        return combined_weight
    
    def _compute_security_weight(
        self,
        security_level1: float,
        security_level2: float,
        user_clearance: float
    ) -> float:
        """
        Compute security weight based on user clearance.
        
        The security weight controls visibility based on access control.
        If user clearance is below required level, the weight is reduced.
        
        Args:
            security_level1: Required clearance for first glyph
            security_level2: Required clearance for second glyph
            user_clearance: User's clearance level (0.0-1.0)
        
        Returns:
            Security weight (0.0-1.0)
        """
        # Required clearance is the maximum of both glyphs
        required_clearance = max(security_level1, security_level2)
        
        # If no security requirement, full access
        if required_clearance == 0.0:
            return 1.0
        
        # Compute security weight
        if user_clearance >= required_clearance:
            # Full access
            return 1.0
        else:
            # Partial access based on clearance ratio
            return user_clearance / required_clearance
    
    def convert_cosine_to_hamming(self, cosine_score: float) -> float:
        """
        Convert cosine similarity [-1, 1] to hamming similarity [0, 1].
        
        Relationship: hamming = (cosine + 1) / 2
        
        Args:
            cosine_score: Cosine similarity score
        
        Returns:
            Hamming similarity score
        
        Example:
            >>> calculator = SimilarityCalculator()
            >>> hamming = calculator.convert_cosine_to_hamming(0.4)
            >>> print(hamming)
            0.7
        """
        return (cosine_score + 1.0) / 2.0
    
    def convert_hamming_to_cosine(self, hamming_score: float) -> float:
        """
        Convert hamming similarity [0, 1] to cosine similarity [-1, 1].
        
        Relationship: cosine = 2 * hamming - 1
        
        Args:
            hamming_score: Hamming similarity score
        
        Returns:
            Cosine similarity score
        
        Example:
            >>> calculator = SimilarityCalculator()
            >>> cosine = calculator.convert_hamming_to_cosine(0.7)
            >>> print(cosine)
            0.4
        """
        return 2.0 * hamming_score - 1.0
    
    def _build_fact_tree(
        self,
        glyph1: Glyph,
        glyph2: Glyph,
        edge_type: str,
        user_clearance: float,
        metric: str,
        computation_time: float
    ) -> FactTree:
        """
        Build comprehensive fact tree explaining the similarity computation.
        
        The fact tree includes:
        - Full glyph structures with all data
        - Edge analysis
        - Raw similarity computation with mathematical explanation
        - Similarity weighting
        - Security weighting with data context
        - Final score calculation
        - Visibility decision
        
        Args:
            glyph1: First glyph
            glyph2: Second glyph
            edge_type: Type of edge for comparison
            user_clearance: User's security clearance
            metric: Similarity metric used
            computation_time: Time taken for computation (seconds)
        
        Returns:
            FactTree with complete computation explanation
        """
        fact_tree = FactTree()
        
        # Add glyph structures
        fact_tree.add_fact(
            path=["glyphs"],
            description="Glyphs Being Compared",
            value=None
        )
        
        # Deconstruct glyph 1
        glyph1_data = self._deconstruct_glyph(glyph1)
        fact_tree.add_fact(
            path=["glyphs", "glyph1"],
            description=f"Glyph 1: {glyph1.identifier}",
            value=glyph1.identifier,
            data_sample=glyph1_data,
            citations=[Citation(
                glyph_id=glyph1.identifier,
                component="full_structure",
                timestamp=glyph1.timestamp,
                version=glyph1.version,
                data_hash=self._hash_glyph(glyph1)
            )]
        )
        
        # Deconstruct glyph 2
        glyph2_data = self._deconstruct_glyph(glyph2)
        fact_tree.add_fact(
            path=["glyphs", "glyph2"],
            description=f"Glyph 2: {glyph2.identifier}",
            value=glyph2.identifier,
            data_sample=glyph2_data,
            citations=[Citation(
                glyph_id=glyph2.identifier,
                component="full_structure",
                timestamp=glyph2.timestamp,
                version=glyph2.version,
                data_hash=self._hash_glyph(glyph2)
            )]
        )
        
        # Get edge data
        vector1, weights1, security_level1 = self._extract_edge_data(glyph1, edge_type)
        vector2, weights2, security_level2 = self._extract_edge_data(glyph2, edge_type)
        
        # Add edge analysis
        fact_tree.add_fact(
            path=["edge_analysis"],
            description=f"Edge Analysis ({edge_type})",
            value=edge_type,
            data_sample={
                "edge_type": edge_type,
                "glyph1_edge": {
                    "source": glyph1.identifier,
                    "vector_dimensions": len(vector1.data),
                    "weights": weights1,
                    "security_level": security_level1
                },
                "glyph2_edge": {
                    "source": glyph2.identifier,
                    "vector_dimensions": len(vector2.data),
                    "weights": weights2,
                    "security_level": security_level2
                }
            }
        )
        
        # Compute raw similarity with detailed data
        if metric == "cosine":
            raw_score = self._cosine_similarity(vector1, vector2)
            formula = "cos(v1, v2) = (v1 · v2) / (||v1|| * ||v2||)"
        else:
            raw_score = self._hamming_similarity(vector1, vector2)
            formula = "hamming(v1, v2) = (dimensions_agree / total_dimensions)"
        
        dot_product = np.dot(vector1.data, vector2.data)
        dimensions_agree = np.sum(vector1.data == vector2.data)
        
        fact_tree.add_fact(
            path=["raw_similarity"],
            description="Raw Similarity",
            value=raw_score,
            math_explanation=formula,
            citations=[
                Citation(
                    glyph_id=glyph1.identifier,
                    component=edge_type,
                    timestamp=glyph1.timestamp,
                    version=glyph1.version,
                    data_hash=self._hash_vector(vector1)
                ),
                Citation(
                    glyph_id=glyph2.identifier,
                    component=edge_type,
                    timestamp=glyph2.timestamp,
                    version=glyph2.version,
                    data_hash=self._hash_vector(vector2)
                )
            ],
            data_sample={
                "metric": metric,
                "dot_product": int(dot_product),
                "norm_product": float(np.linalg.norm(vector1.data) * np.linalg.norm(vector2.data)),
                "dimensions_compared": len(vector1.data),
                "dimensions_agreeing": int(dimensions_agree),
                "dimensions_disagreeing": len(vector1.data) - int(dimensions_agree),
                "agreement_percentage": float(dimensions_agree / len(vector1.data) * 100),
                "vector1_sample": vector1.data[:10].tolist(),
                "vector2_sample": vector2.data[:10].tolist()
            }
        )
        
        # Compute similarity weights
        similarity_weight = self._compute_similarity_weight(weights1, weights2)
        fact_tree.add_fact(
            path=["similarity_weights"],
            description="Similarity Weights",
            value=similarity_weight,
            math_explanation="w_total = w_cortex * w_layer * w_segment * w_role",
            citations=[
                Citation(
                    glyph_id=glyph1.identifier,
                    component="weights",
                    timestamp=glyph1.timestamp,
                    version=glyph1.version,
                    data_hash=self._hash_dict(weights1)
                )
            ],
            data_sample={
                "glyph1_weights": weights1,
                "glyph2_weights": weights2,
                "combined_weight": similarity_weight
            }
        )
        
        # Compute security weights
        security_weight = self._compute_security_weight(
            security_level1,
            security_level2,
            user_clearance
        )
        required_clearance = max(security_level1, security_level2)
        
        # Compute data context (dimensions, filtering)
        data_context = self._compute_data_context(
            glyph1, glyph2, vector1, vector2, user_clearance, edge_type
        )
        
        fact_tree.add_fact(
            path=["security_weights"],
            description="Security Weights",
            value=security_weight,
            math_explanation="w_security = min(user_clearance / required_clearance, 1.0)",
            citations=[
                Citation(
                    glyph_id=glyph1.identifier,
                    component="security_levels",
                    timestamp=glyph1.timestamp,
                    version=glyph1.version,
                    data_hash=self._hash_dict(glyph1.security_levels)
                )
            ],
            data_sample={
                "user_clearance": user_clearance,
                "required_clearance": required_clearance,
                "security_multiplier": security_weight,
                "glyph1_security_level": security_level1,
                "glyph2_security_level": security_level2
            },
            data_context=data_context
        )
        
        # Compute final score
        final_score = raw_score * similarity_weight * security_weight
        
        # Get data context for final score
        data_context_final = self._compute_data_context(
            glyph1, glyph2, vector1, vector2, user_clearance, edge_type
        )
        data_context_final["computation_time"] = computation_time
        
        fact_tree.add_fact(
            path=["final_score"],
            description="Final Score",
            value=final_score,
            math_explanation="final = raw * w_similarity * w_security",
            data_sample={
                "raw_score": raw_score,
                "similarity_weight": similarity_weight,
                "security_weight": security_weight,
                "final_score": final_score,
                "calculation": f"{raw_score:.4f} * {similarity_weight:.4f} * {security_weight:.4f} = {final_score:.4f}"
            },
            data_context=data_context_final
        )
        
        # Visibility decision
        visible = (final_score >= self.threshold) and (user_clearance >= required_clearance)
        fact_tree.add_fact(
            path=["visibility"],
            description="Visibility Decision",
            value=visible,
            data_sample={
                "threshold_check": f"{final_score:.4f} >= {self.threshold:.4f}",
                "threshold_passed": final_score >= self.threshold,
                "clearance_check": f"{user_clearance:.4f} >= {required_clearance:.4f}",
                "clearance_passed": user_clearance >= required_clearance,
                "result": "Visible" if visible else "Hidden"
            }
        )
        
        return fact_tree
    
    def _compute_data_context(
        self,
        glyph1: Glyph,
        glyph2: Glyph,
        vector1: Vector,
        vector2: Vector,
        user_clearance: float,
        edge_type: str
    ) -> Dict[str, Any]:
        """
        Compute data context including dimensions, filtering, and volume metrics.
        
        Provides transparency about:
        - Total dimensions in the vectors
        - Visible dimensions after security filtering
        - Filtered dimensions (hidden due to clearance)
        - Filtering percentage
        - Filtering breakdown by hierarchy level
        - Computation time estimate
        - Data volume metrics
        
        Args:
            glyph1: First glyph
            glyph2: Second glyph
            vector1: First vector being compared
            vector2: Second vector being compared
            user_clearance: User's security clearance
            edge_type: Type of edge being compared
        
        Returns:
            Dictionary with data context information
        """
        total_dimensions = len(vector1.data)
        
        # Compute filtering breakdown by hierarchy level
        filtering_breakdown = self._compute_filtering_breakdown(
            glyph1, user_clearance, edge_type
        )
        
        # Calculate total filtered dimensions
        total_filtered = sum(filtering_breakdown.values())
        visible_dimensions = total_dimensions - total_filtered
        filtering_percentage = (total_filtered / total_dimensions * 100) if total_dimensions > 0 else 0.0
        
        # Compute data volume metrics
        total_data_points = total_dimensions * 2  # Two vectors
        effective_data_points = visible_dimensions * 2
        
        return {
            "total_dimensions": total_dimensions,
            "visible_dimensions": visible_dimensions,
            "filtered_dimensions": total_filtered,
            "filtering_percentage": filtering_percentage,
            "filtering_breakdown": filtering_breakdown,
            "comparison_type": edge_type,
            "glyphs_compared": 2,
            "total_data_points": total_data_points,
            "effective_data_points": effective_data_points
        }
    
    def _compute_filtering_breakdown(
        self,
        glyph: Glyph,
        user_clearance: float,
        edge_type: str
    ) -> Dict[str, int]:
        """
        Compute filtering breakdown by hierarchy level.
        
        Estimates how many dimensions are filtered at each level:
        - Cortex level
        - Layer level
        - Segment level
        - Role level
        
        This is a simplified estimation based on security levels.
        In a real implementation, this would track actual dimension filtering.
        
        Args:
            glyph: Glyph being analyzed
            user_clearance: User's security clearance
            edge_type: Type of edge being compared
        
        Returns:
            Dictionary mapping hierarchy level to filtered dimension count
        """
        breakdown = {
            "cortex_level": 0,
            "layer_level": 0,
            "segment_level": 0,
            "role_level": 0
        }
        
        # Get total dimensions
        total_dims = glyph.global_cortex.dimension
        
        # Estimate filtering at each level based on security clearance
        # This is a simplified model - real implementation would track actual filtering
        
        # Cortex level filtering
        cortex_security = glyph.security_levels.get("cortex", 0.0)
        if user_clearance < cortex_security:
            clearance_ratio = user_clearance / cortex_security if cortex_security > 0 else 1.0
            breakdown["cortex_level"] = int(total_dims * (1 - clearance_ratio) * 0.1)
        
        # Layer level filtering
        layer_security = glyph.security_levels.get("layer", 0.0)
        if user_clearance < layer_security:
            clearance_ratio = user_clearance / layer_security if layer_security > 0 else 1.0
            breakdown["layer_level"] = int(total_dims * (1 - clearance_ratio) * 0.2)
        
        # Segment level filtering
        segment_security = glyph.security_levels.get("segment", 0.0)
        if user_clearance < segment_security:
            clearance_ratio = user_clearance / segment_security if segment_security > 0 else 1.0
            breakdown["segment_level"] = int(total_dims * (1 - clearance_ratio) * 0.3)
        
        # Role level filtering
        role_security = glyph.security_levels.get("role", 0.0)
        if user_clearance < role_security:
            clearance_ratio = user_clearance / role_security if role_security > 0 else 1.0
            breakdown["role_level"] = int(total_dims * (1 - clearance_ratio) * 0.4)
        
        return breakdown
    
    def _deconstruct_glyph(self, glyph: Glyph) -> Dict[str, Any]:
        """
        Deconstruct glyph into full hierarchical structure for fact tree.
        
        Extracts all data from the glyph including:
        - Basic metadata (name, space_id, timestamp, version)
        - Global cortex with vector sample
        - All layers with their cortices
        - All segments with their cortices
        - All roles with their vectors and values
        - Security levels
        
        Args:
            glyph: Glyph to deconstruct
        
        Returns:
            Dictionary with complete glyph structure
        
        Note:
            This function is optimized to minimize expensive operations like
            norm computation and hashing. Vector samples are limited to 10 elements.
        """
        # Helper function to extract vector info (optimized)
        def vector_info(vec: Vector) -> Dict[str, Any]:
            return {
                "vector_sample": vec.data[:10].tolist(),
                "dimension": vec.dimension,
                "norm": float(np.linalg.norm(vec.data)),
                "hash": self._hash_vector(vec)
            }
        
        result = {
            "name": glyph.name,
            "space_id": glyph.space_id,
            "timestamp": glyph.timestamp.isoformat(),
            "version": glyph.version,
            "global_cortex": vector_info(glyph.global_cortex),
            "layers": {},
            "security_levels": glyph.security_levels,
            "metadata": glyph.metadata
        }
        
        # Add layers
        for layer_name, layer in glyph.layers.items():
            layer_data = {
                "cortex": vector_info(layer.cortex),
                "weights": layer.weights,
                "segments": {}
            }
            
            # Add segments
            for segment_name, segment in layer.segments.items():
                segment_data = {
                    "cortex": vector_info(segment.cortex),
                    "weights": segment.weights,
                    "roles": {}
                }
                
                # Add roles
                for role_name, role_vector in segment.roles.items():
                    role_info = vector_info(role_vector)
                    role_info["value"] = segment.role_values.get(role_name)
                    segment_data["roles"][role_name] = role_info
                
                layer_data["segments"][segment_name] = segment_data
            
            result["layers"][layer_name] = layer_data
        
        return result
    
    def _hash_glyph(self, glyph: Glyph) -> str:
        """Compute hash of glyph for citation."""
        data = f"{glyph.identifier}:{glyph.space_id}:{glyph.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _hash_vector(self, vector: Vector) -> str:
        """Compute hash of vector for citation."""
        return hashlib.sha256(vector.data.tobytes()).hexdigest()[:16]
    
    def _hash_dict(self, d: Dict) -> str:
        """Compute hash of dictionary for citation."""
        data = str(sorted(d.items()))
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def __repr__(self) -> str:
        """String representation of the calculator."""
        return (
            f"SimilarityCalculator(threshold={self.threshold}, "
            f"default_metric='{self.default_metric}')"
        )
