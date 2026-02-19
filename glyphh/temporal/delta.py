"""
Temporal delta computation for version tracking and updates.

This module provides the TemporalEncoder class for computing temporal deltas
between vector versions and applying deltas to update vectors.
"""

import numpy as np
from typing import List, Dict
from glyphh.core.types import Vector
from glyphh.core.ops import bind, bundle


class TemporalEncoder:
    """
    Encoder for temporal delta computation and application.
    
    The TemporalEncoder provides methods for:
    - Computing temporal deltas between vector versions
    - Applying deltas to update vectors
    - Encoding segments for incremental updates
    - Merging segments using weighted bundle operations
    
    Temporal deltas are computed using the bind operation with inverse:
        delta = bind(v2, inverse(v1))
    
    This allows us to apply the delta to v1 to get v2:
        v2 = bind(v1, delta)
    
    Example:
        >>> encoder = TemporalEncoder()
        >>> v1 = Vector(data=np.array([-1, 1, -1, 1], dtype=np.int8), dimension=4, space_id="test")
        >>> v2 = Vector(data=np.array([1, 1, -1, -1], dtype=np.int8), dimension=4, space_id="test")
        >>> delta = encoder.compute_temporal_delta(v1, v2)
        >>> v2_reconstructed = encoder.apply_temporal_delta(v1, delta)
        >>> assert np.array_equal(v2.data, v2_reconstructed.data)
    """
    
    def compute_temporal_delta(self, v1: Vector, v2: Vector) -> Vector:
        """
        Compute change vector between two versions.
        
        The delta is computed using bind:
            delta = bind(v2, v1)
        
        This delta can be applied to v1 to reconstruct v2:
            v2 = bind(v1, delta) = bind(v1, bind(v2, v1)) = v2
        
        This works because for bipolar vectors, v1 * v1 = 1 (identity).
        
        Args:
            v1: Earlier version vector
            v2: Later version vector
        
        Returns:
            Delta vector representing the change from v1 to v2
        
        Raises:
            ValueError: If vectors have different dimensions or space_ids
        
        Example:
            >>> encoder = TemporalEncoder()
            >>> v1 = Vector(data=np.array([-1, 1, -1, 1], dtype=np.int8), dimension=4, space_id="test")
            >>> v2 = Vector(data=np.array([1, 1, -1, -1], dtype=np.int8), dimension=4, space_id="test")
            >>> delta = encoder.compute_temporal_delta(v1, v2)
        """
        # Validate vectors are in same space
        if v1.space_id != v2.space_id:
            raise ValueError(
                f"Cannot compute delta between vectors from different spaces: "
                f"v1.space_id={v1.space_id}, v2.space_id={v2.space_id}"
            )
        
        # Validate dimensions match
        if v1.dimension != v2.dimension:
            raise ValueError(
                f"Cannot compute delta between vectors with different dimensions: "
                f"v1.dimension={v1.dimension}, v2.dimension={v2.dimension}"
            )
        
        # Compute delta: bind(v2, v1)
        delta_data = bind(v2.data, v1.data)
        
        return Vector(
            data=delta_data,
            dimension=v1.dimension,
            space_id=v1.space_id
        )
    
    def apply_temporal_delta(self, base: Vector, delta: Vector) -> Vector:
        """
        Apply delta to base vector to get new version.
        
        The new version is computed using bind:
            new_version = bind(base, delta)
        
        Args:
            base: Base vector to apply delta to
            delta: Delta vector to apply
        
        Returns:
            New version vector after applying delta
        
        Raises:
            ValueError: If vectors have different dimensions or space_ids
        
        Example:
            >>> encoder = TemporalEncoder()
            >>> base = Vector(data=np.array([-1, 1, -1, 1], dtype=np.int8), dimension=4, space_id="test")
            >>> delta = Vector(data=np.array([-1, 1, 1, -1], dtype=np.int8), dimension=4, space_id="test")
            >>> new_version = encoder.apply_temporal_delta(base, delta)
        """
        # Validate vectors are in same space
        if base.space_id != delta.space_id:
            raise ValueError(
                f"Cannot apply delta from different space: "
                f"base.space_id={base.space_id}, delta.space_id={delta.space_id}"
            )
        
        # Validate dimensions match
        if base.dimension != delta.dimension:
            raise ValueError(
                f"Cannot apply delta with different dimension: "
                f"base.dimension={base.dimension}, delta.dimension={delta.dimension}"
            )
        
        # Apply delta: bind(base, delta)
        new_data = bind(base.data, delta.data)
        
        return Vector(
            data=new_data,
            dimension=base.dimension,
            space_id=base.space_id
        )
    
    def encode_segment(self, segment_data: Dict, space_id: str, dimension: int) -> Vector:
        """
        Encode a single segment for incremental updates.
        
        This is a placeholder for segment encoding. In practice, this would
        use the full encoder to create role bindings and bundle them.
        
        Args:
            segment_data: Dictionary of segment data
            space_id: Vector space identifier
            dimension: Vector dimension
        
        Returns:
            Encoded segment vector
        
        Note:
            This is a simplified implementation. Full segment encoding
            should use the Encoder class to create proper role bindings.
        """
        # This is a placeholder - in practice, use full Encoder
        # For now, just create a simple vector
        from glyphh.core.ops import generate_symbol
        
        # Generate a symbol based on segment data
        segment_key = str(sorted(segment_data.items()))
        return Vector(
            data=generate_symbol(42, segment_key, dimension),
            dimension=dimension,
            space_id=space_id
        )
    
    def merge_segments(
        self,
        segments: List[Vector],
        weights: List[float] = None
    ) -> Vector:
        """
        Merge multiple segments using weighted bundle operation.
        
        If weights are provided, each segment is repeated proportionally
        to its weight before bundling. This allows for weighted aggregation
        while maintaining the majority-vote semantics of bundle.
        
        Args:
            segments: List of segment vectors to merge
            weights: Optional list of weights (must sum to 1.0 if provided)
        
        Returns:
            Merged vector (weighted bundle of segments)
        
        Raises:
            ValueError: If segments list is empty, dimensions don't match,
                       space_ids don't match, or weights don't sum to 1.0
        
        Example:
            >>> encoder = TemporalEncoder()
            >>> seg1 = Vector(data=np.array([-1, 1, -1, 1], dtype=np.int8), dimension=4, space_id="test")
            >>> seg2 = Vector(data=np.array([1, 1, -1, -1], dtype=np.int8), dimension=4, space_id="test")
            >>> merged = encoder.merge_segments([seg1, seg2], weights=[0.7, 0.3])
        """
        if not segments:
            raise ValueError("Cannot merge empty segment list")
        
        # Validate all segments have same dimension and space_id
        dimension = segments[0].dimension
        space_id = segments[0].space_id
        
        for i, seg in enumerate(segments):
            if seg.dimension != dimension:
                raise ValueError(
                    f"Dimension mismatch: segment 0 has dimension {dimension}, "
                    f"segment {i} has dimension {seg.dimension}"
                )
            if seg.space_id != space_id:
                raise ValueError(
                    f"Space ID mismatch: segment 0 has space_id {space_id}, "
                    f"segment {i} has space_id {seg.space_id}"
                )
        
        # If no weights provided, use equal weights
        if weights is None:
            weights = [1.0 / len(segments)] * len(segments)
        
        # Validate weights
        if len(weights) != len(segments):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of segments ({len(segments)})"
            )
        
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {sum(weights)}"
            )
        
        # Weighted bundle: repeat each segment proportionally to its weight
        # Convert weights to integer repetitions (scale by 100 for precision)
        scale = 100
        repetitions = [max(1, int(w * scale)) for w in weights]
        
        # Create weighted list of segments
        weighted_segments = []
        for seg, rep in zip(segments, repetitions):
            weighted_segments.extend([seg.data] * rep)
        
        # Bundle all weighted segments
        merged_data = bundle(weighted_segments)
        
        return Vector(
            data=merged_data,
            dimension=dimension,
            space_id=space_id
        )
    
    def _inverse(self, v: np.ndarray) -> np.ndarray:
        """
        Compute inverse of a bipolar vector.
        
        For bipolar vectors {-1, +1}, the inverse is simply the negation:
            inverse(v) = -v
        
        This is because bind(v, inverse(v)) = bind(v, -v) = -v^2 = -1 (identity for bind)
        
        Args:
            v: Bipolar vector
        
        Returns:
            Inverse vector (negation)
        
        Example:
            >>> encoder = TemporalEncoder()
            >>> v = np.array([-1, 1, -1, 1], dtype=np.int8)
            >>> v_inv = encoder._inverse(v)
            >>> print(v_inv)
            [1, -1, 1, -1]
        """
        return (-v).astype(np.int8)
