"""
EdgeGenerator for creating spatial and temporal edges.

This module implements the EdgeGenerator class that extracts edges from glyphs
at all hierarchy levels (cortex, layer, segment, role). These edges enable
flexible similarity computation and change detection.

The EdgeGenerator is a CORE DIFFERENTIATOR for Glyphh vs Knowledge Graphs and
Vector DBs, enabling multi-level verification and reasoning.
"""

from typing import List, Dict, Optional
from glyphh.core.types import Edge, Glyph, Vector
from glyphh.core.rust_ops import bind


class EdgeGenerator:
    """
    Generator for spatial and temporal edges.
    
    The EdgeGenerator extracts edges from glyphs at different hierarchical levels,
    enabling flexible similarity computation and change detection. This is a core
    differentiator for Glyphh, allowing verification at multiple levels:
    - Global (cortex): Overall similarity
    - Layer: Layer-specific patterns
    - Segment: Segment-specific features
    - Role: Individual attribute comparison
    
    Edge Types:
    -----------
    Spatial Edges (4 types):
    - neural_cortex: Global similarity between glyphs
    - neural_layer: Layer-specific similarity
    - neural_segment: Segment-specific similarity
    - neural_role: Role-specific similarity
    
    Temporal Edges (4 types):
    - temporal_cortex: Global change detection
    - temporal_layer: Layer-specific change
    - temporal_segment: Segment-specific change
    - temporal_role: Role-specific change
    
    Example:
        >>> generator = EdgeGenerator()
        >>> 
        >>> # Generate spatial edges for similarity computation
        >>> spatial_edges = generator.generate_spatial_edges(glyph)
        >>> print(f"Generated {len(spatial_edges)} spatial edges")
        >>> 
        >>> # Generate temporal edges for change detection
        >>> temporal_edges = generator.generate_temporal_edges(glyph_v1, glyph_v2)
        >>> print(f"Generated {len(temporal_edges)} temporal edges")
        >>> 
        >>> # Get specific edge type
        >>> cortex_edges = [e for e in spatial_edges if e.type == "neural_cortex"]
        >>> layer_edges = [e for e in spatial_edges if e.type == "neural_layer"]
    """
    
    def generate_spatial_edges(self, glyph: Glyph) -> List[Edge]:
        """
        Generate 4 spatial edge types for similarity computation.
        
        Spatial edges represent the glyph's structure at different hierarchical
        levels, enabling similarity computation at each level:
        
        1. neural_cortex: Global similarity (entire glyph)
        2. neural_layer: Layer-specific similarity (per layer)
        3. neural_segment: Segment-specific similarity (per segment)
        4. neural_role: Role-specific similarity (per role/attribute)
        
        Args:
            glyph: Glyph to extract edges from
        
        Returns:
            List of spatial edges (one cortex edge, N layer edges, M segment edges, K role edges)
        
        Example:
            >>> generator = EdgeGenerator()
            >>> glyph = encoder.encode(concept)
            >>> edges = generator.generate_spatial_edges(glyph)
            >>> 
            >>> # Edges are organized by type
            >>> cortex_edge = [e for e in edges if e.type == "neural_cortex"][0]
            >>> layer_edges = [e for e in edges if e.type == "neural_layer"]
            >>> segment_edges = [e for e in edges if e.type == "neural_segment"]
            >>> role_edges = [e for e in edges if e.type == "neural_role"]
        """
        edges = []
        
        # 1. Neural cortex: global similarity
        edges.append(Edge(
            type="neural_cortex",
            source=glyph.identifier,
            target=None,
            vector=glyph.global_cortex,
            weights={"cortex": 1.0},
            security_level=glyph.security_levels.get("cortex", 0.0),
            layer=None,
            segment=None,
            role=None
        ))
        
        # 2. Neural layer: per-layer similarity
        for layer_idx, (layer_name, layer) in enumerate(glyph.layers.items()):
            edges.append(Edge(
                type="neural_layer",
                source=glyph.identifier,
                target=None,
                vector=layer.cortex,
                weights={
                    "layer": layer.weights.get("layer", 1.0),
                    **layer.weights
                },
                security_level=glyph.security_levels.get(f"layer_{layer_idx}", 
                                                         glyph.security_levels.get("layer", 0.0)),
                layer=layer_idx,
                segment=None,
                role=None
            ))
            
            # 3. Neural segment: per-segment similarity
            for seg_idx, (segment_name, segment) in enumerate(layer.segments.items()):
                edges.append(Edge(
                    type="neural_segment",
                    source=glyph.identifier,
                    target=None,
                    vector=segment.cortex,
                    weights={
                        "segment": segment.weights.get("segment", 1.0),
                        **segment.weights
                    },
                    security_level=glyph.security_levels.get(
                        f"segment_{layer_idx}_{seg_idx}",
                        glyph.security_levels.get("segment", 0.0)
                    ),
                    layer=layer_idx,
                    segment=seg_idx,
                    role=None
                ))
                
                # 4. Neural role: per-role similarity
                for role_name, role_vector in segment.roles.items():
                    edges.append(Edge(
                        type="neural_role",
                        source=glyph.identifier,
                        target=None,
                        vector=role_vector,
                        weights={
                            "role": segment.weights.get(role_name, 1.0),
                            **{role_name: segment.weights.get(role_name, 1.0)}
                        },
                        security_level=glyph.security_levels.get(
                            f"role_{layer_idx}_{seg_idx}_{role_name}",
                            glyph.security_levels.get("role", 0.0)
                        ),
                        layer=layer_idx,
                        segment=seg_idx,
                        role=role_name
                    ))
        
        return edges
    
    def generate_temporal_edges(self, glyph_v1: Glyph, glyph_v2: Glyph) -> List[Edge]:
        """
        Generate 4 temporal edge types for change detection.
        
        Temporal edges represent changes between two versions of a glyph at
        different hierarchical levels, enabling change detection at each level:
        
        1. temporal_cortex: Global change (entire glyph)
        2. temporal_layer: Layer-specific change (per layer)
        3. temporal_segment: Segment-specific change (per segment)
        4. temporal_role: Role-specific change (per role/attribute)
        
        The temporal delta is computed using the bind operation with inverse:
        delta = bind(v2, inverse(v1))
        
        This allows us to:
        - Detect what changed between versions
        - Apply deltas to create new versions
        - Track evolution over time
        
        Args:
            glyph_v1: Earlier version of the glyph
            glyph_v2: Later version of the glyph
        
        Returns:
            List of temporal edges (one cortex edge, N layer edges, M segment edges, K role edges)
        
        Raises:
            ValueError: If glyphs have different space_ids or structures
        
        Example:
            >>> generator = EdgeGenerator()
            >>> glyph_v1 = encoder.encode(concept_v1)
            >>> glyph_v2 = encoder.encode(concept_v2)
            >>> edges = generator.generate_temporal_edges(glyph_v1, glyph_v2)
            >>> 
            >>> # Edges represent changes at each level
            >>> cortex_change = [e for e in edges if e.type == "temporal_cortex"][0]
            >>> layer_changes = [e for e in edges if e.type == "temporal_layer"]
            >>> segment_changes = [e for e in edges if e.type == "temporal_segment"]
            >>> role_changes = [e for e in edges if e.type == "temporal_role"]
        """
        # Validate glyphs are from same vector space
        if glyph_v1.space_id != glyph_v2.space_id:
            raise ValueError(
                f"Cannot compute temporal edges for glyphs from different vector spaces: "
                f"{glyph_v1.space_id} vs {glyph_v2.space_id}"
            )
        
        edges = []
        
        # 1. Temporal cortex: global change detection
        cortex_delta = self._compute_temporal_delta(
            glyph_v1.global_cortex,
            glyph_v2.global_cortex
        )
        edges.append(Edge(
            type="temporal_cortex",
            source=glyph_v1.identifier,
            target=glyph_v2.identifier,
            vector=cortex_delta,
            weights={"cortex": 1.0},
            security_level=max(
                glyph_v1.security_levels.get("cortex", 0.0),
                glyph_v2.security_levels.get("cortex", 0.0)
            ),
            layer=None,
            segment=None,
            role=None
        ))
        
        # 2. Temporal layer: per-layer change detection
        # Match layers by name
        common_layers = set(glyph_v1.layers.keys()) & set(glyph_v2.layers.keys())
        
        for layer_idx, layer_name in enumerate(sorted(common_layers)):
            layer_v1 = glyph_v1.layers[layer_name]
            layer_v2 = glyph_v2.layers[layer_name]
            
            layer_delta = self._compute_temporal_delta(
                layer_v1.cortex,
                layer_v2.cortex
            )
            
            edges.append(Edge(
                type="temporal_layer",
                source=glyph_v1.identifier,
                target=glyph_v2.identifier,
                vector=layer_delta,
                weights={
                    "layer": max(
                        layer_v1.weights.get("layer", 1.0),
                        layer_v2.weights.get("layer", 1.0)
                    )
                },
                security_level=max(
                    glyph_v1.security_levels.get(f"layer_{layer_idx}", 
                                                 glyph_v1.security_levels.get("layer", 0.0)),
                    glyph_v2.security_levels.get(f"layer_{layer_idx}",
                                                 glyph_v2.security_levels.get("layer", 0.0))
                ),
                layer=layer_idx,
                segment=None,
                role=None
            ))
            
            # 3. Temporal segment: per-segment change detection
            # Match segments by name within this layer
            common_segments = set(layer_v1.segments.keys()) & set(layer_v2.segments.keys())
            
            for seg_idx, segment_name in enumerate(sorted(common_segments)):
                segment_v1 = layer_v1.segments[segment_name]
                segment_v2 = layer_v2.segments[segment_name]
                
                segment_delta = self._compute_temporal_delta(
                    segment_v1.cortex,
                    segment_v2.cortex
                )
                
                edges.append(Edge(
                    type="temporal_segment",
                    source=glyph_v1.identifier,
                    target=glyph_v2.identifier,
                    vector=segment_delta,
                    weights={
                        "segment": max(
                            segment_v1.weights.get("segment", 1.0),
                            segment_v2.weights.get("segment", 1.0)
                        )
                    },
                    security_level=max(
                        glyph_v1.security_levels.get(
                            f"segment_{layer_idx}_{seg_idx}",
                            glyph_v1.security_levels.get("segment", 0.0)
                        ),
                        glyph_v2.security_levels.get(
                            f"segment_{layer_idx}_{seg_idx}",
                            glyph_v2.security_levels.get("segment", 0.0)
                        )
                    ),
                    layer=layer_idx,
                    segment=seg_idx,
                    role=None
                ))
                
                # 4. Temporal role: per-role change detection
                # Match roles by name within this segment
                common_roles = set(segment_v1.roles.keys()) & set(segment_v2.roles.keys())
                
                for role_name in sorted(common_roles):
                    role_v1 = segment_v1.roles[role_name]
                    role_v2 = segment_v2.roles[role_name]
                    
                    role_delta = self._compute_temporal_delta(role_v1, role_v2)
                    
                    edges.append(Edge(
                        type="temporal_role",
                        source=glyph_v1.identifier,
                        target=glyph_v2.identifier,
                        vector=role_delta,
                        weights={
                            "role": max(
                                segment_v1.weights.get(role_name, 1.0),
                                segment_v2.weights.get(role_name, 1.0)
                            ),
                            role_name: max(
                                segment_v1.weights.get(role_name, 1.0),
                                segment_v2.weights.get(role_name, 1.0)
                            )
                        },
                        security_level=max(
                            glyph_v1.security_levels.get(
                                f"role_{layer_idx}_{seg_idx}_{role_name}",
                                glyph_v1.security_levels.get("role", 0.0)
                            ),
                            glyph_v2.security_levels.get(
                                f"role_{layer_idx}_{seg_idx}_{role_name}",
                                glyph_v2.security_levels.get("role", 0.0)
                            )
                        ),
                        layer=layer_idx,
                        segment=seg_idx,
                        role=role_name
                    ))
        
        return edges
    
    def _compute_temporal_delta(self, v1: Vector, v2: Vector) -> Vector:
        """
        Compute temporal delta between two vectors.
        
        The temporal delta represents the change from v1 to v2, computed using:
        delta = bind(v2, inverse(v1))
        
        This allows us to:
        - Apply delta to v1 to get v2: bind(v1, delta) = v2
        - Detect what changed between versions
        - Track evolution over time
        
        Args:
            v1: Earlier version vector
            v2: Later version vector
        
        Returns:
            Delta vector representing the change
        
        Raises:
            ValueError: If vectors have different dimensions or space_ids
        
        Example:
            >>> generator = EdgeGenerator()
            >>> v1 = encoder.generate_symbol("red")
            >>> v2 = encoder.generate_symbol("blue")
            >>> delta = generator._compute_temporal_delta(v1, v2)
            >>> 
            >>> # Apply delta to v1 to get v2
            >>> v2_reconstructed = encoder.bind(v1, delta)
            >>> assert np.array_equal(v2_reconstructed.data, v2.data)
        """
        # Validate vectors are compatible
        if v1.dimension != v2.dimension:
            raise ValueError(
                f"Cannot compute temporal delta for vectors with different dimensions: "
                f"{v1.dimension} vs {v2.dimension}"
            )
        
        if v1.space_id != v2.space_id:
            raise ValueError(
                f"Cannot compute temporal delta for vectors from different spaces: "
                f"{v1.space_id} vs {v2.space_id}"
            )
        
        # Compute inverse of v1 (for bipolar vectors, inverse is the vector itself)
        # Since bind(v, v) = all 1s, and bind(v, -v) = all -1s
        # For bipolar vectors, inverse(v) = v (because v * v = 1 for all dimensions)
        v1_inverse = v1
        
        # Compute delta: bind(v2, inverse(v1))
        delta_data = bind(v2.data, v1_inverse.data)
        
        return Vector(
            data=delta_data,
            dimension=v1.dimension,
            space_id=v1.space_id
        )
    
    def get_edges_by_type(self, edges: List[Edge], edge_type: str) -> List[Edge]:
        """
        Filter edges by type.
        
        Args:
            edges: List of edges to filter
            edge_type: Edge type to filter for (e.g., "neural_cortex", "temporal_layer")
        
        Returns:
            List of edges matching the specified type
        
        Example:
            >>> generator = EdgeGenerator()
            >>> all_edges = generator.generate_spatial_edges(glyph)
            >>> cortex_edges = generator.get_edges_by_type(all_edges, "neural_cortex")
            >>> layer_edges = generator.get_edges_by_type(all_edges, "neural_layer")
        """
        return [edge for edge in edges if edge.type == edge_type]
    
    def get_edges_by_hierarchy(
        self,
        edges: List[Edge],
        layer: Optional[int] = None,
        segment: Optional[int] = None,
        role: Optional[str] = None
    ) -> List[Edge]:
        """
        Filter edges by hierarchical level.
        
        Args:
            edges: List of edges to filter
            layer: Layer index to filter for (None = any layer)
            segment: Segment index to filter for (None = any segment)
            role: Role name to filter for (None = any role)
        
        Returns:
            List of edges matching the specified hierarchy level
        
        Example:
            >>> generator = EdgeGenerator()
            >>> all_edges = generator.generate_spatial_edges(glyph)
            >>> 
            >>> # Get all edges for layer 0
            >>> layer_0_edges = generator.get_edges_by_hierarchy(all_edges, layer=0)
            >>> 
            >>> # Get all edges for layer 0, segment 1
            >>> segment_edges = generator.get_edges_by_hierarchy(all_edges, layer=0, segment=1)
            >>> 
            >>> # Get all edges for a specific role
            >>> color_edges = generator.get_edges_by_hierarchy(all_edges, role="color")
        """
        filtered = edges
        
        if layer is not None:
            filtered = [e for e in filtered if e.layer == layer]
        
        if segment is not None:
            filtered = [e for e in filtered if e.segment == segment]
        
        if role is not None:
            filtered = [e for e in filtered if e.role == role]
        
        return filtered
    
    def __repr__(self) -> str:
        """String representation of the EdgeGenerator."""
        return "EdgeGenerator(spatial_types=4, temporal_types=4)"
