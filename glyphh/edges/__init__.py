"""
Edge-based reasoning framework for the Glyphh SDK.

This module provides the EdgeGenerator class for creating spatial and temporal edges
that enable flexible similarity computation at different hierarchical levels.

Edge Types:
-----------
Spatial Edges (similarity at different hierarchical levels):
1. neural_cortex: Global similarity across entire glyphs
2. neural_layer: Layer-specific similarity (compare layer N to layer N)
3. neural_segment: Segment-specific similarity (compare segment M to segment M)
4. neural_role: Role-specific similarity (compare "color" to "color")

Temporal Edges (change detection over time):
5. temporal_cortex: Global change between versions
6. temporal_layer: Layer-specific change over time
7. temporal_segment: Segment-specific change over time
8. temporal_role: Role/attribute change over time

Each edge type supports weighted similarity computation with security filtering.
"""

from glyphh.edges.generator import EdgeGenerator

__all__ = ["EdgeGenerator"]
