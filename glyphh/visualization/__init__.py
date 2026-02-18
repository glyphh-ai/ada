"""
Visualization module for the Glyphh SDK.

This module provides visualization helpers for:
- Similarity graphs with networkx
- Vector heatmaps
- Fact tree visualization
- Glyph structure inspection
"""

from glyphh.visualization.similarity_graph import (
    visualize_similarity_graph,
    create_similarity_matrix
)
from glyphh.visualization.vector_heatmap import (
    visualize_vector_heatmap,
    visualize_vector_comparison
)
from glyphh.visualization.fact_tree_viz import (
    visualize_fact_tree,
    print_fact_tree
)

__all__ = [
    # Similarity graph
    "visualize_similarity_graph",
    "create_similarity_matrix",
    
    # Vector heatmap
    "visualize_vector_heatmap",
    "visualize_vector_comparison",
    
    # Fact tree
    "visualize_fact_tree",
    "print_fact_tree",
]
