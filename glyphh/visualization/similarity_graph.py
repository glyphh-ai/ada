"""
Similarity Graph Visualization

This module provides functions to visualize similarity relationships between
glyphs as a graph using networkx.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from glyphh.core.types import Glyph
from glyphh.similarity.calculator import SimilarityCalculator


def create_similarity_matrix(
    glyphs: List[Glyph],
    edge_type: str = "neural_cortex",
    user_clearance: float = 1.0,
    threshold: float = 0.5
) -> Tuple[np.ndarray, List[str]]:
    """
    Create a similarity matrix for a list of glyphs.
    
    Args:
        glyphs: List of glyphs to compare
        edge_type: Edge type for comparison (default: neural_cortex)
        user_clearance: User security clearance (default: 1.0)
        threshold: Similarity threshold (default: 0.5)
    
    Returns:
        Tuple of (similarity_matrix, glyph_names)
        - similarity_matrix: NxN numpy array of similarity scores
        - glyph_names: List of glyph names in matrix order
    
    Example:
        >>> matrix, names = create_similarity_matrix([glyph1, glyph2, glyph3])
        >>> print(f"Similarity between {names[0]} and {names[1]}: {matrix[0, 1]:.4f}")
    """
    n = len(glyphs)
    matrix = np.zeros((n, n))
    calculator = SimilarityCalculator(threshold=threshold)
    
    # Compute pairwise similarities
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                result = calculator.compute_similarity(
                    glyphs[i],
                    glyphs[j],
                    edge_type=edge_type,
                    user_clearance=user_clearance
                )
                matrix[i, j] = result.score
                matrix[j, i] = result.score
    
    glyph_names = [glyph.name for glyph in glyphs]
    return matrix, glyph_names


def visualize_similarity_graph(
    glyphs: List[Glyph],
    edge_type: str = "neural_cortex",
    user_clearance: float = 1.0,
    threshold: float = 0.5,
    min_edge_weight: float = 0.3,
    output_format: str = "ascii"
) -> str:
    """
    Visualize similarity relationships as a graph.
    
    Args:
        glyphs: List of glyphs to visualize
        edge_type: Edge type for comparison (default: neural_cortex)
        user_clearance: User security clearance (default: 1.0)
        threshold: Similarity threshold (default: 0.5)
        min_edge_weight: Minimum similarity to show edge (default: 0.3)
        output_format: Output format - "ascii" or "networkx" (default: ascii)
    
    Returns:
        String representation of the graph
    
    Example:
        >>> graph = visualize_similarity_graph([glyph1, glyph2, glyph3])
        >>> print(graph)
    """
    # Create similarity matrix
    matrix, names = create_similarity_matrix(
        glyphs, edge_type, user_clearance, threshold
    )
    
    if output_format == "ascii":
        return _visualize_ascii_graph(matrix, names, min_edge_weight)
    elif output_format == "networkx":
        return _visualize_networkx_graph(matrix, names, min_edge_weight)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def _visualize_ascii_graph(
    matrix: np.ndarray,
    names: List[str],
    min_edge_weight: float
) -> str:
    """
    Create ASCII representation of similarity graph.
    
    Args:
        matrix: Similarity matrix
        names: Glyph names
        min_edge_weight: Minimum similarity to show edge
    
    Returns:
        ASCII graph representation
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Similarity Graph (ASCII)")
    lines.append("=" * 70)
    lines.append("")
    
    # Show nodes
    lines.append("Nodes:")
    for i, name in enumerate(names):
        lines.append(f"  [{i}] {name}")
    lines.append("")
    
    # Show edges
    lines.append(f"Edges (similarity >= {min_edge_weight}):")
    edge_count = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if matrix[i, j] >= min_edge_weight:
                edge_count += 1
                # Visual representation of edge strength
                strength = int(matrix[i, j] * 10)
                bar = "#" * strength + "." * (10 - strength)
                lines.append(
                    f"  [{i}] {names[i][:20]:20} --- {matrix[i, j]:.4f} {bar} --- [{j}] {names[j][:20]:20}"
                )
    
    if edge_count == 0:
        lines.append(f"  (No edges with similarity >= {min_edge_weight})")
    
    lines.append("")
    lines.append(f"Total nodes: {len(names)}")
    lines.append(f"Total edges: {edge_count}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def _visualize_networkx_graph(
    matrix: np.ndarray,
    names: List[str],
    min_edge_weight: float
) -> str:
    """
    Create networkx representation of similarity graph.
    
    This requires networkx to be installed. If not available, falls back to ASCII.
    
    Args:
        matrix: Similarity matrix
        names: Glyph names
        min_edge_weight: Minimum similarity to show edge
    
    Returns:
        Graph description or ASCII fallback
    """
    try:
        import networkx as nx
    except ImportError:
        return (
            "NetworkX not installed. Install with: pip install networkx\n"
            "Falling back to ASCII visualization:\n\n"
            + _visualize_ascii_graph(matrix, names, min_edge_weight)
        )
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, name in enumerate(names):
        G.add_node(i, label=name)
    
    # Add edges
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if matrix[i, j] >= min_edge_weight:
                G.add_edge(i, j, weight=matrix[i, j])
    
    # Create description
    lines = []
    lines.append("=" * 70)
    lines.append("Similarity Graph (NetworkX)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Nodes: {G.number_of_nodes()}")
    lines.append(f"Edges: {G.number_of_edges()}")
    lines.append("")
    
    # Show graph info
    lines.append("Graph Properties:")
    lines.append(f"  - Density: {nx.density(G):.4f}")
    
    if G.number_of_edges() > 0:
        lines.append(f"  - Average clustering: {nx.average_clustering(G):.4f}")
        
        # Find connected components
        components = list(nx.connected_components(G))
        lines.append(f"  - Connected components: {len(components)}")
        
        # Find most similar pair
        max_weight = 0
        max_edge = None
        for i, j in G.edges():
            weight = G[i][j]['weight']
            if weight > max_weight:
                max_weight = weight
                max_edge = (i, j)
        
        if max_edge:
            i, j = max_edge
            lines.append(f"  - Most similar pair: {names[i]} <-> {names[j]} ({max_weight:.4f})")
    
    lines.append("")
    lines.append("Node Degrees:")
    for node in G.nodes():
        degree = G.degree(node)
        lines.append(f"  [{node}] {names[node][:30]:30} → {degree} connections")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("Note: Use networkx.draw() to visualize graphically")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def print_similarity_matrix(
    matrix: np.ndarray,
    names: List[str],
    precision: int = 4
) -> str:
    """
    Print similarity matrix in a formatted table.
    
    Args:
        matrix: Similarity matrix
        names: Glyph names
        precision: Decimal precision for scores (default: 4)
    
    Returns:
        Formatted matrix string
    
    Example:
        >>> matrix, names = create_similarity_matrix([glyph1, glyph2, glyph3])
        >>> print(print_similarity_matrix(matrix, names))
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Similarity Matrix")
    lines.append("=" * 70)
    lines.append("")
    
    # Header row
    header = " " * 15
    for name in names:
        header += f"{name[:12]:>12} "
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows
    for i, name in enumerate(names):
        row = f"{name[:12]:>12}  |"
        for j in range(len(names)):
            row += f" {matrix[i, j]:>11.{precision}f}"
        lines.append(row)
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)
