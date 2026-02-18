"""
Vector Heatmap Visualization

This module provides functions to visualize bipolar vectors as heatmaps.
"""

from typing import Optional, Tuple
import numpy as np
from glyphh.core.types import Vector


def visualize_vector_heatmap(
    vector: np.ndarray,
    sample_size: int = 100,
    width: int = 50,
    title: Optional[str] = None
) -> str:
    """
    Visualize a bipolar vector as an ASCII heatmap.
    
    Args:
        vector: Bipolar vector to visualize (numpy array or Vector object)
        sample_size: Number of dimensions to sample (default: 100)
        width: Width of heatmap in characters (default: 50)
        title: Optional title for the heatmap
    
    Returns:
        ASCII heatmap string
    
    Example:
        >>> heatmap = visualize_vector_heatmap(glyph.global_cortex.data)
        >>> print(heatmap)
    """
    # Extract data if Vector object
    if isinstance(vector, Vector):
        vector = vector.data
    
    # Sample the vector if needed
    if len(vector) > sample_size:
        indices = np.linspace(0, len(vector) - 1, sample_size, dtype=int)
        sampled = vector[indices]
    else:
        sampled = vector
    
    lines = []
    lines.append("=" * 70)
    if title:
        lines.append(f"Vector Heatmap: {title}")
    else:
        lines.append("Vector Heatmap")
    lines.append("=" * 70)
    lines.append("")
    
    # Calculate rows and columns
    rows = max(1, len(sampled) // width)
    cols = min(width, len(sampled))
    
    # Display heatmap
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            idx = i * cols + j
            if idx < len(sampled):
                val = sampled[idx]
                symbol = "#" if val > 0 else "."
                row_str += symbol
        lines.append(row_str)
    
    lines.append("")
    lines.append("Legend: # = +1 (positive), . = -1 (negative)")
    lines.append("")
    
    # Statistics
    positive = np.sum(sampled > 0)
    negative = np.sum(sampled < 0)
    total = len(sampled)
    
    lines.append(f"Statistics (sampled {total} dimensions):")
    lines.append(f"  - Positive dimensions: {positive} ({positive / total * 100:.1f}%)")
    lines.append(f"  - Negative dimensions: {negative} ({negative / total * 100:.1f}%)")
    lines.append(f"  - Balance: {abs(positive - negative) / total * 100:.1f}% deviation from 50/50")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def visualize_vector_comparison(
    vector1: np.ndarray,
    vector2: np.ndarray,
    sample_size: int = 100,
    width: int = 50,
    name1: str = "Vector 1",
    name2: str = "Vector 2"
) -> str:
    """
    Visualize comparison between two bipolar vectors.
    
    Shows:
    - Both vectors side by side
    - Agreement/disagreement visualization
    - Similarity statistics
    
    Args:
        vector1: First bipolar vector
        vector2: Second bipolar vector
        sample_size: Number of dimensions to sample (default: 100)
        width: Width of each heatmap in characters (default: 50)
        name1: Name for first vector (default: "Vector 1")
        name2: Name for second vector (default: "Vector 2")
    
    Returns:
        Comparison visualization string
    
    Example:
        >>> comparison = visualize_vector_comparison(
        ...     glyph1.global_cortex.data,
        ...     glyph2.global_cortex.data,
        ...     name1="Red Car",
        ...     name2="Blue Car"
        ... )
        >>> print(comparison)
    """
    # Extract data if Vector objects
    if isinstance(vector1, Vector):
        vector1 = vector1.data
    if isinstance(vector2, Vector):
        vector2 = vector2.data
    
    # Validate dimensions match
    if len(vector1) != len(vector2):
        raise ValueError(
            f"Vector dimensions must match: {len(vector1)} != {len(vector2)}"
        )
    
    # Sample both vectors
    if len(vector1) > sample_size:
        indices = np.linspace(0, len(vector1) - 1, sample_size, dtype=int)
        sampled1 = vector1[indices]
        sampled2 = vector2[indices]
    else:
        sampled1 = vector1
        sampled2 = vector2
    
    lines = []
    lines.append("=" * 70)
    lines.append("Vector Comparison")
    lines.append("=" * 70)
    lines.append("")
    
    # Calculate rows and columns
    rows = max(1, len(sampled1) // width)
    cols = min(width, len(sampled1))
    
    # Display vector 1
    lines.append(f"{name1}:")
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            idx = i * cols + j
            if idx < len(sampled1):
                val = sampled1[idx]
                symbol = "#" if val > 0 else "."
                row_str += symbol
        lines.append(row_str)
    lines.append("")
    
    # Display vector 2
    lines.append(f"{name2}:")
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            idx = i * cols + j
            if idx < len(sampled2):
                val = sampled2[idx]
                symbol = "#" if val > 0 else "."
                row_str += symbol
        lines.append(row_str)
    lines.append("")
    
    # Display agreement/disagreement
    lines.append("Agreement (+ = agree, - = disagree):")
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            idx = i * cols + j
            if idx < len(sampled1):
                agree = sampled1[idx] == sampled2[idx]
                symbol = "+" if agree else "-"
                row_str += symbol
        lines.append(row_str)
    lines.append("")
    
    lines.append("Legend:")
    lines.append("  # = +1 (positive), . = -1 (negative)")
    lines.append("  + = dimensions agree, - = dimensions disagree")
    lines.append("")
    
    # Compute similarity statistics
    agreement = np.sum(sampled1 == sampled2)
    disagreement = np.sum(sampled1 != sampled2)
    total = len(sampled1)
    
    # Hamming similarity
    hamming_sim = agreement / total
    
    # Cosine similarity
    dot_product = np.dot(sampled1, sampled2)
    norm_product = np.linalg.norm(sampled1) * np.linalg.norm(sampled2)
    cosine_sim = dot_product / norm_product if norm_product > 0 else 0
    
    lines.append(f"Similarity Statistics (sampled {total} dimensions):")
    lines.append(f"  - Dimensions agreeing: {agreement} ({hamming_sim * 100:.1f}%)")
    lines.append(f"  - Dimensions disagreeing: {disagreement} ({(1 - hamming_sim) * 100:.1f}%)")
    lines.append(f"  - Hamming similarity: {hamming_sim:.4f}")
    lines.append(f"  - Cosine similarity: {cosine_sim:.4f}")
    
    # Interpretation
    lines.append("")
    lines.append("Interpretation:")
    if hamming_sim >= 0.9:
        lines.append("  → Very high similarity (nearly identical)")
    elif hamming_sim >= 0.7:
        lines.append("  → High similarity (very similar)")
    elif hamming_sim >= 0.5:
        lines.append("  → Moderate similarity (somewhat similar)")
    elif hamming_sim >= 0.3:
        lines.append("  → Low similarity (somewhat different)")
    else:
        lines.append("  → Very low similarity (very different)")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def visualize_vector_distribution(
    vector: np.ndarray,
    bins: int = 20,
    title: Optional[str] = None
) -> str:
    """
    Visualize the distribution of vector values.
    
    For bipolar vectors, this shows the balance between +1 and -1 values.
    
    Args:
        vector: Vector to visualize
        bins: Number of bins for histogram (default: 20)
        title: Optional title
    
    Returns:
        Distribution visualization string
    
    Example:
        >>> dist = visualize_vector_distribution(glyph.global_cortex.data)
        >>> print(dist)
    """
    # Extract data if Vector object
    if isinstance(vector, Vector):
        vector = vector.data
    
    lines = []
    lines.append("=" * 70)
    if title:
        lines.append(f"Vector Distribution: {title}")
    else:
        lines.append("Vector Distribution")
    lines.append("=" * 70)
    lines.append("")
    
    # For bipolar vectors, just show +1 and -1 counts
    positive = np.sum(vector > 0)
    negative = np.sum(vector < 0)
    zero = np.sum(vector == 0)
    total = len(vector)
    
    # Create bar chart
    max_count = max(positive, negative)
    pos_bar_len = int((positive / max_count) * 50) if max_count > 0 else 0
    neg_bar_len = int((negative / max_count) * 50) if max_count > 0 else 0
    
    lines.append("Value Distribution:")
    lines.append(f"  +1: {'#' * pos_bar_len} {positive} ({positive / total * 100:.1f}%)")
    lines.append(f"  -1: {'#' * neg_bar_len} {negative} ({negative / total * 100:.1f}%)")
    
    if zero > 0:
        zero_bar_len = int((zero / max_count) * 50)
        lines.append(f"   0: {'#' * zero_bar_len} {zero} ({zero / total * 100:.1f}%)")
    
    lines.append("")
    lines.append(f"Total dimensions: {total}")
    lines.append(f"Balance: {abs(positive - negative)} dimension difference")
    
    # Check if vector is properly bipolar
    if zero > 0:
        lines.append("")
        lines.append("Warning: Vector contains zero values (not strictly bipolar)")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)
