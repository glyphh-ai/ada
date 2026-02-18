"""
Similarity computation module for the Glyphh SDK.

This module provides weighted similarity computation with dual weighting:
- Similarity weights (importance)
- Security weights (visibility/access control)

Key Components:
- SimilarityCalculator: Main class for computing weighted similarity
- SimilarityResult: Result dataclass with score, visibility, and metadata
"""

from glyphh.similarity.calculator import SimilarityCalculator, SimilarityResult

__all__ = ["SimilarityCalculator", "SimilarityResult"]
