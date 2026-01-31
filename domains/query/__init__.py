"""
Query domain - Query processing and result generation.

This domain handles:
- Similarity search with weighted scoring
- Fact tree generation
- Temporal prediction
"""

from domains.query.service import QueryService

__all__ = ["QueryService"]
