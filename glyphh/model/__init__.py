"""
Model packaging and deployment module.

This module provides functionality for packaging Glyphh models for deployment:
- GlyphhModel: Packaged model with glyphs, configuration, and metadata
- Model serialization/deserialization to .glyphh files
- Model validation for completeness and consistency
"""

from glyphh.model.package import GlyphhModel

__all__ = ["GlyphhModel"]
