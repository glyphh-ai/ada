"""
GQL (Glyph Query Language) domain for Glyphh Runtime.

This module provides the DatabaseGlyphStorage implementation that enables
the GQL executor to work with database-stored glyphs.
"""

from domains.gql.storage import DatabaseGlyphStorage

__all__ = ["DatabaseGlyphStorage"]
