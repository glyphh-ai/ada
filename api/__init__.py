"""Glyph AI server API package.

This package is primarily served via Uvicorn. For convenience, we re-export
the FastAPI application as `api:app` in addition to `api.main:app`.
"""

from .main import app

__all__ = ["app"]
