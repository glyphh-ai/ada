"""
Listeners Domain.

Handles async glyph ingestion via HTTP batch endpoints.
"""

from domains.listeners.async_service import AsyncListenerService

__all__ = ["AsyncListenerService"]
