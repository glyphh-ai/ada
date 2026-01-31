"""
Listeners Domain.

Handles real-time glyph ingestion via WebSocket and HTTP batch endpoints.
"""

from domains.listeners.service import BatchResult, ConnectionInfo, ListenerService

__all__ = ["ListenerService", "ConnectionInfo", "BatchResult"]
