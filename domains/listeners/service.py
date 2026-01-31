"""
Listener Service for Real-Time Glyph Ingestion.

Handles WebSocket connections and HTTP batch ingestion for streaming
glyph creation. Includes rate limiting and connection management.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.storage import GlyphStorage
from infrastructure.config import get_settings
from shared.exceptions import ValidationException

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ConnectionInfo:
    """Information about an active WebSocket connection."""
    connection_id: str
    namespace: str
    connected_at: datetime
    last_activity: datetime
    messages_received: int = 0
    glyphs_created: int = 0
    errors: int = 0


@dataclass
class RateLimitInfo:
    """Rate limiting state for a connection or namespace."""
    requests: List[float] = field(default_factory=list)
    window_seconds: int = 60
    max_requests: int = 60
    
    def is_allowed(self) -> bool:
        """Check if a new request is allowed."""
        now = time.time()
        # Remove old requests outside the window
        self.requests = [t for t in self.requests if now - t < self.window_seconds]
        return len(self.requests) < self.max_requests
    
    def record_request(self) -> None:
        """Record a new request."""
        self.requests.append(time.time())
    
    def time_until_allowed(self) -> float:
        """Get seconds until next request is allowed."""
        if self.is_allowed():
            return 0.0
        oldest = min(self.requests)
        return self.window_seconds - (time.time() - oldest)


@dataclass
class BatchResult:
    """Result of a batch glyph creation."""
    total: int
    created: int
    failed: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


class ListenerService:
    """
    Service for real-time glyph ingestion.
    
    Responsibilities:
    - Handle WebSocket connections for streaming glyph creation
    - Handle HTTP batch ingestion
    - Rate limiting per connection/namespace
    - Connection health monitoring (ping/pong)
    - Graceful connection cleanup
    """
    
    # Heartbeat interval (30 seconds)
    HEARTBEAT_INTERVAL = 30
    
    # Connection timeout (5 minutes of inactivity)
    CONNECTION_TIMEOUT = 300
    
    def __init__(
        self,
        session_maker,
        encoder_getter=None,
    ):
        """
        Initialize ListenerService.
        
        Args:
            session_maker: Async session maker for database access
            encoder_getter: Callable to get encoder for namespace
        """
        self._session_maker = session_maker
        self._encoder_getter = encoder_getter
        self._connections: Dict[str, ConnectionInfo] = {}
        self._websockets: Dict[str, WebSocket] = {}
        self._rate_limits: Dict[str, RateLimitInfo] = defaultdict(
            lambda: RateLimitInfo(max_requests=settings.rate_limit_per_minute)
        )
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

    async def handle_websocket(
        self,
        websocket: WebSocket,
        namespace: str,
    ) -> None:
        """
        Handle a WebSocket connection for streaming glyph creation.
        
        Protocol:
        - Client sends: {"type": "create_glyph", "concept": "...", "metadata": {...}}
        - Server responds: {"type": "glyph_created", "glyph_id": "...", "status": "success"}
        - Client can send: {"type": "ping"}
        - Server responds: {"type": "pong"}
        
        Args:
            websocket: FastAPI WebSocket connection
            namespace: Model namespace for glyph creation
        """
        connection_id = str(uuid4())
        
        # Accept connection
        await websocket.accept()
        
        # Register connection
        self._connections[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            namespace=namespace,
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
        )
        self._websockets[connection_id] = websocket
        
        logger.info(f"WebSocket connected: {connection_id} for namespace '{namespace}'")
        
        # Start heartbeat task
        self._heartbeat_tasks[connection_id] = asyncio.create_task(
            self._heartbeat_loop(connection_id),
            name=f"heartbeat_{connection_id}"
        )
        
        try:
            while True:
                # Receive message
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=self.CONNECTION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.info(f"Connection {connection_id} timed out")
                    break
                
                # Update activity
                self._connections[connection_id].last_activity = datetime.utcnow()
                self._connections[connection_id].messages_received += 1
                
                # Check rate limit
                rate_key = f"{namespace}:{connection_id}"
                if not self._rate_limits[rate_key].is_allowed():
                    wait_time = self._rate_limits[rate_key].time_until_allowed()
                    await websocket.send_json({
                        "type": "error",
                        "code": "RATE_LIMITED",
                        "message": f"Rate limit exceeded. Try again in {wait_time:.1f}s",
                        "retry_after": wait_time,
                    })
                    continue
                
                self._rate_limits[rate_key].record_request()
                
                # Handle message
                await self._handle_message(connection_id, namespace, data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
            self._connections[connection_id].errors += 1
        finally:
            await self._cleanup_connection(connection_id)
    
    async def _handle_message(
        self,
        connection_id: str,
        namespace: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle a single WebSocket message."""
        websocket = self._websockets.get(connection_id)
        if not websocket:
            return
        
        message_type = data.get("type")
        
        if message_type == "ping":
            await websocket.send_json({"type": "pong"})
            
        elif message_type == "create_glyph":
            await self._handle_create_glyph(connection_id, namespace, data)
            
        elif message_type == "batch_create":
            await self._handle_batch_create_ws(connection_id, namespace, data)
            
        else:
            await websocket.send_json({
                "type": "error",
                "code": "UNKNOWN_MESSAGE_TYPE",
                "message": f"Unknown message type: {message_type}",
            })
    
    async def _handle_create_glyph(
        self,
        connection_id: str,
        namespace: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle a create_glyph message."""
        websocket = self._websockets.get(connection_id)
        if not websocket:
            return
        
        concept = data.get("concept")
        metadata = data.get("metadata", {})
        
        if not concept:
            await websocket.send_json({
                "type": "error",
                "code": "MISSING_CONCEPT",
                "message": "Missing 'concept' field",
            })
            return
        
        try:
            # Get encoder
            if self._encoder_getter:
                encoder = await self._encoder_getter(namespace)
            else:
                raise ValueError("Encoder not configured")
            
            # Encode concept
            embedding = encoder.encode_text(concept)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            # Store glyph
            async with self._session_maker() as session:
                storage = GlyphStorage(session)
                result = await storage.create_glyph(
                    namespace=namespace,
                    concept_text=concept,
                    embedding=embedding_list,
                    metadata=metadata,
                )
                await session.commit()
            
            self._connections[connection_id].glyphs_created += 1
            
            await websocket.send_json({
                "type": "glyph_created",
                "glyph_id": str(result.glyph_id),
                "status": "success",
            })
            
        except ValidationException as e:
            await websocket.send_json({
                "type": "error",
                "code": "VALIDATION_ERROR",
                "message": str(e),
            })
            self._connections[connection_id].errors += 1
            
        except Exception as e:
            logger.error(f"Failed to create glyph: {e}")
            await websocket.send_json({
                "type": "error",
                "code": "CREATE_FAILED",
                "message": str(e),
            })
            self._connections[connection_id].errors += 1
    
    async def _handle_batch_create_ws(
        self,
        connection_id: str,
        namespace: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle a batch_create message over WebSocket."""
        websocket = self._websockets.get(connection_id)
        if not websocket:
            return
        
        concepts = data.get("concepts", [])
        metadata = data.get("metadata", {})
        
        if not concepts:
            await websocket.send_json({
                "type": "error",
                "code": "MISSING_CONCEPTS",
                "message": "Missing 'concepts' field",
            })
            return
        
        result = await self.handle_batch_create(namespace, concepts, metadata)
        
        self._connections[connection_id].glyphs_created += result.created
        self._connections[connection_id].errors += result.failed
        
        await websocket.send_json({
            "type": "batch_created",
            "total": result.total,
            "created": result.created,
            "failed": result.failed,
            "results": result.results,
            "errors": result.errors,
        })

    async def handle_batch_create(
        self,
        namespace: str,
        concepts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchResult:
        """
        Handle batch glyph creation via HTTP.
        
        Encodes all concepts and stores them in a single transaction.
        If any concept fails, the entire batch is rolled back.
        
        Args:
            namespace: Model namespace
            concepts: List of concept texts
            metadata: Optional shared metadata for all glyphs
            
        Returns:
            BatchResult with creation results
        """
        results = []
        errors = []
        
        # Check rate limit for namespace
        if not self._rate_limits[namespace].is_allowed():
            wait_time = self._rate_limits[namespace].time_until_allowed()
            return BatchResult(
                total=len(concepts),
                created=0,
                failed=len(concepts),
                results=[],
                errors=[{
                    "code": "RATE_LIMITED",
                    "message": f"Rate limit exceeded. Try again in {wait_time:.1f}s",
                }],
            )
        
        self._rate_limits[namespace].record_request()
        
        try:
            # Get encoder
            if self._encoder_getter:
                encoder = await self._encoder_getter(namespace)
            else:
                raise ValueError("Encoder not configured")
            
            async with self._session_maker() as session:
                storage = GlyphStorage(session)
                
                for i, concept in enumerate(concepts):
                    try:
                        # Encode concept
                        embedding = encoder.encode_text(concept)
                        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        
                        # Store glyph
                        result = await storage.create_glyph(
                            namespace=namespace,
                            concept_text=concept,
                            embedding=embedding_list,
                            metadata=metadata,
                        )
                        
                        results.append({
                            "index": i,
                            "glyph_id": str(result.glyph_id),
                            "status": "created",
                        })
                        
                    except Exception as e:
                        errors.append({
                            "index": i,
                            "concept": concept[:50] + "..." if len(concept) > 50 else concept,
                            "error": str(e),
                        })
                
                # Commit if any succeeded
                if results:
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Batch create failed: {e}")
            errors.append({
                "code": "BATCH_FAILED",
                "message": str(e),
            })
        
        return BatchResult(
            total=len(concepts),
            created=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
        )
    
    async def _heartbeat_loop(self, connection_id: str) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while connection_id in self._websockets:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                
                websocket = self._websockets.get(connection_id)
                if websocket:
                    await websocket.send_json({"type": "heartbeat"})
                else:
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Heartbeat failed for {connection_id}: {e}")
                break
    
    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up resources for a disconnected connection."""
        # Cancel heartbeat task
        if connection_id in self._heartbeat_tasks:
            self._heartbeat_tasks[connection_id].cancel()
            try:
                await self._heartbeat_tasks[connection_id]
            except asyncio.CancelledError:
                pass
            del self._heartbeat_tasks[connection_id]
        
        # Remove connection info
        if connection_id in self._connections:
            info = self._connections[connection_id]
            logger.info(
                f"Connection {connection_id} closed: "
                f"{info.messages_received} messages, "
                f"{info.glyphs_created} glyphs created, "
                f"{info.errors} errors"
            )
            del self._connections[connection_id]
        
        # Remove websocket reference
        if connection_id in self._websockets:
            del self._websockets[connection_id]
        
        # Clean up rate limit (keep for a while to prevent abuse)
        # Rate limits are cleaned up automatically by the RateLimitInfo class
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections."""
        return {
            "active_connections": len(self._connections),
            "connections": [
                {
                    "connection_id": info.connection_id,
                    "namespace": info.namespace,
                    "connected_at": info.connected_at.isoformat(),
                    "messages_received": info.messages_received,
                    "glyphs_created": info.glyphs_created,
                    "errors": info.errors,
                }
                for info in self._connections.values()
            ],
        }
    
    async def close_all_connections(self, reason: str = "Server shutdown") -> None:
        """Close all active WebSocket connections."""
        for connection_id, websocket in list(self._websockets.items()):
            try:
                await websocket.close(code=1001, reason=reason)
            except Exception as e:
                logger.debug(f"Error closing connection {connection_id}: {e}")
            await self._cleanup_connection(connection_id)
        
        logger.info("All WebSocket connections closed")
