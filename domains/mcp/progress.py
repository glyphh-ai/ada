"""
MCP Progress Notifications Handler.

Implements MCP protocol progress notifications for long-running operations.
When a client provides a progressToken, the server sends notifications/progress
messages during execution.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPProgressNotification:
    """MCP protocol progress notification data."""
    progress_token: str
    progress: int  # 0-100
    total: Optional[int] = None
    message: Optional[str] = None


class MCPProgressHandler:
    """
    Handles MCP progress notifications for long-running operations.
    
    When a client provides a progressToken, the server sends
    notifications/progress messages during execution per MCP protocol.
    
    Usage:
        handler = MCPProgressHandler(send_notification_func)
        await handler.notify(token, progress=50, message="Processing...")
    """
    
    def __init__(
        self,
        send_notification: Callable[[dict], Coroutine[Any, Any, None]],
    ):
        """
        Initialize progress handler.
        
        Args:
            send_notification: Async function to send JSON-RPC notifications
        """
        self._send = send_notification
        self._last_progress: dict[str, int] = {}
    
    async def notify(
        self,
        token: str,
        progress: int,
        total: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        """
        Send progress notification to client.
        
        Args:
            token: Progress token from client request
            progress: Progress percentage (0-100)
            total: Optional total items being processed
            message: Optional status message
        """
        # Clamp progress to valid range
        progress = max(0, min(100, progress))
        
        # Build notification per MCP protocol
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {
                "progressToken": token,
                "progress": progress,
            },
        }
        
        if total is not None:
            notification["params"]["total"] = total
        
        if message is not None:
            notification["params"]["message"] = message
        
        try:
            await self._send(notification)
            self._last_progress[token] = progress
            logger.debug(f"Sent progress notification: {progress}% - {message}")
        except Exception as e:
            logger.warning(f"Failed to send progress notification: {e}")
    
    async def notify_if_changed(
        self,
        token: str,
        progress: int,
        total: Optional[int] = None,
        message: Optional[str] = None,
        min_delta: int = 5,
    ) -> None:
        """
        Send progress notification only if progress changed significantly.
        
        Useful for avoiding flooding the client with too many notifications.
        
        Args:
            token: Progress token from client request
            progress: Progress percentage (0-100)
            total: Optional total items being processed
            message: Optional status message
            min_delta: Minimum progress change to trigger notification
        """
        last = self._last_progress.get(token, -min_delta)
        if abs(progress - last) >= min_delta or progress == 100:
            await self.notify(token, progress, total, message)
    
    def clear_token(self, token: str) -> None:
        """Clear tracking for a token (call when operation completes)."""
        self._last_progress.pop(token, None)


class ProgressTracker:
    """
    Helper class for tracking progress through multi-step operations.
    
    Usage:
        tracker = ProgressTracker(handler, token, total_items=100)
        for i, item in enumerate(items):
            process(item)
            await tracker.update(i + 1, f"Processed {i + 1} items")
        await tracker.complete("Done!")
    """
    
    def __init__(
        self,
        handler: Optional[MCPProgressHandler],
        token: Optional[str],
        total_items: int = 100,
        update_interval: int = 10,
    ):
        """
        Initialize progress tracker.
        
        Args:
            handler: MCPProgressHandler instance (can be None if no token)
            token: Progress token from client (can be None)
            total_items: Total number of items to process
            update_interval: Send update every N items (or 10% whichever is smaller)
        """
        self._handler = handler
        self._token = token
        self._total = total_items
        self._interval = min(update_interval, max(1, total_items // 10))
        self._last_update = 0
    
    @property
    def is_active(self) -> bool:
        """Check if progress tracking is active."""
        return self._handler is not None and self._token is not None
    
    async def update(
        self,
        current: int,
        message: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Update progress.
        
        Args:
            current: Current item number (1-based)
            message: Optional status message
            force: Force send even if interval not reached
        """
        if not self.is_active:
            return
        
        # Check if we should send an update
        if not force and (current - self._last_update) < self._interval:
            return
        
        progress = int((current / self._total) * 100) if self._total > 0 else 0
        await self._handler.notify(
            self._token,
            progress=progress,
            total=self._total,
            message=message,
        )
        self._last_update = current
    
    async def complete(self, message: Optional[str] = None) -> None:
        """Mark operation as complete."""
        if not self.is_active:
            return
        
        await self._handler.notify(
            self._token,
            progress=100,
            total=self._total,
            message=message or "Complete",
        )
        self._handler.clear_token(self._token)
