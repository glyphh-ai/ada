"""
Glyphh Operations Metering — tracks encoding calls per org per month.

A "glyphh operation" is any MCP tool invocation that flows through
ToolHandler.call_tool() (nl_query, gql_query, model-specific tools, etc.).

Usage is persisted to ~/.glyphh/usage.json and keyed by org_id + month.
The meter is intentionally file-based so it works for local runtimes
without requiring a database or platform connectivity.

Soft enforcement: the meter logs warnings but never blocks requests.
Hard enforcement (shutdown) will be handled by the Platform later.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

USAGE_FILE = Path.home() / ".glyphh" / "usage.json"


def _current_month_key() -> str:
    """Return the current month key, e.g. '2026-04'."""
    return datetime.utcnow().strftime("%Y-%m")


class OperationsMeter:
    """Thread-safe in-memory + file-backed operation counter."""

    def __init__(self, usage_file: Path = USAGE_FILE):
        self._file = usage_file
        self._lock = threading.Lock()
        self._counts: dict[str, dict[str, int]] = {}  # {month: {org_id: count}}
        self._dirty = False
        self._load()

    def _load(self):
        """Load usage from disk."""
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text())
                if isinstance(data, dict):
                    self._counts = data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load usage file: {e}")

    def _flush(self):
        """Persist current counts to disk."""
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(self._counts, indent=2))
            self._dirty = False
        except OSError as e:
            logger.warning(f"Could not write usage file: {e}")

    def record(self, org_id: str) -> int:
        """Record one operation for the given org. Returns new total for the month."""
        month = _current_month_key()
        with self._lock:
            if month not in self._counts:
                self._counts[month] = {}
            current = self._counts[month].get(org_id, 0) + 1
            self._counts[month][org_id] = current
            self._dirty = True
            # Flush every 10 operations to avoid excessive IO
            if current % 10 == 0:
                self._flush()
            return current

    def get_usage(self, org_id: str, month: Optional[str] = None) -> int:
        """Get current month's operation count for an org."""
        month = month or _current_month_key()
        with self._lock:
            return self._counts.get(month, {}).get(org_id, 0)

    def get_all_usage(self, month: Optional[str] = None) -> dict[str, int]:
        """Get all orgs' usage for a month."""
        month = month or _current_month_key()
        with self._lock:
            return dict(self._counts.get(month, {}))

    def flush(self):
        """Force-persist counts to disk."""
        with self._lock:
            if self._dirty:
                self._flush()


# Module-level singleton
_meter: Optional[OperationsMeter] = None


def get_meter() -> OperationsMeter:
    """Get the global operations meter (lazy-initialized)."""
    global _meter
    if _meter is None:
        _meter = OperationsMeter()
    return _meter
