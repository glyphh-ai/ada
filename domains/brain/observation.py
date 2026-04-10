"""
Observation log — every think() call gets recorded for the dream loop.

The dream loop consumes observations to mine patterns, reinforce routes,
and crystallize new capabilities.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Observation:
    """A single think() request and its outcome."""
    input: str
    capability: Optional[str]  # Which capability handled it (None = LLM fallback)
    confidence: float  # Routing confidence (0.0 = no match, 1.0 = perfect)
    result: Optional[str] = None  # Summary of the response
    llm_fallback: bool = False  # True if Ada needed Haiku for routing
    timestamp: float = field(default_factory=time.time)


class ObservationLog:
    """Ring buffer of recent observations for dream loop consumption."""

    def __init__(self, maxlen: int = 1000):
        self._buffer: deque[Observation] = deque(maxlen=maxlen)

    def record(self, obs: Observation) -> None:
        self._buffer.append(obs)

    def drain(self, limit: int = 100) -> list[Observation]:
        """Return up to `limit` recent observations without removing them."""
        return list(self._buffer)[-limit:]

    @property
    def count(self) -> int:
        return len(self._buffer)

    @property
    def llm_fallback_count(self) -> int:
        return sum(1 for o in self._buffer if o.llm_fallback)
