"""
Thought — Ada's fundamental memory primitive.

A Thought is a single concept encoded as an HDC vector with a natural
language summary.  Thoughts are stored persistently and recalled by
semantic similarity — Ada's lifelong memory starts here.

Encoding uses CharacterEncoder (positional char n-grams) so the vectors
are deterministic, handle misspellings, and require no training data.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glyphh.core.ops import bundle, cosine_similarity
from glyphh.linguistics.character import CharacterEncoder


# ── Thought dataclass ──────────────────────────────────────────────────────

@dataclass
class Thought:
    """A single memory unit — an HDC vector with natural language content.

    Attributes:
        id:         Unique identifier.
        content:    Natural language summary of the thought.
        vector:     HDC bipolar vector encoding of the content.
        strength:   Hebbian reinforcement strength [0, 1].  Grows on recall.
        created_at: Unix timestamp when the thought was first stored.
        recalled_at: Unix timestamp of last recall (or None).
        metadata:   Arbitrary key-value pairs (source, tags, etc.).
    """

    id: str
    content: str
    vector: np.ndarray
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    recalled_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, amount: float = 0.1) -> None:
        """Hebbian reinforcement — recalled thoughts grow stronger."""
        self.strength = min(1.0, self.strength + amount)
        self.recalled_at = time.time()

    def decay(self, rate: float = 0.01) -> None:
        """Temporal decay — unused thoughts fade over time."""
        self.strength = max(0.0, self.strength - rate)


# ── ThoughtEncoder ─────────────────────────────────────────────────────────

class ThoughtEncoder:
    """Encodes natural language into HDC vectors for thought storage and recall.

    Uses CharacterEncoder (layer 1 of the linguistics engine) to produce
    a bag-of-words bundle over the input text.  This is deterministic,
    handles typos and morphological variants, and needs no training data.

    The resulting vector captures *lexical* similarity — two thoughts that
    use similar words will be close in cosine space.  This is exactly what
    we want for recall-by-language.
    """

    def __init__(self, dimension: int = 10_000, seed: int = 42) -> None:
        self._char_encoder = CharacterEncoder(dimension=dimension, seed=seed)
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """Encode natural language text as a bipolar HDC vector."""
        return self._char_encoder.encode_text(text)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two thought vectors."""
        return float(cosine_similarity(a, b))

    def create_thought(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Thought:
        """Create a new Thought from natural language content."""
        return Thought(
            id=uuid.uuid4().hex[:12],
            content=content,
            vector=self.encode(content),
            metadata=metadata or {},
        )
