"""HDC Intent Cache — learns from confirmed classifications for fast repeat patterns.

.. deprecated::
    Replaced by GlyphSpace + QueryCache for new code. GlyphSpace uses the
    model's actual Glyph encoding (much richer than BoW) and leverages
    QueryCache's Hebbian reinforcement. This module is retained for backward
    compatibility but is no longer used by SchemaIntentClassifier.

Encodes query+state into HDC vectors, stores classifications, returns
cached results for similar queries. This is the mechanism by which HDC
"learns" from decisions:

  1. Classifier resolves "go to the archive" → navigate(location="archive")
  2. IntentCache encodes query + state → HDC vector, stores classification
  3. Next query "switch to the archive" → similar HDC vector → cache hit
  4. confirm(True) strengthens the entry (Hebbian)

Uses seed=113 (independent from other HDC subsystems).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol


_SEED = 113

# Hebbian bounds
_MIN_STRENGTH = 0.3
_MAX_STRENGTH = 3.0

# Default stop words for keyword extraction (lightweight, no file needed)
_STOP_WORDS = frozenset({
    "the", "a", "an", "to", "for", "and", "or", "in", "on", "at", "of",
    "is", "it", "do", "can", "please", "how", "what", "where", "when",
    "i", "me", "my", "this", "that", "with", "from", "by",
})


@dataclass
class CachedClassification:
    """A cached LLM classification with its HDC vector."""

    vector: np.ndarray
    classification: dict[str, Any]
    strength: float = 1.0


class IntentCache:
    """HDC-based cache of LLM intent classifications.

    Stores (query_vector, classification) pairs. On lookup, encodes
    the new query and finds the nearest cached vector by cosine
    similarity. If above threshold, returns the cached classification
    without calling the LLM.

    Hebbian reinforcement strengthens correct entries and weakens
    incorrect ones, so the cache self-corrects over time.
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: int = _SEED,
        threshold: float = 0.85,
        capacity: int = 500,
    ):
        self._dim = dimension
        self._seed = seed
        self._threshold = threshold
        self._capacity = capacity

        # Role vectors for encoding
        self._role_query = generate_symbol(seed, "role_query", dimension)
        self._role_state = generate_symbol(seed, "role_state", dimension)

        # Cache storage
        self._entries: list[CachedClassification] = []
        self._last_hit_index: int | None = None

    def encode(self, query: str, state_primary: str = "") -> np.ndarray:
        """Encode query + state context into an HDC vector.

        Uses BoW (bag-of-words) encoding of query words bound with
        state context. Similar queries with similar state produce
        similar vectors.
        """
        # Extract and encode query words
        words = self._extract_words(query)
        if not words:
            return generate_symbol(self._seed, "empty_query", self._dim)

        word_vecs = [
            generate_symbol(self._seed, f"w_{w}", self._dim)
            for w in words[:15]
        ]
        query_vec = bundle(word_vecs) if len(word_vecs) > 1 else word_vecs[0]
        query_bound = bind(self._role_query, query_vec)

        if state_primary:
            state_vec = generate_symbol(self._seed, f"st_{state_primary}", self._dim)
            state_bound = bind(self._role_state, state_vec)
            return bundle([query_bound, state_bound])

        return query_bound

    def lookup(self, query: str, state_primary: str = "") -> dict[str, Any] | None:
        """Find a cached classification for a similar query.

        Returns the classification dict if a match is found above threshold,
        or None if no match. Sets _last_hit_index for reinforce().
        """
        if not self._entries:
            self._last_hit_index = None
            return None

        query_vec = self.encode(query, state_primary)

        best_sim = -1.0
        best_idx = -1

        for i, entry in enumerate(self._entries):
            sim = cosine_similarity(query_vec, entry.vector)
            # Weight by strength (Hebbian)
            weighted_sim = sim * min(entry.strength, 1.0)
            if weighted_sim > best_sim:
                best_sim = weighted_sim
                best_idx = i

        if best_sim >= self._threshold and best_idx >= 0:
            self._last_hit_index = best_idx
            result = dict(self._entries[best_idx].classification)
            result["cache_similarity"] = round(best_sim, 4)
            return result

        self._last_hit_index = None
        return None

    def store(
        self,
        query: str,
        state_primary: str,
        classification: dict[str, Any],
    ) -> None:
        """Store an LLM classification for future reuse."""
        vec = self.encode(query, state_primary)

        # Check if we already have a very similar entry
        for entry in self._entries:
            sim = cosine_similarity(vec, entry.vector)
            if sim > 0.95:
                # Update existing entry instead of adding duplicate
                entry.classification = classification
                entry.strength = min(_MAX_STRENGTH, entry.strength + 0.1)
                return

        # Evict weakest if at capacity
        if len(self._entries) >= self._capacity:
            weakest_idx = min(
                range(len(self._entries)),
                key=lambda i: self._entries[i].strength,
            )
            self._entries.pop(weakest_idx)
            # Adjust last_hit_index if needed
            if self._last_hit_index is not None:
                if self._last_hit_index == weakest_idx:
                    self._last_hit_index = None
                elif self._last_hit_index > weakest_idx:
                    self._last_hit_index -= 1

        self._entries.append(CachedClassification(
            vector=vec,
            classification=classification,
            strength=1.0,
        ))

    def reinforce(self, correct: bool) -> None:
        """Hebbian reinforcement on the most recently looked-up entry.

        Strengthens correct classifications, weakens incorrect ones.
        Entries that decay below _MIN_STRENGTH are evicted.
        """
        if self._last_hit_index is None:
            return
        if self._last_hit_index >= len(self._entries):
            self._last_hit_index = None
            return

        entry = self._entries[self._last_hit_index]
        if correct:
            entry.strength = min(_MAX_STRENGTH, entry.strength + 0.2)
        else:
            entry.strength = max(0.0, entry.strength - 0.3)
            if entry.strength < _MIN_STRENGTH:
                self._entries.pop(self._last_hit_index)
                self._last_hit_index = None

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._last_hit_index = None

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._entries)

    def _extract_words(self, query: str) -> list[str]:
        """Extract meaningful words from query (stop words removed)."""
        cleaned = re.sub(r"[^\w\s]", "", query.lower())
        return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]
