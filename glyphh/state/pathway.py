"""
Pathway encoding and library for glyphh.state.

A Pathway is a named sequence of actions encoded as a single HDC vector via
position-binding and weighted bundling with decay.  The PathwayLibrary stores
known patterns and applies Hebbian strengthening whenever a pattern is
confirmed correct — the more a pathway fires, the stronger it becomes.

This mirrors the brain's mechanism for cementing neural pathways:
  bind(position, action)   →  ordered association  (position encodes step order)
  weighted_bundle(history) →  superposition with decay  (recent steps dominate)
  cosine_similarity(state, pattern) →  "am I in this pathway?"
  strengthen(pattern)     →  Hebbian: fire together, wire together
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from glyphh.core.ops import bind, cosine_similarity, generate_symbol


# ── Weighted bundle (not in ops.py — implemented here for state module) ──

def _weighted_bundle(pairs: list[tuple[np.ndarray, float]], dimension: int) -> np.ndarray:
    """Weighted majority vote of bipolar vectors.

    Each vector contributes proportional to its weight.  The result is
    binarised (sign) to stay bipolar.

    Args:
        pairs: List of (vector, weight) where vector is bipolar int8.
        dimension: Dimension of vectors (used for zero-init).

    Returns:
        Bipolar int8 array of shape (dimension,).
    """
    total = np.zeros(dimension, dtype=np.float32)
    for vec, weight in pairs:
        total += vec.astype(np.float32) * weight
    return np.where(total >= 0, 1, -1).astype(np.int8)


# ── Pathway ──

@dataclass
class Pathway:
    """A named action-sequence pattern analogous to a neural pathway.

    The vector encodes the full sequence (via PathwayEncoder) as a single
    point in HD space.  strength increases with each confirmed use —
    the Hebbian "fire → wire" mechanism.
    """

    name: str
    vector: np.ndarray          # HDC encoding of the sequence
    strength: float = 1.0       # grows with Hebbian reinforcement
    fire_count: int = 0         # how many times this pathway was confirmed

    def strengthen(self, amount: float = 0.15) -> None:
        """Hebbian reinforcement.

        Uses diminishing returns so strength asymptotes rather than diverges:
          Δstrength = amount / (1 + 0.1 * fire_count)
        """
        self.fire_count += 1
        delta = amount / (1.0 + 0.1 * self.fire_count)
        self.strength = min(3.0, self.strength + delta)


# ── PathwayEncoder ──

class PathwayEncoder:
    """Encodes a sequence of action vectors into a single pathway vector.

    Mechanism:
      1. Each action vector is bound to a position symbol — this encodes
         step order into the vector without destroying the action signal.
      2. All bound steps are combined via weighted bundle with exponential
         decay — recent steps dominate (working memory effect).

    This is the HDC equivalent of a recurrent neural pathway:
      - Binding preserves order information
      - Weighted bundling creates a superposition of the whole trajectory
      - Decay means recent context outweighs distant history

    Args:
        dimension:  Vector dimension.  Must match the vectors you pass in.
        seed:       Seed for position symbol generation.  Keep consistent
                    across a ConversationState instance.
        decay:      Exponential decay factor [0, 1].  0.75 means each
                    step is 25% less influential than the step after it.
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: int = 42,
        decay: float = 0.75,
    ) -> None:
        self._dimension = dimension
        self._seed = seed
        self._decay = decay
        self._history: list[np.ndarray] = []   # bound (position ⊗ action) vectors

    def update(self, action_vector: np.ndarray) -> None:
        """Add a new action to the pathway.

        Args:
            action_vector: Bipolar int8 vector representing the action taken.
        """
        step_idx = len(self._history)
        pos = generate_symbol(self._seed, f"state_pos_{step_idx}", self._dimension)
        self._history.append(bind(pos, action_vector))

    def get_state(self) -> Optional[np.ndarray]:
        """Return the current pathway vector (decayed weighted bundle).

        Returns None when no actions have been recorded yet.
        """
        if not self._history:
            return None
        n = len(self._history)
        pairs = [
            (vec, self._decay ** (n - i - 1))
            for i, vec in enumerate(self._history)
        ]
        return _weighted_bundle(pairs, self._dimension)

    def reset(self) -> None:
        """Clear all recorded history (start of a new conversation)."""
        self._history.clear()

    @property
    def depth(self) -> int:
        """Number of action steps currently encoded."""
        return len(self._history)


# ── PathwayLibrary ──

class PathwayLibrary:
    """Library of named pathway patterns with Hebbian strengthening.

    Pre-seeded patterns act as long-term memory — starting at strength 1.0
    and growing with each confirmed use.  New patterns (learned at runtime)
    start weak and cement through repeated confirmation.

    Args:
        dimension:  Vector dimension.
        seed:       Seed for position symbol generation — must match the
                    PathwayEncoder used to create pattern vectors.
        decay:      Decay factor used when encoding library patterns.
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: int = 42,
        decay: float = 0.75,
    ) -> None:
        self._dimension = dimension
        self._seed = seed
        self._decay = decay
        self._pathways: dict[str, Pathway] = {}

    # ── Pattern management ──

    def add(
        self,
        name: str,
        action_vectors: list[np.ndarray],
        strength: float = 1.0,
    ) -> None:
        """Encode an action sequence and store it as a named pattern.

        Args:
            name:           Unique pattern name, e.g. "setup_then_execute".
            action_vectors: Ordered list of bipolar action vectors.
            strength:       Initial strength (default 1.0).
        """
        encoder = PathwayEncoder(self._dimension, self._seed, self._decay)
        for v in action_vectors:
            encoder.update(v)
        vec = encoder.get_state()
        if vec is not None:
            self._pathways[name] = Pathway(
                name=name,
                vector=vec,
                strength=strength,
            )

    def __len__(self) -> int:
        return len(self._pathways)

    def __contains__(self, name: str) -> bool:
        return name in self._pathways

    # ── Matching ──

    def match(
        self,
        state_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[Pathway, float]]:
        """Find pathways most similar to the current state vector.

        Similarity is scaled by pathway strength (Hebbian: stronger pathways
        are more influential matches).

        Args:
            state_vector:  Current pathway vector from PathwayEncoder.
            top_k:         Maximum number of results to return.

        Returns:
            List of (Pathway, weighted_score) sorted descending.
        """
        results = []
        for pathway in self._pathways.values():
            raw_sim = float(cosine_similarity(state_vector, pathway.vector))
            weighted = raw_sim * pathway.strength
            results.append((pathway, weighted))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Hebbian reinforcement ──

    def strengthen(self, name: str, amount: float = 0.15) -> None:
        """Hebbian update: fire → wire.

        Call after a pathway was confirmed correct (ground truth matched).

        Args:
            name:    Pattern name to strengthen.
            amount:  Base amount to increase strength by (subject to diminishing
                     returns based on fire_count).
        """
        if name in self._pathways:
            self._pathways[name].strengthen(amount)

    # ── Score helpers ──

    def get_continuation_boost(
        self,
        candidate_vector: np.ndarray,
        state_vector: np.ndarray,
        top_k: int = 3,
    ) -> float:
        """Score how much a candidate action is supported by active patterns.

        If the current state matches library patterns, and the candidate is
        aligned with those patterns, return a positive boost.  This is the
        "pathway activation" signal — the pattern is firing, so related
        actions are primed.

        Returns a float in [0, 0.4] — capped to avoid overwhelming the
        query-semantic signal from Model A.
        """
        if not self._pathways:
            return 0.0

        matches = self.match(state_vector, top_k=top_k)
        total = 0.0
        for pathway, match_score in matches:
            if match_score > 0.1:
                cand_sim = float(cosine_similarity(candidate_vector, pathway.vector))
                total += cand_sim * match_score * 0.5

        return min(0.4, total)

    @property
    def pathways(self) -> dict[str, Pathway]:
        return dict(self._pathways)
