"""
Gradient Patterns — learned paths through Ada's thought space.

When the cognitive gradient successfully resolves a query, the path
it took becomes a pattern. The pattern encodes:
  - The query vector (start)
  - The answer vector (end)
  - The intermediate residual steps
  - The success signal (did the user confirm, did it converge)

Over time, these patterns create shortcuts. Instead of walking the
full gradient, Ada recognizes "I've seen this kind of query before"
and jumps directly to the answer region.

This is how Ada removes hallucinations from the LLM:
  1. Gradient search finds facts (or doesn't)
  2. If converged → confident answer from FACTS, not LLM imagination
  3. If not converged → "I don't know" (no hallucination)
  4. Successful paths get encoded as patterns
  5. Patterns get Hebbian reinforcement on reuse
  6. Strong patterns become reflexes — sub-ms, no gradient walk needed

The gradient pattern is the error signal:
  - Correct answer → reinforce path → future queries shortcut
  - Wrong answer → weaken path → force full gradient walk
  - Unknown → no path created → "I don't know"

This is deterministic gradient descent with learned momentum.
"""

from __future__ import annotations

import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from glyphh.core.ops import cosine_similarity

logger = logging.getLogger(__name__)

# Pattern match threshold — above this, use the shortcut
PATTERN_MATCH_THRESHOLD = 0.75

# Minimum strength to use a pattern (weak patterns = do full gradient)
MIN_PATTERN_STRENGTH = 0.3


@dataclass
class GradientPattern:
    """A learned path through thought space."""
    pattern_id: str
    query_vector: np.ndarray         # where the search started
    answer_content: str              # what was found
    answer_speaker: str              # who said it
    answer_vector: Optional[np.ndarray] = None  # answer's content vector
    residual_steps: int = 0          # how many gradient steps it took
    strength: float = 1.0            # Hebbian — grows on reuse, decays on failure
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    success_count: int = 0           # confirmed correct
    failure_count: int = 0           # confirmed wrong

    def reinforce(self, amount: float = 0.2) -> None:
        """Strengthen this pattern — the path worked."""
        self.strength = min(5.0, self.strength + amount / (1.0 + 0.1 * self.strength))
        self.last_used = time.time()
        self.use_count += 1
        self.success_count += 1

    def weaken(self, amount: float = 0.5) -> None:
        """Weaken this pattern — the path was wrong."""
        self.strength = max(0.0, self.strength - amount)
        self.failure_count += 1

    def decay(self, factor: float = 0.98) -> None:
        """Time-based decay — unused patterns fade."""
        self.strength *= factor

    @property
    def confidence(self) -> float:
        """How much to trust this pattern."""
        if self.use_count == 0:
            return 0.5
        success_rate = self.success_count / self.use_count
        return self.strength * success_rate


class GradientPatternLibrary:
    """Library of learned gradient paths.

    Stores patterns and matches new queries against them.
    If a strong pattern matches, returns the shortcut answer
    without walking the full gradient.
    """

    def __init__(self):
        self._patterns: dict[str, GradientPattern] = {}

    def match(self, query_vector: np.ndarray) -> Optional[GradientPattern]:
        """Find a matching pattern for this query vector.

        Returns the strongest matching pattern above threshold,
        or None if no shortcut exists (must do full gradient walk).
        """
        if not self._patterns:
            return None

        best: Optional[GradientPattern] = None
        best_score = 0.0

        for pattern in self._patterns.values():
            if pattern.strength < MIN_PATTERN_STRENGTH:
                continue

            sim = float(cosine_similarity(query_vector, pattern.query_vector))
            if sim < PATTERN_MATCH_THRESHOLD:
                continue

            # Score = similarity × pattern confidence
            score = sim * pattern.confidence
            if score > best_score:
                best_score = score
                best = pattern

        return best

    def record(
        self,
        query_vector: np.ndarray,
        answer_content: str,
        answer_speaker: str,
        answer_vector: Optional[np.ndarray] = None,
        residual_steps: int = 0,
    ) -> GradientPattern:
        """Record a successful gradient path as a pattern.

        Called when the gradient converges to an answer.
        """
        # Generate deterministic ID from query vector
        vec_bytes = query_vector.tobytes() if isinstance(query_vector, np.ndarray) else b""
        pattern_id = hashlib.md5(vec_bytes + answer_content.encode()).hexdigest()[:12]

        # Check if pattern already exists
        if pattern_id in self._patterns:
            existing = self._patterns[pattern_id]
            existing.reinforce()
            return existing

        pattern = GradientPattern(
            pattern_id=pattern_id,
            query_vector=query_vector,
            answer_content=answer_content,
            answer_speaker=answer_speaker,
            answer_vector=answer_vector,
            residual_steps=residual_steps,
        )

        self._patterns[pattern_id] = pattern
        logger.debug(f"Recorded gradient pattern: {pattern_id} → {answer_content[:50]}")
        return pattern

    def reinforce(self, pattern_id: str) -> None:
        """The answer was correct — strengthen the path."""
        if pattern_id in self._patterns:
            self._patterns[pattern_id].reinforce()

    def weaken(self, pattern_id: str) -> None:
        """The answer was wrong — weaken the path."""
        if pattern_id in self._patterns:
            self._patterns[pattern_id].weaken()

    def decay_all(self, factor: float = 0.98) -> int:
        """Decay all patterns. Called by dream loop. Returns pruned count."""
        pruned = 0
        to_remove = []
        for pid, pattern in self._patterns.items():
            pattern.decay(factor)
            if pattern.strength < 0.01:
                to_remove.append(pid)
                pruned += 1
        for pid in to_remove:
            del self._patterns[pid]
        return pruned

    @property
    def count(self) -> int:
        return len(self._patterns)

    @property
    def strong_patterns(self) -> int:
        return sum(1 for p in self._patterns.values() if p.strength >= 1.0)
