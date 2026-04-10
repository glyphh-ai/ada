"""
User Derivative — Ada learns who the user is by observing them.

Not a profile. A gradient. Each interaction shifts the derivative.
Patterns that repeat get reinforced. Patterns that don't repeat decay.
Just like a child learns from a parent — through exposure, not rules.

The derivative encodes:
  - Communication style (casual/formal, terse/verbose)
  - Topics the user cares about (family, work, tech)
  - How they ask questions (direct, roundabout)
  - What they correct (what matters to them)
  - Emotional patterns (stoic, expressive)

Stored as HDC vectors. Hebbian reinforcement on repetition.
Dream loop mines deeper patterns and crystallizes them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from glyphh.core.ops import cosine_similarity
from glyphh.memory.thought_space import ThoughtGlyphSpace

logger = logging.getLogger(__name__)


@dataclass
class InteractionSignal:
    """One observation about a user interaction."""
    dimension: str        # what aspect: "style", "topic", "emotion", "correction"
    vector: np.ndarray    # HDC vector encoding of the pattern
    strength: float = 1.0
    last_seen: float = field(default_factory=time.time)
    count: int = 1


class UserDerivative:
    """Learns the user's patterns from interactions.

    Each interaction is observed along multiple dimensions:
      - style: how they communicate
      - topic: what they talk about
      - emotion: how they feel
      - correction: what they care about getting right

    Patterns are encoded as HDC vectors and reinforced when
    similar patterns repeat. The derivative is the accumulated
    bundle of all reinforced patterns — Ada's model of who
    the user is.
    """

    # Hebbian parameters
    REINFORCE_AMOUNT = 0.15
    DECAY_FACTOR = 0.995
    STRENGTH_CAP = 5.0
    PRUNE_THRESHOLD = 0.05
    SIMILARITY_MERGE = 0.80  # patterns this similar get merged

    def __init__(self, thought_space: ThoughtGlyphSpace) -> None:
        self._encoder = thought_space.encoder
        self._signals: list[InteractionSignal] = []
        self._derivative_cache: Optional[dict[str, np.ndarray]] = None

    def observe(
        self,
        input_text: str,
        response_text: str,
        extracted_facts: list[str],
        extracted_question: Optional[str],
        extracted_emotion: Optional[str],
        is_correction: bool,
        recalled_facts: list[tuple],
    ) -> None:
        """Observe an interaction and update the derivative.

        Called after every think cycle. Encodes the interaction
        pattern and either reinforces an existing signal or
        creates a new one.
        """
        self._derivative_cache = None  # invalidate

        # ── Topic signal — what the user talks about
        if input_text:
            topic_vec = self._encode(input_text)
            if topic_vec is not None:
                self._record("topic", topic_vec)

        # ── Style signal — how they communicate
        # Encode the raw input (captures phrasing, length, formality)
        if input_text and len(input_text.split()) >= 2:
            style_vec = self._encode(input_text)
            if style_vec is not None:
                self._record("style", style_vec)

        # ── Emotion signal — how they feel
        if extracted_emotion:
            emotion_vec = self._encode(f"the user feels {extracted_emotion}")
            if emotion_vec is not None:
                self._record("emotion", emotion_vec)

        # ── Correction signal — what they care about
        if is_correction:
            correction_vec = self._encode(input_text)
            if correction_vec is not None:
                self._record("correction", correction_vec)

    def get_context(self, input_text: str, top_k: int = 3) -> list[str]:
        """Get relevant derivative context for response generation.

        Returns human-readable descriptions of the strongest patterns
        relevant to this input, for inclusion in LLM prompts.
        """
        if not self._signals:
            return []

        input_vec = self._encode(input_text)
        if input_vec is None:
            return []

        # Find signals most relevant to this input
        scored = []
        for signal in self._signals:
            sim = float(cosine_similarity(input_vec, signal.vector))
            if sim > 0.3:
                scored.append((signal, sim * signal.strength))

        scored.sort(key=lambda x: x[1], reverse=True)

        context = []
        for signal, score in scored[:top_k]:
            if signal.dimension == "topic" and signal.count >= 3:
                context.append(
                    f"The user frequently discusses topics related to this "
                    f"(seen {signal.count} times)"
                )
            elif signal.dimension == "emotion" and signal.count >= 2:
                context.append(
                    f"The user has expressed similar emotions before "
                    f"(seen {signal.count} times)"
                )
            elif signal.dimension == "correction":
                context.append(
                    "The user has corrected similar information before — "
                    "be precise here"
                )

        return context

    def style_summary(self) -> Optional[str]:
        """Summarize the user's communication style for LLM prompts.

        Returns a brief directive based on observed patterns, or None
        if not enough data yet.
        """
        style_signals = [s for s in self._signals if s.dimension == "style"]
        if len(style_signals) < 5:
            return None  # Not enough observations

        # Average word count of inputs (proxy for verbosity)
        strong = [s for s in style_signals if s.strength >= 1.0]
        if len(strong) < 3:
            return None

        return (
            f"The user has interacted {len(style_signals)} times. "
            f"Match their communication style."
        )

    def decay(self) -> int:
        """Decay all signals. Called by dream loop. Returns pruned count."""
        pruned = 0
        surviving = []
        for signal in self._signals:
            signal.strength *= self.DECAY_FACTOR
            if signal.strength >= self.PRUNE_THRESHOLD:
                surviving.append(signal)
            else:
                pruned += 1
        self._signals = surviving
        self._derivative_cache = None
        return pruned

    @property
    def signal_count(self) -> int:
        return len(self._signals)

    @property
    def strong_signals(self) -> int:
        return sum(1 for s in self._signals if s.strength >= 1.0)

    def stats(self) -> dict:
        """Stats for dream/debug output."""
        by_dim = {}
        for s in self._signals:
            by_dim.setdefault(s.dimension, []).append(s)
        return {
            "total_signals": len(self._signals),
            "strong_signals": self.strong_signals,
            "dimensions": {
                dim: len(sigs) for dim, sigs in by_dim.items()
            },
        }

    # ── Internal ─────────────────────────────────────────────

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text into an HDC vector via the thought space encoder."""
        try:
            glyph = self._encoder.encode_thought(text, speaker="incoming")
            vec = glyph.metadata.get("_content_vector")
            if vec is not None:
                return np.array(vec, dtype=np.float64)
        except Exception:
            pass
        return None

    def _record(self, dimension: str, vector: np.ndarray) -> None:
        """Record a signal — merge with existing if similar, else create new."""
        # Find most similar existing signal in same dimension
        best_match = None
        best_sim = 0.0
        for signal in self._signals:
            if signal.dimension != dimension:
                continue
            sim = float(cosine_similarity(vector, signal.vector))
            if sim > best_sim:
                best_sim = sim
                best_match = signal

        if best_match and best_sim >= self.SIMILARITY_MERGE:
            # Reinforce existing pattern
            best_match.strength = min(
                self.STRENGTH_CAP,
                best_match.strength + self.REINFORCE_AMOUNT,
            )
            best_match.count += 1
            best_match.last_seen = time.time()
            # Blend vectors slightly toward new observation
            alpha = 0.1
            best_match.vector = (
                (1 - alpha) * best_match.vector + alpha * vector
            )
        else:
            # New pattern
            self._signals.append(InteractionSignal(
                dimension=dimension,
                vector=vector,
            ))
