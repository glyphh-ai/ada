"""
ConversationState — first-class conversation state for the Glyphh SDK.

Tracks the trajectory of a multi-turn conversation as an HDC pathway vector,
enabling next-action prediction and automatic strengthening of frequently-used
patterns through Hebbian reinforcement.

The core insight: in the human brain, state is not a snapshot — it is the
currently active superposition of neural pathways that fired to get here.
ConversationState encodes exactly this: the decaying weighted superposition of
all actions taken so far, represented as a single point in HD space.

Two questions the state answers for every turn:
  1. "Where am I?" — the pathway vector encodes the full trajectory
  2. "What should come next?" — cosine similarity to candidates + library patterns

Usage:
    from glyphh.state import ConversationState

    state = ConversationState(dimension=10000, seed=42, decay=0.75)

    # Pre-seed known patterns (optional — acts as prior knowledge)
    state.add_pathway("navigate_then_operate", action_glyphs=[cd_glyph, mv_glyph])
    state.add_pathway("navigate_then_search",  action_glyphs=[cd_glyph, grep_glyph])

    # Each turn: update then predict
    state.update(action_glyphs=[cd_glyph, mv_glyph])
    scores = state.predict_next(
        query_glyph=query_glyph,
        candidates={"cd": cd_glyph, "mv": mv_glyph, "grep": grep_glyph},
    )
    # → {"mv": 0.78, "grep": 0.59, "cd": 0.22}  (sorted descending)

    # After ground truth is known — Hebbian reinforcement
    state.confirm(confirmed_glyphs=[mv_glyph])
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Glyph
from glyphh.state.pathway import PathwayEncoder, PathwayLibrary


class ConversationState:
    """HDC-based conversation state tracker.

    Encodes conversation history as a decaying pathway vector (positional
    binding + weighted bundle).  Three signals are blended to predict the
    next action:

      query_weight   (default 0.55)  — semantic similarity of query to candidate
      pathway_weight (default 0.30)  — how well candidate continues the current trajectory
      pattern_weight (default 0.15)  — how strongly library patterns support the candidate

    The library provides Hebbian memory: confirmed correct actions strengthen
    the patterns that were active at the time, cementing frequently-used
    pathways over many conversations.

    Args:
        dimension:       HD vector dimension (must match the Glyphs you pass in).
        seed:            Seed for position symbol generation.
        decay:           Exponential decay for working memory [0, 1].
                         0.75 → each step is 25% less influential than the next.
        query_weight:    Weight for query-to-candidate semantic signal.
        pathway_weight:  Weight for pathway-continuation signal.
        pattern_weight:  Weight for library pattern-activation signal.
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: int = 42,
        decay: float = 0.75,
        query_weight: float = 0.55,
        pathway_weight: float = 0.30,
        pattern_weight: float = 0.15,
    ) -> None:
        self._dimension = dimension
        self._encoder = PathwayEncoder(dimension=dimension, seed=seed, decay=decay)
        self._library = PathwayLibrary(dimension=dimension, seed=seed, decay=decay)
        self._query_weight = query_weight
        self._pathway_weight = pathway_weight
        self._pattern_weight = pattern_weight

        # Snapshot of library matches at last predict_next call — used by confirm()
        self._last_active_patterns: list[str] = []

    # ── Pathway update ──

    def update(self, action_glyphs: list[Glyph]) -> None:
        """Record the actions taken this turn.

        Each action's global cortex vector is bound to its position in the
        sequence and added to the rolling pathway superposition.

        Args:
            action_glyphs: Glyphs for each function called this turn.
        """
        for glyph in action_glyphs:
            self._encoder.update(glyph.global_cortex.data)

    def update_raw(self, action_vectors: list[np.ndarray]) -> None:
        """Record actions using raw bipolar vectors.

        Useful when Glyphs are not available — e.g. when only function names
        are known and a lightweight symbol is used to represent them.

        Args:
            action_vectors: List of bipolar int8 arrays.
        """
        for vec in action_vectors:
            self._encoder.update(vec)

    # ── Next-action prediction ──

    def predict_next(
        self,
        query_glyph: Glyph,
        candidates: dict[str, Glyph],
    ) -> dict[str, float]:
        """Score candidate actions for the next turn.

        Blends three independent HDC signals:

          1. Query alignment — cosine similarity between the current query
             and each candidate (Model A's signal, reused here).

          2. Pathway continuation — cosine similarity between the current
             pathway state vector and each candidate.  A candidate that
             frequently follows the current trajectory will score higher.

          3. Library pattern boost — if the current state matches a known
             pathway pattern (e.g. "navigate_then_operate"), candidates
             that continue that pattern receive additional weight.

        Args:
            query_glyph:  Glyph for the current turn's query.
            candidates:   {function_name: glyph} for all available actions.

        Returns:
            {function_name: score} sorted descending.  Scores are in [-1, 1]
            range (cosine-based blend).
        """
        if not candidates:
            return {}

        query_vector = query_glyph.global_cortex.data
        state_vector = self._encoder.get_state()

        # Snapshot active patterns for potential Hebbian confirmation
        if state_vector is not None:
            matches = self._library.match(state_vector, top_k=3)
            self._last_active_patterns = [
                p.name for p, score in matches if score > 0.1
            ]
        else:
            self._last_active_patterns = []

        scores: dict[str, float] = {}
        for name, glyph in candidates.items():
            cand_vector = glyph.global_cortex.data

            # Signal 1: query → candidate semantic similarity
            query_sim = float(cosine_similarity(query_vector, cand_vector))

            # Signal 2: pathway state → candidate continuation
            if state_vector is not None:
                path_sim = float(cosine_similarity(state_vector, cand_vector))
            else:
                path_sim = 0.0

            # Signal 3: library pattern activation boost
            if state_vector is not None:
                pattern_boost = self._library.get_continuation_boost(
                    cand_vector, state_vector
                )
            else:
                pattern_boost = 0.0

            scores[name] = (
                query_sim    * self._query_weight
                + path_sim   * self._pathway_weight
                + pattern_boost * self._pattern_weight
            )

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # ── Hebbian reinforcement ──

    def confirm(self, confirmed_glyphs: list[Glyph]) -> None:
        """Mark confirmed actions as correct (Hebbian: fire → wire).

        Strengthens library patterns that were active when predict_next was
        called.  Over many conversations, frequently-confirmed patterns
        accumulate strength, making them progressively more influential
        in future predictions.

        Should be called after ground truth is available — e.g. after eval
        scoring, or after an LLM confirms the action was correct.

        Args:
            confirmed_glyphs:  Glyphs for the actions that were correct.
                               (Used for future per-action tracking; currently
                               pattern-level Hebbian is applied.)
        """
        for pattern_name in self._last_active_patterns:
            self._library.strengthen(pattern_name)

    # ── Library management ──

    def add_pathway(
        self,
        name: str,
        action_glyphs: list[Glyph],
        strength: float = 1.0,
    ) -> None:
        """Pre-seed a known pathway pattern.

        Patterns act as long-term prior knowledge — the model starts knowing
        these patterns exist, and confirms/strengthens them through use.

        Args:
            name:          Unique pattern name, e.g. "navigate_then_operate".
            action_glyphs: Ordered Glyphs representing the pattern steps.
            strength:      Initial strength (default 1.0).
        """
        vectors = [g.global_cortex.data for g in action_glyphs]
        self._library.add(name, vectors, strength=strength)

    def add_pathway_raw(
        self,
        name: str,
        action_vectors: list[np.ndarray],
        strength: float = 1.0,
    ) -> None:
        """Pre-seed a known pathway using raw vectors.

        Args:
            name:            Pattern name.
            action_vectors:  Ordered bipolar int8 arrays.
            strength:        Initial strength.
        """
        self._library.add(name, action_vectors, strength=strength)

    # ── Introspection ──

    def active_pathways(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Return the library patterns most aligned with the current state.

        Args:
            top_k: Maximum number of patterns to return.

        Returns:
            List of (pattern_name, weighted_score) sorted descending.
            Empty list if no history recorded yet.
        """
        state_vector = self._encoder.get_state()
        if state_vector is None:
            return []
        matches = self._library.match(state_vector, top_k=top_k)
        return [(p.name, round(score, 4)) for p, score in matches]

    def get_state_vector(self) -> Optional[np.ndarray]:
        """Return the raw pathway state vector.

        Useful for external scoring — e.g. boosting Model A scores based on
        the pathway signal outside of predict_next.

        Returns:
            Bipolar int8 array, or None if no history yet.
        """
        return self._encoder.get_state()

    @property
    def depth(self) -> int:
        """Number of action steps encoded in the current pathway."""
        return self._encoder.depth

    @property
    def library_size(self) -> int:
        """Number of patterns in the library."""
        return len(self._library)

    def reset(self) -> None:
        """Reset the pathway encoder for a new conversation.

        The library is intentionally preserved — patterns are long-term memory
        that should persist across conversations and accumulate strength.
        """
        self._encoder.reset()
        self._last_active_patterns = []
