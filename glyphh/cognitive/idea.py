"""
IdeaEncoder + IdeaSpace — episodic memory for the cognitive loop.

An "idea" is a composed HDC glyph that encodes an entire situation:
  action + target + state + keywords → single bipolar vector

The IdeaSpace stores idea→outcome pairs with Hebbian reinforcement
and temporal decay. Ideas that fire correctly strengthen; unused
ideas fade. This is the episodic memory that the cognitive loop
reads from and writes to.

Uses seed=101 (independent vector space from other models).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol


_DIM = 10000
_SEED = 101

# Hebbian bounds
_MIN_STRENGTH = 0.3
_MAX_STRENGTH = 3.0

# Temporal decay per tick (turn)
_DECAY_RATE = 0.97


def _weighted_bundle(pairs: list[tuple[np.ndarray, float]], dimension: int) -> np.ndarray:
    """Weighted majority-vote of bipolar vectors."""
    total = np.zeros(dimension, dtype=np.float32)
    for vec, weight in pairs:
        total += vec.astype(np.float32) * weight
    return np.where(total >= 0, 1, -1).astype(np.int8)


class IdeaEncoder:
    """Compose a situation into a single HDC vector.

    Roles:
      action   — what the user wants to do (canonical verb)
      target   — what they're acting on (canonical noun)
      state    — where they are (current context, session state, etc.)
      keywords — BoW of query words (captures phrasing nuance)
      context  — recent action history (what just happened)
    """

    def __init__(self, dimension: int = _DIM, seed: int = _SEED):
        self._dim = dimension
        self._seed = seed
        self._role_action = generate_symbol(seed, "role_action", dimension)
        self._role_target = generate_symbol(seed, "role_target", dimension)
        self._role_state = generate_symbol(seed, "role_state", dimension)
        self._role_keywords = generate_symbol(seed, "role_keywords", dimension)
        self._role_context = generate_symbol(seed, "role_context", dimension)

    def encode(
        self,
        action: str = "",
        target: str = "",
        state: str = "",
        keywords: list[str] | None = None,
        context: list[str] | None = None,
    ) -> np.ndarray:
        """Encode a situation as a composed idea-glyph.

        Each signal is bound to its role, then all are bundled.
        Keywords use BoW so different phrasings of the same idea
        converge via shared sub-words.
        """
        components = []

        if action:
            act_vec = generate_symbol(self._seed, f"act_{action}", self._dim)
            components.append(bind(self._role_action, act_vec))

        if target:
            tgt_vec = generate_symbol(self._seed, f"tgt_{target}", self._dim)
            components.append(bind(self._role_target, tgt_vec))

        if state:
            st_vec = generate_symbol(self._seed, f"st_{state}", self._dim)
            components.append(bind(self._role_state, st_vec))

        if keywords:
            kw_vecs = [
                generate_symbol(self._seed, f"kw_{w}", self._dim)
                for w in keywords[:12]
            ]
            if kw_vecs:
                kw_bundle = bundle(kw_vecs) if len(kw_vecs) > 1 else kw_vecs[0]
                components.append(bind(self._role_keywords, kw_bundle))

        if context:
            ctx_vecs = [
                generate_symbol(self._seed, f"ctx_{c}", self._dim)
                for c in context[:5]
            ]
            if ctx_vecs:
                ctx_bundle = bundle(ctx_vecs) if len(ctx_vecs) > 1 else ctx_vecs[0]
                components.append(bind(self._role_context, ctx_bundle))

        if not components:
            # Empty idea — return a deterministic fallback
            return generate_symbol(self._seed, "empty_idea", self._dim)

        return bundle(components) if len(components) > 1 else components[0]

    def encode_from_query(
        self,
        query: str,
        intent: dict,
        state: str = "",
        recent_actions: list[str] | None = None,
    ) -> np.ndarray:
        """Convenience: encode from a raw query + intent extraction result.

        Args:
            query: Raw user query text
            intent: Intent extraction result dict —
                    {action, target, domain, keywords}
            state: Current state label (e.g. current context)
            recent_actions: Last N function names called
        """
        # Extract keywords from query (strip stop words, lowercase)
        keywords_str = intent.get("keywords", "")
        keywords = keywords_str.split() if keywords_str else []

        # Add extra words from query not captured by intent
        query_words = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split())
        for w in query_words:
            if len(w) > 2 and w not in keywords:
                keywords.append(w)

        return self.encode(
            action=intent.get("action", ""),
            target=intent.get("target", ""),
            state=state,
            keywords=keywords[:12],
            context=recent_actions,
        )


@dataclass
class Idea:
    """An episode in the idea space: situation → outcome."""

    vector: np.ndarray
    outcome: list[dict]   # [{func_name: {args}}, ...]
    strength: float = 1.0
    fire_count: int = 0
    last_fired: int = 0   # Turn number when last recalled
    label: str = ""       # Optional human-readable label

    def strengthen(self, amount: float = 0.15) -> None:
        """Hebbian: fire → wire. Diminishing returns."""
        delta = amount / (1.0 + 0.1 * self.fire_count)
        self.strength = min(_MAX_STRENGTH, self.strength + delta)
        self.fire_count += 1

    def weaken(self, factor: float = 0.85) -> None:
        """Weaken on incorrect recall."""
        self.strength = max(_MIN_STRENGTH, self.strength * factor)


class IdeaSpace:
    """Episodic memory with Hebbian reinforcement and temporal decay.

    Stores idea→outcome pairs. Recall finds similar ideas weighted
    by strength and recency. Ideas that fire correctly strengthen;
    unused ideas decay over time.
    """

    def __init__(
        self,
        dimension: int = _DIM,
        seed: int = _SEED,
        decay_rate: float = _DECAY_RATE,
    ):
        self._dim = dimension
        self._seed = seed
        self._decay_rate = decay_rate
        self._encoder = IdeaEncoder(dimension, seed)
        self._ideas: list[Idea] = []
        self._turn: int = 0
        self._last_recalled: Idea | None = None

    @property
    def encoder(self) -> IdeaEncoder:
        return self._encoder

    @property
    def size(self) -> int:
        return len(self._ideas)

    @property
    def turn(self) -> int:
        return self._turn

    def store(
        self,
        idea_vector: np.ndarray,
        outcome: list[dict],
        strength: float = 1.0,
        label: str = "",
    ) -> Idea:
        """Store an idea→outcome episode."""
        idea = Idea(
            vector=idea_vector,
            outcome=outcome,
            strength=strength,
            last_fired=self._turn,
            label=label,
        )
        self._ideas.append(idea)
        return idea

    def recall(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.25,
    ) -> list[tuple[Idea, float]]:
        """Find similar past ideas, weighted by strength and recency.

        Returns [(Idea, weighted_score)] sorted descending.
        Score = cosine_similarity * strength * temporal_factor.
        """
        if not self._ideas:
            return []

        results = []
        for idea in self._ideas:
            raw_sim = cosine_similarity(query_vector, idea.vector)
            if raw_sim < min_similarity:
                continue

            # Temporal factor: recent ideas score higher
            age = self._turn - idea.last_fired
            temporal = self._decay_rate ** age

            weighted = raw_sim * idea.strength * temporal
            results.append((idea, weighted))

        results.sort(key=lambda x: x[1], reverse=True)

        # Track what we recalled for later reinforcement
        if results:
            self._last_recalled = results[0][0]

        return results[:top_k]

    def reinforce(self, idea: Idea, was_correct: bool) -> None:
        """Hebbian reinforcement on a specific idea."""
        if was_correct:
            idea.strengthen()
            idea.last_fired = self._turn
        else:
            idea.weaken()

    def reinforce_last(self, was_correct: bool) -> None:
        """Reinforce the most recently recalled idea."""
        if self._last_recalled is not None:
            self.reinforce(self._last_recalled, was_correct)

    def tick(self) -> None:
        """Advance turn counter."""
        self._turn += 1

    def reset(self) -> None:
        """Clear working state. Ideas (long-term memory) persist."""
        self._turn = 0
        self._last_recalled = None
        # Reset strengths but keep ideas
        for idea in self._ideas:
            idea.strength = 1.0
            idea.fire_count = 0
            idea.last_fired = 0

    def clear(self) -> None:
        """Full reset — clear all ideas."""
        self._ideas.clear()
        self._turn = 0
        self._last_recalled = None

    def seed_from_episodes(
        self,
        episodes: list[dict],
        state: str = "",
    ) -> int:
        """Seed the idea space from pre-built episodes.

        Each episode: {"query": str, "intent": dict, "outcome": list[dict]}
        Returns number of ideas seeded.
        """
        count = 0
        for ep in episodes:
            intent = ep.get("intent", {})
            query = ep.get("query", "")
            outcome = ep.get("outcome", [])
            if not outcome:
                continue

            vec = self._encoder.encode_from_query(
                query=query,
                intent=intent,
                state=ep.get("state", state),
                recent_actions=ep.get("recent_actions"),
            )
            self.store(vec, outcome, strength=1.0, label=ep.get("label", ""))
            count += 1

        return count
