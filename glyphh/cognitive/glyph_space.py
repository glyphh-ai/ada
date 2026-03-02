"""GlyphSpace — unified glyph storage + scoring + caching for the cognitive loop.

Bridges the GQL infrastructure (InMemoryGlyphStorage, QueryCache) with the
cognitive loop's routing pipeline. Function Glyphs from a ModelScorer are
stored in InMemoryGlyphStorage; queries are scored via a pluggable
ScoringStrategy; results are cached in QueryCache with Hebbian reinforcement.

Usage:
    strategy = BFCLScoringStrategy()   # model-specific
    space = GlyphSpace(scoring_strategy=strategy)
    space.configure(scorer.get_func_glyphs())

    result = space.find_similar(query_glyph)
    space.reinforce(correct=True)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from glyphh.core import Vector
from glyphh.core.ops import cosine_similarity
from glyphh.gql.cache import QueryCache
from glyphh.gql.storage import InMemoryGlyphStorage

from .model_scorer import ScorerResult

logger = logging.getLogger(__name__)


# ── Scoring strategy protocol ──────────────────────────────────────────

@runtime_checkable
class ScoringStrategy(Protocol):
    """Model-specific Glyph similarity scoring.

    Models implement this to customize how query Glyphs are scored
    against function Glyphs. For example, BFCL uses 4-level hierarchical
    similarity (cortex 5%, layer 10%, segment 25%, role 60%).
    """

    def score_pair(self, query_glyph: Any, target_glyph: Any) -> float:
        """Score similarity between a query Glyph and a target Glyph.

        Args:
            query_glyph: The encoded query (SDK Glyph)
            target_glyph: A stored function Glyph (SDK Glyph)

        Returns:
            Similarity score in [0.0, 1.0]
        """
        ...


class DefaultScoringStrategy:
    """Cosine similarity on global cortex vectors.

    Suitable for models that don't need hierarchical scoring.
    """

    def score_pair(self, query_glyph: Any, target_glyph: Any) -> float:
        q_cortex = getattr(query_glyph, "global_cortex", None)
        t_cortex = getattr(target_glyph, "global_cortex", None)
        if q_cortex is None or t_cortex is None:
            return 0.0
        return float(cosine_similarity(q_cortex.data, t_cortex.data))


# ── Confidence thresholds ──────────────────────────────────────────────

_HIGH_CONFIDENCE = 0.50
_IRRELEVANCE_THRESHOLD = 0.15


# ── GlyphSpace ─────────────────────────────────────────────────────────

def _vector_similarity(v1: Vector, v2: Vector) -> float:
    """Cosine similarity between two Vector objects (for QueryCache)."""
    return float(cosine_similarity(v1.data, v2.data))


class GlyphSpace:
    """Unified glyph space for cognitive loop routing.

    Stores function Glyphs in InMemoryGlyphStorage (GQL layer).
    Scores queries via a pluggable ScoringStrategy.
    Caches results in QueryCache with Hebbian reinforcement.
    Domain-agnostic — works with any encoder config + model scorer.

    Lifecycle:
        1. __init__: create with a ScoringStrategy
        2. configure(func_glyphs): load function Glyphs from model
        3. find_similar(query_glyph): score query → ScorerResult
        4. reinforce(correct): Hebbian reinforcement on last cache hit
    """

    def __init__(
        self,
        scoring_strategy: ScoringStrategy | None = None,
        cache_threshold: float = 0.90,
        cache_size: int = 500,
    ):
        self._storage = InMemoryGlyphStorage(glyphs={})
        self._scoring = scoring_strategy or DefaultScoringStrategy()
        self._cache = QueryCache(
            similarity_fn=_vector_similarity,
            similarity_threshold=cache_threshold,
            max_size=cache_size,
        )
        self._configured = False

    def configure(self, func_glyphs: dict[str, Any]) -> None:
        """Load function Glyphs into storage.

        Called from SchemaIntentClassifier.configure() after the model
        has encoded function definitions. Clears previous state.

        Args:
            func_glyphs: Mapping of function_name → SDK Glyph
        """
        self._storage.clear()
        for glyph_id, glyph in func_glyphs.items():
            self._storage.add_glyph(glyph_id, glyph)
        self._cache.invalidate_all()
        self._configured = True
        logger.debug("GlyphSpace configured with %d function glyphs", len(func_glyphs))

    def find_similar(self, query_glyph: Any) -> ScorerResult:
        """Score a query Glyph against all stored function Glyphs.

        Pipeline:
          1. Check QueryCache (semantic match on cortex vector)
          2. Score against all functions via ScoringStrategy
          3. Build ScorerResult with irrelevance detection
          4. Cache the result

        Args:
            query_glyph: The encoded query (SDK Glyph from model.encode_query())

        Returns:
            ScorerResult with functions, confidence, all_scores, is_irrelevant
        """
        if not self._configured:
            return ScorerResult()

        # 1. Cache check using cortex vector
        cortex = getattr(query_glyph, "global_cortex", None)
        if cortex is not None:
            query_vec = Vector(
                data=cortex.data,
                dimension=len(cortex.data),
                space_id="glyph_space_cache",
            )
            cached = self._cache.get(query_vec)
            if cached is not None:
                result, similarity = cached
                logger.debug("GlyphSpace cache hit (sim=%.3f)", similarity)
                return result

        # 2. Score against all stored glyphs
        all_glyphs = self._storage.list_glyphs()
        scores: list[dict[str, Any]] = []

        for glyph_id, target_glyph in all_glyphs.items():
            sim = self._scoring.score_pair(query_glyph, target_glyph)
            scores.append({"function": glyph_id, "score": sim})

        scores.sort(key=lambda x: x["score"], reverse=True)

        # 3. Build result
        result = self._build_result(scores)

        # 4. Cache the result
        if cortex is not None:
            glyph_refs = set(all_glyphs.keys())
            self._cache.put(query_vec, "glyph_space_query", result, glyph_refs=glyph_refs)

        return result

    def reinforce(self, correct: bool) -> None:
        """Hebbian reinforcement on the most recently cached entry."""
        self._cache.reinforce(correct)

    @property
    def cache_size(self) -> int:
        """Number of entries in the query cache."""
        return self._cache.get_size()

    @property
    def glyph_count(self) -> int:
        """Number of function Glyphs in storage."""
        return len(self._storage.list_glyphs())

    def _build_result(self, scores: list[dict[str, Any]]) -> ScorerResult:
        """Convert sorted score list into a ScorerResult.

        Detects irrelevance when the best score is below threshold.
        """
        if not scores:
            return ScorerResult(is_irrelevant=True)

        best = scores[0]
        best_score = best["score"]

        # Irrelevance: best match is too weak
        if best_score < _IRRELEVANCE_THRESHOLD:
            return ScorerResult(
                confidence=best_score,
                all_scores=scores,
                is_irrelevant=True,
            )

        return ScorerResult(
            functions=[best["function"]],
            confidence=best_score,
            all_scores=scores,
        )
