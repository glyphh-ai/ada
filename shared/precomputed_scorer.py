"""
PrecomputedScorer — ModelScorer backed by pre-computed similarity results.

Bridges async pgvector search → sync CognitiveLoop by pre-loading results.
Created per-query with the latest similarity search output, then injected
into a cached CognitiveLoop instance.

Usage:
    results = [{"concept_text": "slack-send-message", "score": 0.95, ...}, ...]
    scorer = PrecomputedScorer(results)
    loop = CognitiveLoop(model_scorer=scorer)
"""

from __future__ import annotations

from typing import Any

from glyphh.cognitive.model_scorer import ScorerResult


class PrecomputedScorer:
    """ModelScorer that returns pre-computed pgvector similarity results."""

    def __init__(self, similarity_results: list[dict]):
        """
        Args:
            similarity_results: Top matches from pgvector search.
                Each dict: {concept_text, score, metadata?, glyph_id?}
        """
        self._results = similarity_results or []
        self._func_names: set[str] = set()

    def configure(self, func_defs: list[dict[str, Any]]) -> None:
        """Store known function names for validation."""
        self._func_names = {f["name"] for f in func_defs}

    def score(self, query: str) -> ScorerResult:
        """Return pre-computed similarity results as a ScorerResult."""
        if not self._results:
            return ScorerResult(is_irrelevant=True, confidence=0.0)

        functions = [r["concept_text"] for r in self._results]
        all_scores = [
            {"function": r["concept_text"], "score": r["score"]}
            for r in self._results
        ]

        top_score = self._results[0]["score"]
        gap = (
            self._results[0]["score"] - self._results[1]["score"]
            if len(self._results) > 1
            else 0.1
        )
        gap_factor = min(gap / 0.05, 1.0)
        confidence = top_score * (0.6 + 0.4 * gap_factor)

        return ScorerResult(
            functions=functions[:1],
            arguments={},
            confidence=confidence,
            all_scores=all_scores,
            is_irrelevant=False,
        )

    def score_multi(self, query: str) -> ScorerResult:
        """Score for multiple matching functions (same as score)."""
        return self.score(query)

    def encode_query(self, query: str) -> Any:
        """Not used — pgvector handles encoding."""
        return None

    def get_func_glyphs(self) -> dict[str, Any]:
        """Not used — glyphs live in pgvector, not in-memory."""
        return {}

    def scoring_strategy(self) -> Any:
        """Use default scoring strategy."""
        return None
