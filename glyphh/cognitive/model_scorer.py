"""
ModelScorer protocol — domain models provide HDC encoding and scoring.

A ModelScorer is the bridge between a domain-specific HDC model and
the CognitiveLoop. The model encodes function definitions and queries
into its own vector space, then scores them. The CognitiveLoop uses
these scores as the primary classification signal.

Protocol:
    scorer.configure(func_defs)   — encode function definitions (once per session)
    scorer.score(query)           — score query against encoded functions
    scorer.score_multi(query)     — score for multiple matching functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ScorerResult:
    """Result from a ModelScorer.score() call.

    Designed to map directly into SchemaIntentClassifier's return format:
    {functions, arguments, confidence, source}.
    """

    functions: list[str] = field(default_factory=list)
    arguments: dict[str, dict] = field(default_factory=dict)
    confidence: float = 0.0
    all_scores: list[dict] = field(default_factory=list)  # [{function, score}]
    is_irrelevant: bool = False


@runtime_checkable
class ModelScorer(Protocol):
    """Protocol for domain-specific HDC model scorers.

    Any model that implements configure() and score() can be plugged
    into SchemaIntentClassifier as the first classification path.

    The scorer is reconfigured per entry (different function sets per
    BFCL entry), so configure() must be fast — it encodes the function
    definitions into the model's vector space.
    """

    def configure(self, func_defs: list[dict[str, Any]]) -> None:
        """Encode function definitions into the model's vector space.

        Called from SchemaIntentClassifier.configure(). Must be fast —
        called once per session/entry, not per query.

        Args:
            func_defs: Function schemas [{name, description, parameters}]
        """
        ...

    def score(self, query: str) -> ScorerResult:
        """Score a query against the configured function definitions.

        Returns a ScorerResult with:
          - functions: ordered list of matched function names (best first)
          - arguments: {} (model doesn't extract args, LLM does that)
          - confidence: 0.0-1.0 composite score
          - all_scores: per-function scores for debugging
          - is_irrelevant: True if query doesn't match any function
        """
        ...

    def score_multi(self, query: str) -> ScorerResult:
        """Score a query for multiple matching functions.

        Models that implement multi-function gap analysis override this.
        Default implementations can delegate to score().
        """
        ...

    def encode_query(self, query: str) -> Any:
        """Encode a query into a Glyph in the model's vector space.

        Used by GlyphSpace to get a query Glyph for similarity scoring
        against stored function Glyphs. Models that implement this method
        enable the GlyphSpace routing path.

        Returns:
            An SDK Glyph, or None if the model doesn't support this.
        """
        ...

    def get_func_glyphs(self) -> dict[str, Any]:
        """Return the encoded function Glyphs (after configure()).

        GlyphSpace stores these in InMemoryGlyphStorage for scoring.

        Returns:
            Mapping of function_name → SDK Glyph.
            Empty dict if not configured or not supported.
        """
        ...

    def scoring_strategy(self) -> Any:
        """Return the model's scoring strategy for GlyphSpace.

        Models with custom similarity logic (e.g. hierarchical multi-level
        scoring) return a ScoringStrategy instance. Models that don't
        implement this get DefaultScoringStrategy (cortex cosine similarity).

        Returns:
            A ScoringStrategy instance, or None for default.
        """
        ...
