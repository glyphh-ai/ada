"""Intent classification from function schemas — GlyphSpace routing.

Classification pipeline:
  1. GlyphSpace   (model encodes query → score against function Glyphs)
  2. Fallback     (direct ModelScorer.score() for backwards compatibility)

When a ModelScorer implements the full protocol (encode_query, get_func_glyphs,
scoring_strategy), the GlyphSpace path is used. QueryCache in GlyphSpace
provides Hebbian-reinforced caching of repeat patterns.

Models that only implement score() fall back to the direct-call path.
"""

from __future__ import annotations

import logging
from typing import Any

from .glyph_space import GlyphSpace
from .model_scorer import ModelScorer, ScorerResult

logger = logging.getLogger(__name__)


class SchemaIntentClassifier:
    """Intent classifier backed by GlyphSpace.

    Usage:
        classifier = SchemaIntentClassifier(
            model_scorer=scorer,     # optional
            dimension=10000,
        )
        classifier.configure(functions, action_to_func)  # at begin() time

        result = classifier.classify(query, state, recent_actions)
        # result = {functions, arguments, confidence, source}

        classifier.confirm(correct=True)  # Hebbian reinforcement
    """

    # Confidence tiers for gating
    _HIGH_CONFIDENCE = 0.50     # Trust result directly
    _UNCERTAIN_CONFIDENCE = 0.20  # Still return, but lower confidence

    def __init__(
        self,
        dimension: int = 10000,
        cache_threshold: float = 0.85,
        model_scorer: ModelScorer | None = None,
    ):
        self._scorer = model_scorer
        self._glyph_space: GlyphSpace | None = None

        # Set at configure() time
        self._func_schemas: dict[str, dict] = {}
        self._action_to_func: dict[str, str] = {}
        self._configured = False

    def configure(
        self,
        functions: list[dict[str, Any]],
        action_to_func: dict[str, str],
    ) -> None:
        """Configure the classifier with available function schemas.

        Called from CognitiveLoop.begin(). Called once per session, not per query.

        Args:
            functions: Function schemas [{name, description, parameters}]
            action_to_func: Mapping from action verbs to function names
        """
        self._func_schemas = {f["name"]: f for f in functions}
        self._action_to_func = action_to_func

        # Configure model scorer with function definitions
        if self._scorer is not None:
            self._scorer.configure(functions)

            # Build GlyphSpace if scorer supports the full protocol
            strategy = None
            if hasattr(self._scorer, "scoring_strategy"):
                strategy = self._scorer.scoring_strategy()

            func_glyphs: dict[str, Any] = {}
            if hasattr(self._scorer, "get_func_glyphs"):
                func_glyphs = self._scorer.get_func_glyphs()

            if func_glyphs:
                self._glyph_space = GlyphSpace(scoring_strategy=strategy)
                self._glyph_space.configure(func_glyphs)
                logger.debug(
                    "GlyphSpace initialized with %d glyphs", len(func_glyphs),
                )
            else:
                self._glyph_space = None

        self._configured = True

    def classify(
        self,
        query: str,
        state: dict[str, Any],
        recent_actions: list[str],
    ) -> dict[str, Any]:
        """Classify intent via GlyphSpace or fallback ModelScorer.

        Returns:
            {
                "functions": ["func_name", ...],
                "arguments": {"func_name": {"param": "value"}},
                "confidence": 0.0-1.0,
                "source": "glyph_space" | "model_scorer" | ... | "none",
            }
        """
        if not self._configured:
            return {
                "functions": [],
                "arguments": {},
                "confidence": 0.0,
                "source": "unconfigured",
            }

        # ── Primary path: GlyphSpace (model encodes query → score against storage) ──
        if (
            self._glyph_space is not None
            and self._scorer is not None
            and hasattr(self._scorer, "encode_query")
        ):
            query_glyph = self._scorer.encode_query(query)
            if query_glyph is not None:
                scorer_result = self._glyph_space.find_similar(query_glyph)
                return self._scorer_result_to_dict(scorer_result, source="glyph_space")

        # ── Fallback: direct ModelScorer scoring ──
        if self._scorer is not None:
            # Prefer score_multi() for multi-function detection (gap analysis)
            if hasattr(self._scorer, "score_multi"):
                scorer_result = self._scorer.score_multi(query)
            else:
                scorer_result = self._scorer.score(query)

            if scorer_result.is_irrelevant:
                logger.debug("ModelScorer: irrelevant (conf=%.3f)", scorer_result.confidence)
                return {
                    "functions": [],
                    "arguments": {},
                    "confidence": scorer_result.confidence,
                    "source": "model_scorer_irrelevant",
                    "all_scores": scorer_result.all_scores,
                }

            if scorer_result.confidence >= self._HIGH_CONFIDENCE:
                return self._scorer_result_to_dict(scorer_result, source="model_scorer")

            if scorer_result.confidence >= self._UNCERTAIN_CONFIDENCE:
                return self._scorer_result_to_dict(scorer_result, source="model_scorer")

            # LOW confidence — fall through
            logger.debug(
                "ModelScorer LOW (conf=%.3f), falling through",
                scorer_result.confidence,
            )

        # ── No classification source available ──
        return {
            "functions": [],
            "arguments": {},
            "confidence": 0.0,
            "source": "none",
        }

    def confirm(self, correct: bool) -> None:
        """Hebbian reinforcement on the most recently cached entry."""
        if self._glyph_space is not None:
            self._glyph_space.reinforce(correct)

    @property
    def cache_size(self) -> int:
        """Number of entries in the GlyphSpace cache."""
        if self._glyph_space is not None:
            return self._glyph_space.cache_size
        return 0

    def _scorer_result_to_dict(
        self,
        result: ScorerResult,
        source: str,
    ) -> dict[str, Any]:
        """Convert a ScorerResult into the classify() return format."""
        if result.is_irrelevant:
            return {
                "functions": [],
                "arguments": {},
                "confidence": result.confidence,
                "source": f"{source}_irrelevant",
                "all_scores": result.all_scores,
            }

        return {
            "functions": result.functions,
            "arguments": result.arguments,
            "confidence": result.confidence,
            "source": source,
            "all_scores": result.all_scores,
        }
