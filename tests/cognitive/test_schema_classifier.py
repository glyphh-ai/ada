"""Tests for SchemaIntentClassifier — GlyphSpace + ModelScorer fallback."""

import pytest
import numpy as np

from glyphh.cognitive.schema_classifier import SchemaIntentClassifier
from glyphh.cognitive.model_scorer import ScorerResult
from glyphh.core.types import Vector, Glyph, Layer, Segment

# Reuse test domain fixtures
from .conftest import FUNC_SCHEMAS, DOMAIN_DICT


# ── Helpers ──

SPACE_ID = "test_space"


def _make_glyph(name: str, dim: int = 100, seed: int = 42) -> Glyph:
    """Create a minimal Glyph for testing with valid identifier format."""
    rng = np.random.RandomState(seed + abs(hash(name)) % 10000)
    cortex_data = rng.choice([-1, 1], size=dim).astype(np.int8)
    cortex = Vector(data=cortex_data, dimension=dim, space_id=SPACE_ID)

    role_data = rng.choice([-1, 1], size=dim).astype(np.int8)
    role_vec = Vector(data=role_data, dimension=dim, space_id=SPACE_ID)

    seg_data = rng.choice([-1, 1], size=dim).astype(np.int8)
    seg_cortex = Vector(data=seg_data, dimension=dim, space_id=SPACE_ID)
    segment = Segment(name="identity", cortex=seg_cortex, roles={"name": role_vec})

    layer_data = rng.choice([-1, 1], size=dim).astype(np.int8)
    layer_cortex = Vector(data=layer_data, dimension=dim, space_id=SPACE_ID)
    layer = Layer(name="signature", cortex=layer_cortex, segments={"identity": segment})

    return Glyph(
        identifier=f"{name}@2024-01-01T00:00:00Z#v1",
        name=name,
        space_id=SPACE_ID,
        global_cortex=cortex,
        layers={"signature": layer},
        metadata={"function_name": name},
    )


# ── Mock scorers ──

class MockModelScorer:
    """Mock scorer that returns configurable results (no GlyphSpace protocol)."""

    def __init__(self, results=None, default_confidence=0.8):
        self._results = results or {}
        self._default_confidence = default_confidence
        self._configured = False

    def configure(self, functions):
        self._configured = True

    def score(self, query: str) -> ScorerResult:
        if query in self._results:
            return self._results[query]
        return ScorerResult(
            functions=["display"],
            arguments={"display": {"item_name": "test"}},
            confidence=self._default_confidence,
            all_scores=[{"function": "display", "score": self._default_confidence}],
            is_irrelevant=False,
        )

    def score_multi(self, query: str) -> ScorerResult:
        return self.score(query)


class MockGlyphModelScorer:
    """Mock scorer that implements the full GlyphSpace protocol."""

    def __init__(self, dim=100, default_confidence=0.8):
        self._dim = dim
        self._default_confidence = default_confidence
        self._configured = False
        self._func_glyphs: dict[str, Glyph] = {}

    def configure(self, functions):
        self._configured = True
        self._func_glyphs = {
            f["name"]: _make_glyph(f["name"], dim=self._dim)
            for f in functions
        }

    def score(self, query: str) -> ScorerResult:
        return ScorerResult(
            functions=["display"],
            confidence=self._default_confidence,
            all_scores=[{"function": "display", "score": self._default_confidence}],
        )

    def score_multi(self, query: str) -> ScorerResult:
        return self.score(query)

    def encode_query(self, query: str) -> Glyph:
        """Encode query as a Glyph (deterministic per query string)."""
        return _make_glyph(f"query_{query}", dim=self._dim)

    def get_func_glyphs(self) -> dict[str, Glyph]:
        return dict(self._func_glyphs)

    def scoring_strategy(self):
        """Return a simple cosine strategy."""
        return None  # Use DefaultScoringStrategy


ACTION_TO_FUNC = DOMAIN_DICT["action_to_func"]


class TestSchemaClassifierConfigure:
    """Tests for configure()."""

    def test_configure_sets_configured_flag(self):
        scorer = MockModelScorer()
        classifier = SchemaIntentClassifier(dimension=1000, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        assert classifier._configured

    def test_configure_configures_scorer(self):
        scorer = MockModelScorer()
        classifier = SchemaIntentClassifier(dimension=1000, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        assert scorer._configured

    def test_configure_with_glyph_scorer_creates_glyph_space(self):
        scorer = MockGlyphModelScorer(dim=100)
        classifier = SchemaIntentClassifier(dimension=100, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        assert classifier._glyph_space is not None
        assert classifier._glyph_space.glyph_count == len(FUNC_SCHEMAS)

    def test_configure_without_glyph_protocol_no_glyph_space(self):
        scorer = MockModelScorer()
        classifier = SchemaIntentClassifier(dimension=1000, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        assert classifier._glyph_space is None

    def test_reconfigure_clears_glyph_space_cache(self):
        scorer = MockGlyphModelScorer(dim=100)
        classifier = SchemaIntentClassifier(dimension=100, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        # Classify to populate cache
        classifier.classify("show report", {"primary": "root"}, [])

        # Re-configure clears cache
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)
        assert classifier.cache_size == 0


class TestSchemaClassifierClassify:
    """Tests for classify()."""

    def test_unconfigured_returns_empty(self):
        scorer = MockModelScorer()
        classifier = SchemaIntentClassifier(dimension=1000, model_scorer=scorer)

        result = classifier.classify("show report", {"primary": "root"}, [])
        assert result["functions"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "unconfigured"

    def test_classify_with_glyph_scorer_uses_glyph_space(self):
        scorer = MockGlyphModelScorer(dim=100)
        classifier = SchemaIntentClassifier(dimension=100, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify(
            "show the report",
            {"primary": "root"},
            [],
        )

        # Should route through GlyphSpace (source starts with "glyph_space")
        assert result["source"].startswith("glyph_space")

    def test_classify_fallback_scorer_uses_score(self):
        scorer = MockModelScorer(default_confidence=0.8)
        classifier = SchemaIntentClassifier(dimension=1000, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify(
            "show the report",
            {"primary": "root"},
            [],
        )

        assert result["functions"] == ["display"]
        assert result["confidence"] == 0.8
        assert result["source"] == "model_scorer"

    def test_classify_scorer_irrelevant(self):
        scorer = MockModelScorer(results={
            "random gibberish": ScorerResult(
                functions=[],
                arguments={},
                confidence=0.1,
                all_scores=[],
                is_irrelevant=True,
            ),
        })
        classifier = SchemaIntentClassifier(dimension=1000, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify("random gibberish", {"primary": "root"}, [])
        assert result["functions"] == []
        assert result["source"] == "model_scorer_irrelevant"

    def test_classify_scorer_low_confidence_falls_through(self):
        scorer = MockModelScorer(results={
            "xyzzy plugh": ScorerResult(
                functions=["display"],
                arguments={},
                confidence=0.05,
                all_scores=[{"function": "display", "score": 0.05}],
                is_irrelevant=False,
            ),
        })
        classifier = SchemaIntentClassifier(dimension=1000, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify("xyzzy plugh", {"primary": "root"}, [])
        assert result["source"] == "none"
        assert result["functions"] == []
        assert result["confidence"] == 0.0

    def test_no_scorer_returns_none(self):
        classifier = SchemaIntentClassifier(dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify("show report", {"primary": "root"}, [])
        assert result["source"] == "none"
        assert result["functions"] == []


class TestSchemaClassifierConfirm:
    """Tests for confirm() — Hebbian reinforcement."""

    def test_confirm_on_empty_glyph_space_is_noop(self):
        scorer = MockGlyphModelScorer(dim=100)
        classifier = SchemaIntentClassifier(dimension=100, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        # Should not raise
        classifier.confirm(correct=True)
        classifier.confirm(correct=False)

    def test_confirm_without_scorer_is_noop(self):
        classifier = SchemaIntentClassifier(dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        # Should not raise
        classifier.confirm(correct=True)


class TestSchemaClassifierCacheSize:
    """Tests for cache_size property."""

    def test_cache_size_starts_at_zero(self):
        scorer = MockGlyphModelScorer(dim=100)
        classifier = SchemaIntentClassifier(dimension=100, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)
        assert classifier.cache_size == 0

    def test_cache_size_without_glyph_space(self):
        classifier = SchemaIntentClassifier(dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)
        assert classifier.cache_size == 0

    def test_classify_populates_cache(self):
        scorer = MockGlyphModelScorer(dim=100)
        classifier = SchemaIntentClassifier(dimension=100, model_scorer=scorer)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        classifier.classify("show report", {"primary": "root"}, [])
        assert classifier.cache_size == 1
