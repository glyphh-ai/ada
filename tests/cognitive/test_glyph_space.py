"""Tests for GlyphSpace — unified glyph storage + scoring + caching."""

import pytest
import numpy as np

from glyphh.cognitive.glyph_space import (
    GlyphSpace,
    ScoringStrategy,
    DefaultScoringStrategy,
)
from glyphh.cognitive.model_scorer import ScorerResult
from glyphh.core.types import Vector, Glyph, Layer, Segment


DIM = 100
SPACE_ID = "test_space"


def _make_glyph(name: str, seed: int = 42) -> Glyph:
    """Create a minimal Glyph for testing with valid identifier format."""
    rng = np.random.RandomState(seed + abs(hash(name)) % 10000)
    cortex_data = rng.choice([-1, 1], size=DIM).astype(np.int8)
    cortex = Vector(data=cortex_data, dimension=DIM, space_id=SPACE_ID)

    role_data = rng.choice([-1, 1], size=DIM).astype(np.int8)
    role_vec = Vector(data=role_data, dimension=DIM, space_id=SPACE_ID)

    seg_data = rng.choice([-1, 1], size=DIM).astype(np.int8)
    seg_cortex = Vector(data=seg_data, dimension=DIM, space_id=SPACE_ID)
    segment = Segment(name="identity", cortex=seg_cortex, roles={"name": role_vec})

    layer_data = rng.choice([-1, 1], size=DIM).astype(np.int8)
    layer_cortex = Vector(data=layer_data, dimension=DIM, space_id=SPACE_ID)
    layer = Layer(name="signature", cortex=layer_cortex, segments={"identity": segment})

    return Glyph(
        identifier=f"{name}@2024-01-01T00:00:00Z#v1",
        name=name,
        space_id=SPACE_ID,
        global_cortex=cortex,
        layers={"signature": layer},
        metadata={"function_name": name},
    )


class FixedScoreStrategy:
    """Strategy that returns a fixed score per function name (for testing)."""

    def __init__(self, score_map: dict[str, float]):
        self._scores = score_map

    def score_pair(self, query_glyph, target_glyph) -> float:
        name = getattr(target_glyph, "name", "")
        return self._scores.get(name, 0.0)


class TestGlyphSpaceConfigure:
    """Tests for GlyphSpace.configure()."""

    def test_configure_stores_glyphs(self):
        space = GlyphSpace()
        glyphs = {
            "navigate": _make_glyph("navigate"),
            "display": _make_glyph("display"),
        }
        space.configure(glyphs)

        assert space.glyph_count == 2
        assert space._configured

    def test_configure_clears_previous(self):
        space = GlyphSpace()
        space.configure({"navigate": _make_glyph("navigate")})
        assert space.glyph_count == 1

        space.configure({"display": _make_glyph("display"), "lookup": _make_glyph("lookup")})
        assert space.glyph_count == 2

    def test_configure_clears_cache(self):
        space = GlyphSpace()
        space.configure({"display": _make_glyph("display")})

        # Populate cache
        query = _make_glyph("query_show")
        space.find_similar(query)
        assert space.cache_size == 1

        # Reconfigure clears cache
        space.configure({"display": _make_glyph("display")})
        assert space.cache_size == 0


class TestGlyphSpaceFindSimilar:
    """Tests for GlyphSpace.find_similar()."""

    def test_unconfigured_returns_empty(self):
        space = GlyphSpace()
        query = _make_glyph("query")

        result = space.find_similar(query)
        assert result.functions == []
        assert result.confidence == 0.0

    def test_find_similar_returns_best_match(self):
        strategy = FixedScoreStrategy({
            "navigate": 0.3,
            "display": 0.8,
            "lookup": 0.5,
        })
        space = GlyphSpace(scoring_strategy=strategy)
        space.configure({
            "navigate": _make_glyph("navigate"),
            "display": _make_glyph("display"),
            "lookup": _make_glyph("lookup"),
        })

        result = space.find_similar(_make_glyph("query"))
        assert result.functions == ["display"]
        assert result.confidence == 0.8

    def test_find_similar_irrelevant_when_scores_low(self):
        strategy = FixedScoreStrategy({
            "navigate": 0.05,
            "display": 0.10,
        })
        space = GlyphSpace(scoring_strategy=strategy)
        space.configure({
            "navigate": _make_glyph("navigate"),
            "display": _make_glyph("display"),
        })

        result = space.find_similar(_make_glyph("query"))
        assert result.is_irrelevant
        assert result.functions == []

    def test_find_similar_caches_result(self):
        strategy = FixedScoreStrategy({"display": 0.7})
        space = GlyphSpace(scoring_strategy=strategy)
        space.configure({"display": _make_glyph("display")})

        query = _make_glyph("query")
        result1 = space.find_similar(query)
        assert space.cache_size == 1

        # Same query should hit cache
        result2 = space.find_similar(query)
        assert result2.functions == result1.functions

    def test_find_similar_all_scores_sorted(self):
        strategy = FixedScoreStrategy({
            "navigate": 0.3,
            "display": 0.8,
            "lookup": 0.5,
        })
        space = GlyphSpace(scoring_strategy=strategy)
        space.configure({
            "navigate": _make_glyph("navigate"),
            "display": _make_glyph("display"),
            "lookup": _make_glyph("lookup"),
        })

        result = space.find_similar(_make_glyph("query"))
        scores = result.all_scores
        assert len(scores) == 3
        # Should be sorted descending
        assert scores[0]["score"] >= scores[1]["score"] >= scores[2]["score"]


class TestGlyphSpaceDefaultStrategy:
    """Tests for DefaultScoringStrategy."""

    def test_default_strategy_uses_cortex_cosine(self):
        strategy = DefaultScoringStrategy()

        # Two identical glyphs should score high
        glyph = _make_glyph("test")
        score = strategy.score_pair(glyph, glyph)
        assert score > 0.99  # Same vector → cosine ≈ 1.0

    def test_default_strategy_different_glyphs(self):
        strategy = DefaultScoringStrategy()

        g1 = _make_glyph("alpha", seed=1)
        g2 = _make_glyph("beta", seed=99)
        score = strategy.score_pair(g1, g2)
        # Different random vectors → low cosine
        assert -0.5 < score < 0.5

    def test_protocol_compliance(self):
        """DefaultScoringStrategy satisfies ScoringStrategy protocol."""
        strategy = DefaultScoringStrategy()
        assert isinstance(strategy, ScoringStrategy)


class TestGlyphSpaceReinforce:
    """Tests for Hebbian reinforcement."""

    def test_reinforce_without_cache_hit_is_noop(self):
        space = GlyphSpace()
        space.configure({"display": _make_glyph("display")})

        # Should not raise
        space.reinforce(correct=True)
        space.reinforce(correct=False)

    def test_reinforce_after_cache_hit(self):
        strategy = FixedScoreStrategy({"display": 0.7})
        space = GlyphSpace(scoring_strategy=strategy)
        space.configure({"display": _make_glyph("display")})

        query = _make_glyph("query")
        space.find_similar(query)  # populates cache

        # Hit cache
        space.find_similar(query)

        # Reinforce should not raise
        space.reinforce(correct=True)
