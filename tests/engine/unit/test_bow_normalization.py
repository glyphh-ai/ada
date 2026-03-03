"""
Unit tests for morphological normalization in bag-of-words encoding.

Tests verify that _encode_bag_of_words() normalizes word forms before
symbol generation, so "days" and "day", "running" and "run", etc.
produce identical or near-identical BoW vectors.
"""

import pytest
import numpy as np
from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.core.ops import cosine_similarity


@pytest.fixture(scope="module")
def encoder():
    return Encoder(EncoderConfig(dimension=10000, seed=42))


class TestBoWMorphologyNormalization:
    """Verify that BoW encoding normalizes morphological variants."""

    def test_plural_normalization(self, encoder):
        """'days' and 'day' should produce identical BoW vectors."""
        v1 = encoder._encode_bag_of_words("customer days")
        v2 = encoder._encode_bag_of_words("customer day")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim > 0.99, f"'days' vs 'day': similarity {sim:.4f}, expected > 0.99"

    def test_plural_logins(self, encoder):
        """'logins' and 'login' should produce identical BoW vectors."""
        v1 = encoder._encode_bag_of_words("zero logins last month")
        v2 = encoder._encode_bag_of_words("zero login last month")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim > 0.99, f"'logins' vs 'login': similarity {sim:.4f}, expected > 0.99"

    def test_plural_tickets(self, encoder):
        """'tickets' and 'ticket' should produce identical BoW vectors."""
        v1 = encoder._encode_bag_of_words("support tickets filed")
        v2 = encoder._encode_bag_of_words("support ticket filed")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim > 0.99, f"'tickets' vs 'ticket': similarity {sim:.4f}, expected > 0.99"

    def test_plural_customers(self, encoder):
        """'customers' and 'customer' should produce identical BoW vectors."""
        v1 = encoder._encode_bag_of_words("active customers growing")
        v2 = encoder._encode_bag_of_words("active customer growing")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim > 0.99, f"'customers' vs 'customer': similarity {sim:.4f}, expected > 0.99"

    def test_gerund_normalization(self, encoder):
        """'running' and 'run' should produce similar BoW vectors."""
        v1 = encoder._encode_bag_of_words("running process")
        v2 = encoder._encode_bag_of_words("run process")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim > 0.95, f"'running' vs 'run': similarity {sim:.4f}, expected > 0.95"

    def test_past_tense_normalization(self, encoder):
        """'walked' and 'walk' should produce similar BoW vectors."""
        v1 = encoder._encode_bag_of_words("walked forward")
        v2 = encoder._encode_bag_of_words("walk forward")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim > 0.95, f"'walked' vs 'walk': similarity {sim:.4f}, expected > 0.95"


class TestBoWBasicBehavior:
    """Verify BoW still works correctly with normalization enabled."""

    def test_shared_words_high_similarity(self, encoder):
        """Texts sharing words should have high similarity."""
        v1 = encoder._encode_bag_of_words("customer support cases")
        v2 = encoder._encode_bag_of_words("customer support tickets")
        sim = float(cosine_similarity(v1.data, v2.data))
        # Share 2 of 3 words
        assert sim > 0.3, f"Shared words similarity {sim:.4f}, expected > 0.3"

    def test_no_shared_words_low_similarity(self, encoder):
        """Texts with no shared words should have low similarity."""
        v1 = encoder._encode_bag_of_words("customer support cases")
        v2 = encoder._encode_bag_of_words("weather forecast tomorrow")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim < 0.3, f"No shared words similarity {sim:.4f}, expected < 0.3"

    def test_identical_text_perfect_similarity(self, encoder):
        """Same text should produce identical vectors."""
        v1 = encoder._encode_bag_of_words("customer with zero logins")
        v2 = encoder._encode_bag_of_words("customer with zero logins")
        sim = float(cosine_similarity(v1.data, v2.data))
        assert sim > 0.999, f"Identical text similarity {sim:.4f}, expected ~1.0"

    def test_deterministic(self, encoder):
        """Same text should always produce same vector."""
        v1 = encoder._encode_bag_of_words("test determinism")
        v2 = encoder._encode_bag_of_words("test determinism")
        assert np.array_equal(v1.data, v2.data)

    def test_empty_text(self, encoder):
        """Empty text should produce a valid vector."""
        v = encoder._encode_bag_of_words("")
        assert v.dimension == 10000
        assert np.all(np.isin(v.data, [-1, 1]))


class TestMorphologyEngineLazyInit:
    """Verify _get_morphology_engine() lazy initialization."""

    def test_returns_engine(self, encoder):
        """Should return a MorphologyEngine instance."""
        morph = encoder._get_morphology_engine()
        assert morph is not None

    def test_cached(self, encoder):
        """Should return the same instance on repeated calls."""
        morph1 = encoder._get_morphology_engine()
        morph2 = encoder._get_morphology_engine()
        assert morph1 is morph2
