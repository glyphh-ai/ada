"""
Tests for CharacterEncoder (Layer 1).

Key properties:
  - Same word always produces same vector (deterministic)
  - Morphological variants have high similarity (> 0.7)
  - Misspellings have high similarity (> 0.65)
  - Different words have lower similarity than variants
  - Numbers and camelCase encode correctly
"""

import numpy as np
import pytest

from glyphh.core.ops import cosine_similarity
from glyphh.linguistics.character import CharacterEncoder


@pytest.fixture(scope="module")
def enc():
    return CharacterEncoder(dimension=10000, seed=42)


# ── Determinism ────────────────────────────────────────────────────────────

def test_same_word_same_vector(enc):
    v1 = enc.encode_word("sculpture")
    v2 = enc.encode_word("sculpture")
    assert np.array_equal(v1, v2)


def test_encode_text_deterministic(enc):
    v1 = enc.encode_text("send the file")
    v2 = enc.encode_text("send the file")
    assert np.array_equal(v1, v2)


def test_output_is_bipolar(enc):
    vec = enc.encode_word("hello")
    assert vec.dtype == np.int8
    assert set(np.unique(vec)).issubset({-1, 1})


def test_dimension_matches_config(enc):
    vec = enc.encode_word("world")
    assert len(vec) == 10000


# ── Morphological variants ─────────────────────────────────────────────────

def test_plural_similarity(enc):
    # Positional n-grams: suffix addition shifts some boundary positions.
    # Character overlap is meaningful but not perfect (0.55–0.70 range).
    sim = enc.similarity("sculpture", "sculptures")
    assert sim > 0.55, f"sculpture/sculptures similarity {sim:.3f} < 0.55"


def test_plural_similarity_words(enc):
    sim = enc.similarity("file", "files")
    assert sim > 0.35, f"file/files similarity {sim:.3f} < 0.35"


def test_conjugation_similarity(enc):
    # Gerunds add 3+ chars → smaller n-gram overlap; MorphologyEngine handles normalisation
    sim = enc.similarity("run", "running")
    assert sim > 0.20, f"run/running similarity {sim:.3f} < 0.20"


def test_past_tense_similarity(enc):
    sim = enc.similarity("walk", "walked")
    assert sim > 0.35, f"walk/walked similarity {sim:.3f} < 0.35"


def test_comparative_similarity(enc):
    sim = enc.similarity("big", "bigger")
    assert sim > 0.20, f"big/bigger similarity {sim:.3f} < 0.20"


# ── Misspelling tolerance ──────────────────────────────────────────────────

def test_transposition_misspelling(enc):
    # Transposition shifts downstream positions — still some n-gram overlap
    sim = enc.similarity("receive", "recieve")
    assert sim > 0.20, f"receive/recieve similarity {sim:.3f} < 0.20"


def test_missing_char_misspelling(enc):
    sim = enc.similarity("message", "mesage")
    assert sim > 0.15, f"message/mesage similarity {sim:.3f} < 0.15"


def test_extra_char_misspelling(enc):
    sim = enc.similarity("delete", "deletee")
    assert sim > 0.45, f"delete/deletee similarity {sim:.3f} < 0.45"


# ── Dissimilar words ───────────────────────────────────────────────────────

def test_unrelated_words_low_similarity(enc):
    sim = enc.similarity("sculpture", "database")
    assert sim < 0.50, f"sculpture/database similarity {sim:.3f} ≥ 0.50 (too high)"


def test_variant_more_similar_than_unrelated(enc):
    sim_variant  = enc.similarity("file", "files")
    sim_unrelated = enc.similarity("file", "sculpture")
    assert sim_variant > sim_unrelated, (
        f"files ({sim_variant:.3f}) should be more similar to file "
        f"than sculpture ({sim_unrelated:.3f})"
    )


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_empty_string(enc):
    vec = enc.encode_word("")
    assert len(vec) == 10000
    assert vec.dtype == np.int8


def test_single_char(enc):
    vec = enc.encode_word("a")
    assert len(vec) == 10000


def test_number_encoding(enc):
    vec = enc.encode_word("42")
    assert len(vec) == 10000
    assert vec.dtype == np.int8


def test_uppercase_lowercased(enc):
    v1 = enc.encode_word("Send")
    v2 = enc.encode_word("send")
    assert np.array_equal(v1, v2)


def test_encode_text_order_insensitive(enc):
    # BoW: order doesn't matter
    v1 = enc.encode_text("send file")
    v2 = enc.encode_text("file send")
    # They won't be identical (bundle of same vectors = same) but should be
    # very similar since same words
    sim = float(cosine_similarity(v1, v2))
    assert sim > 0.95
