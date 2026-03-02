"""
Tests for MorphologyEngine (Layer 2).

Key properties:
  - Plurals are normalised to singular: "sculptures" → "sculpture"
  - Past tense is normalised to infinitive: "walked" → "walk"
  - Gerunds are normalised to infinitive: "running" → "run"
  - Base forms are preserved: "dog" → "dog"
  - Unknown words fall back to base: "xyzzy" → "xyzzy"
  - Tag is returned correctly: "dogs" → (word, "plural")
"""

import pytest

from glyphh.linguistics.character import CharacterEncoder
from glyphh.linguistics.morphology import MorphologyEngine


@pytest.fixture(scope="module")
def morph():
    enc = CharacterEncoder(dimension=10000, seed=42)
    return MorphologyEngine(enc)


# ── Plural normalisation ────────────────────────────────────────────────────

def test_regular_plural(morph):
    lemma, tag = morph.normalize("dogs")
    assert lemma == "dog", f"Expected 'dog', got '{lemma}'"
    assert tag == "plural"


def test_regular_plural_cats(morph):
    lemma, tag = morph.normalize("cats")
    assert lemma == "cat"
    assert tag == "plural"


def test_regular_plural_books(morph):
    lemma, tag = morph.normalize("books")
    assert lemma == "book"
    assert tag == "plural"


def test_sibilant_plural_boxes(morph):
    lemma, tag = morph.normalize("boxes")
    assert lemma == "box"
    assert tag == "plural"


def test_domain_word_plural_files(morph):
    lemma, tag = morph.normalize("files")
    assert lemma == "file"
    assert tag == "plural"


def test_domain_word_plural_tables(morph):
    lemma, tag = morph.normalize("tables")
    assert lemma == "table"
    assert tag == "plural"


# ── Past tense normalisation ────────────────────────────────────────────────

def test_past_tense_walked(morph):
    lemma, tag = morph.normalize("walked")
    assert lemma == "walk"
    assert tag == "past"


def test_past_tense_moved(morph):
    lemma, tag = morph.normalize("moved")
    assert lemma == "move"
    assert tag == "past"


def test_past_tense_opened(morph):
    lemma, tag = morph.normalize("opened")
    assert lemma == "open"
    assert tag == "past"


# ── Gerund normalisation ────────────────────────────────────────────────────

def test_gerund_running(morph):
    lemma, tag = morph.normalize("running")
    assert lemma == "run"
    assert tag == "gerund"


def test_gerund_walking(morph):
    lemma, tag = morph.normalize("walking")
    assert lemma == "walk"
    assert tag == "gerund"


def test_gerund_building(morph):
    lemma, tag = morph.normalize("building")
    assert lemma == "build"
    assert tag == "gerund"


# ── Comparative normalisation ───────────────────────────────────────────────

def test_comparative_bigger(morph):
    lemma, tag = morph.normalize("bigger")
    assert lemma == "big"
    assert tag == "comparative"


def test_comparative_faster(morph):
    lemma, tag = morph.normalize("faster")
    assert lemma == "fast"
    assert tag == "comparative"


# ── Base form preservation ──────────────────────────────────────────────────

def test_base_form_preserved_noun(morph):
    lemma, tag = morph.normalize("dog")
    assert lemma == "dog"
    assert tag == "base"


def test_base_form_preserved_verb(morph):
    lemma, tag = morph.normalize("run")
    assert lemma == "run"
    assert tag == "base"


# ── Unknown words ───────────────────────────────────────────────────────────

def test_unknown_word_fallback(morph):
    # A random word not in any seed — should return base
    lemma, tag = morph.normalize("xyzzyplurb")
    assert tag == "base"
    assert isinstance(lemma, str)


# ── Transform access ────────────────────────────────────────────────────────

def test_get_transform_exists(morph):
    t = morph.get_transform("plural")
    assert t is not None
    assert len(t) == 10000


def test_get_transform_unknown(morph):
    t = morph.get_transform("nonexistent_transform")
    assert t is None
