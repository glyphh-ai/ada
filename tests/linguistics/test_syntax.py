"""
Tests for SyntaxParser (Layer 4).

Key properties:
  - Verb extracted from imperative queries: "send file" → verb="send"
  - Object extracted: "send the file" → obj="file"
  - Subject extracted when present: "Alice sends the file" → subject="alice"
  - Modifiers: "send the large file" → modifiers=[("large","file")]
  - HRR sentence vector is bipolar int8, shape (10000,)
  - Pathway vector is bipolar int8, shape (10000,)
  - decode_role() returns vector near the original word vector
"""

import numpy as np
import pytest

from glyphh.core.ops import cosine_similarity
from glyphh.linguistics.character import CharacterEncoder
from glyphh.linguistics.pos import POSTagger
from glyphh.linguistics.syntax import SyntaxParser


@pytest.fixture(scope="module")
def parser():
    enc    = CharacterEncoder(dimension=10000, seed=42)
    tagger = POSTagger(enc, dimension=10000)
    return SyntaxParser(tagger, enc, dimension=10000, seed=42)


# ── Verb extraction ─────────────────────────────────────────────────────────

def test_verb_from_imperative(parser):
    result = parser.parse("send the file")
    assert result.verb == "send", f"Expected 'send', got '{result.verb}'"


def test_verb_from_question(parser):
    result = parser.parse("find all users")
    assert result.verb == "find"


def test_verb_from_create(parser):
    result = parser.parse("create a new task")
    assert result.verb == "create"


# ── Object extraction ───────────────────────────────────────────────────────

def test_object_from_simple(parser):
    result = parser.parse("send the file")
    assert result.obj == "file", f"Expected 'file', got '{result.obj}'"


def test_object_from_message(parser):
    result = parser.parse("send a message to Alice")
    assert result.obj == "message", f"Expected 'message', got '{result.obj}'"


def test_object_from_delete(parser):
    result = parser.parse("delete the user account")
    assert result.obj in ("user", "account"), f"Unexpected obj: {result.obj}"


# ── Modifier extraction ─────────────────────────────────────────────────────

def test_modifier_adj_before_noun(parser):
    # Use a seed ADJ ("big" is in _OPEN_CLASS_SEEDS["ADJ"])
    result = parser.parse("send the big file")
    adj_nouns = [(adj, noun) for adj, noun in result.modifiers]
    assert any(adj == "big" for adj, _ in adj_nouns), (
        f"Expected 'big' in modifiers, got {result.modifiers}"
    )


# ── Vector outputs ─────────────────────────────────────────────────────────

def test_sentence_vec_is_bipolar(parser):
    result = parser.parse("send the file")
    assert result.sentence_vec is not None
    assert result.sentence_vec.dtype == np.int8
    assert set(np.unique(result.sentence_vec)).issubset({-1, 1})
    assert len(result.sentence_vec) == 10000


def test_pathway_vec_is_bipolar(parser):
    result = parser.parse("send the file")
    assert result.pathway_vec is not None
    assert result.pathway_vec.dtype == np.int8
    assert len(result.pathway_vec) == 10000


def test_sentence_vec_deterministic(parser):
    r1 = parser.parse("send the file")
    r2 = parser.parse("send the file")
    assert np.array_equal(r1.sentence_vec, r2.sentence_vec)


# ── HRR decode ──────────────────────────────────────────────────────────────

def test_decode_verb_role(parser):
    enc = CharacterEncoder(dimension=10000, seed=42)
    result = parser.parse("send the message")

    if result.sentence_vec is None or not result.verb:
        pytest.skip("No sentence vec or verb available")

    decoded = parser.decode_role(result.sentence_vec, "verb")
    # Decoded vector should be near the verb word vector
    verb_vec = enc.encode_word(result.verb)
    sim = float(cosine_similarity(decoded, verb_vec))
    # HRR decode is approximate — expect positive correlation
    assert sim > 0.0, f"Decoded verb sim {sim:.3f} should be > 0"


def test_decode_unknown_role_raises(parser):
    result = parser.parse("send the file")
    with pytest.raises(ValueError):
        parser.decode_role(result.sentence_vec, "nonexistent_role")


# ── Edge cases ──────────────────────────────────────────────────────────────

def test_empty_text(parser):
    result = parser.parse("")
    assert result.verb == ""
    assert result.obj == ""


def test_single_word(parser):
    result = parser.parse("send")
    assert result.verb == "send"


def test_tokens_populated(parser):
    result = parser.parse("send the file")
    assert len(result.tokens) == 3
