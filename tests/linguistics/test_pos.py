"""
Tests for POSTagger (Layer 3).

Key properties:
  - Closed-class words classified correctly with confidence 1.0
  - Seed VERB words classified as VERB
  - Seed NOUN words classified as NOUN
  - Numbers classified as NUM
  - Punctuation classified as PUNCT
  - Unknown words still get a classification (best prototype match)
  - Online learning: learn() updates prototype
"""

import pytest

from glyphh.linguistics.character import CharacterEncoder
from glyphh.linguistics.pos import POSTagger


@pytest.fixture(scope="module")
def tagger():
    enc = CharacterEncoder(dimension=10000, seed=42)
    return POSTagger(enc, dimension=10000)


# ── Closed-class words (grammar constants) ─────────────────────────────────

def test_determiner_the(tagger):
    pos, conf = tagger.tag_word("the")
    assert pos == "DET"
    assert conf == 1.0


def test_determiner_a(tagger):
    pos, conf = tagger.tag_word("a")
    assert pos == "DET"
    assert conf == 1.0


def test_modal_can(tagger):
    pos, conf = tagger.tag_word("can")
    assert pos == "MODAL"
    assert conf == 1.0


def test_modal_should(tagger):
    pos, conf = tagger.tag_word("should")
    assert pos == "MODAL"
    assert conf == 1.0


def test_preposition_in(tagger):
    pos, conf = tagger.tag_word("in")
    assert pos == "PREP"
    assert conf == 1.0


def test_preposition_for(tagger):
    # "for" appears in both PREP and CONJ — check it's handled
    pos, conf = tagger.tag_word("for")
    assert pos in ("PREP", "CONJ")
    assert conf == 1.0


def test_conjunction_and(tagger):
    pos, conf = tagger.tag_word("and")
    assert pos == "CONJ"
    assert conf == 1.0


# ── Open-class prototype classification ───────────────────────────────────

def test_seed_verb_send(tagger):
    pos, conf = tagger.tag_word("send")
    assert pos == "VERB", f"Expected VERB, got {pos}"
    assert conf > 0.0


def test_seed_verb_create(tagger):
    pos, conf = tagger.tag_word("create")
    assert pos == "VERB"


def test_seed_verb_find(tagger):
    pos, conf = tagger.tag_word("find")
    assert pos == "VERB"


def test_seed_noun_file(tagger):
    pos, conf = tagger.tag_word("file")
    assert pos == "NOUN", f"Expected NOUN, got {pos}"


def test_seed_noun_user(tagger):
    pos, conf = tagger.tag_word("user")
    assert pos == "NOUN"


def test_seed_adj_big(tagger):
    pos, conf = tagger.tag_word("big")
    assert pos == "ADJ"


def test_seed_adv_quickly(tagger):
    pos, conf = tagger.tag_word("quickly")
    assert pos == "ADV"


# ── Special tokens ─────────────────────────────────────────────────────────

def test_number_integer(tagger):
    pos, conf = tagger.tag_word("42")
    assert pos == "NUM"
    assert conf == 1.0


def test_number_float(tagger):
    pos, conf = tagger.tag_word("3.14")
    assert pos == "NUM"
    assert conf == 1.0


def test_number_word_one(tagger):
    pos, conf = tagger.tag_word("one")
    assert pos in ("NUM", "NOUN")  # "one" in NUM seeds but could match NOUN


def test_punctuation(tagger):
    pos, conf = tagger.tag_word("!")
    assert pos == "PUNCT"
    assert conf == 1.0


def test_punctuation_comma(tagger):
    pos, conf = tagger.tag_word(",")
    assert pos == "PUNCT"


# ── Full sentence tagging ──────────────────────────────────────────────────

def test_tag_sentence(tagger):
    results = tagger.tag("send the email quickly")
    words = [w for w, _, _ in results]
    tags  = [t for _, t, _ in results]

    assert "send" in words
    assert "the"  in words
    assert "email" in words
    assert "quickly" in words

    # "the" must be DET
    the_idx = words.index("the")
    assert tags[the_idx] == "DET"


def test_tag_returns_all_tokens(tagger):
    results = tagger.tag("create a new file")
    assert len(results) == 4


# ── Online learning ────────────────────────────────────────────────────────

def test_learn_new_word(tagger):
    # First, check a novel domain word
    pos_before, _ = tagger.tag_word("glyph")
    # Learn it as a NOUN
    tagger.learn("glyph", "NOUN")
    # After learning, should lean toward NOUN
    pos_after, conf_after = tagger.tag_word("glyph")
    assert pos_after == "NOUN"
