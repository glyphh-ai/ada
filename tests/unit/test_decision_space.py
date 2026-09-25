"""Tests for glyphh.decision.DecisionSpace."""

import pytest

from glyphh.decision import DecisionSpace

DIM = 2048


@pytest.fixture
def space():
    return DecisionSpace(dimension=DIM, k=5, min_precedent_sim=0.15)


def loan(amount, purpose, history):
    return {"amount": amount, "purpose": purpose, "credit_history": history}


@pytest.fixture
def seeded(space):
    for i in range(4):
        space.record(loan("small", "equipment", "good"), "approved",
                     facts={"case": f"good-{i}"})
    for i in range(4):
        space.record(loan("large", "speculation", "poor"), "declined",
                     facts={"case": f"poor-{i}"})
    return space


def test_empty_space_is_insufficient(space):
    answer = space.query(loan("small", "equipment", "good"))
    assert not answer.sufficient
    assert answer.confidence == 0.0
    assert "insufficient" in answer.fact_tree.to_text().lower()


def test_query_matches_precedent_cluster(seeded):
    answer = seeded.query(loan("small", "equipment", "good"))
    assert answer.sufficient
    assert answer.top_outcome == "approved"
    assert answer.outcomes["approved"] > 0.9


def test_confidence_drops_under_conflicting_precedent(seeded):
    clear = seeded.query(loan("small", "equipment", "good")).confidence
    # a situation between the clusters draws precedents from both
    mixed = seeded.query(loan("small", "speculation", "poor")).confidence
    assert mixed < clear


def test_novel_situation_declines_to_answer(seeded):
    answer = seeded.query(
        {"weather": "raining", "mood": "curious", "planet": "mars"})
    assert not answer.sufficient
    assert answer.outcomes == {}
    # it still names the nearest case rather than answering blind
    assert answer.precedents


def test_fact_tree_cites_precedents(seeded):
    answer = seeded.query(loan("small", "equipment", "good"))
    tree = answer.fact_tree.to_json()
    text = str(tree)
    assert "Precedent 1" in text
    assert "glyph_id" in text
    assert "data_hash" in text


def test_answer_serializes(seeded):
    answer = seeded.query(loan("small", "equipment", "good"))
    payload = answer.to_json()
    assert payload["sufficient"] is True
    assert payload["top_outcome"] == "approved"
    assert len(payload["precedents"]) == 5


def test_events_change_the_answer(space):
    """Same attributes, different order of events -> different precedent."""
    situation = {"channel": "support"}
    escalate_path = [{"act": "complain"}, {"act": "threaten_cancel"}]
    resolve_path = [{"act": "threaten_cancel"}, {"act": "complain"}]
    for _ in range(3):
        space.record(situation, "escalated", events=escalate_path)
        space.record(situation, "resolved", events=resolve_path)
    answer = space.query(situation, events=escalate_path)
    assert answer.top_outcome == "escalated"
    answer = space.query(situation, events=resolve_path)
    assert answer.top_outcome == "resolved"


def test_empty_situation_raises(space):
    with pytest.raises(ValueError):
        space.encode({})
