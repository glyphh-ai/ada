"""Unit tests for slot/type answerability — the deterministic answer-ness gate."""

import pytest

from glyphh.memory.answerability import (
    expected_type, satisfies, answerability_factor, CLOSED,
)


class TestExpectedType:
    @pytest.mark.parametrize("q,t", [
        ("what is Chris's favorite color?", "color"),
        ("what colour is the car?", "color"),
        ("how tall is the tower?", "number"),
        ("how many sons does he have?", "number"),
        ("at what temperature does water boil?", "number"),
        ("when was she born?", "date"),
        ("what year did it happen?", "date"),
        ("which company employs Chris?", "org"),
        ("what does Brandi do for a living?", "occupation"),
        ("where does Chris work?", "place"),     # 'where' → place (open-class, ok)
        ("what city is the capital?", "place"),
        ("who is Chris's wife?", "person"),
    ])
    def test_detects(self, q, t):
        assert expected_type(q) == t

    def test_none_when_no_strong_type(self):
        assert expected_type("tell me about the project") is None
        assert expected_type("what is the weather like?") is None


class TestSatisfies:
    def test_color(self):
        assert satisfies("Her car is blue.", "color")
        assert not satisfies("The user's name is Chris.", "color")

    def test_number(self):
        assert satisfies("Water boils at 100 degrees.", "number")
        assert not satisfies("The capital of France is Paris.", "number")

    def test_date(self):
        assert satisfies("She was born in 1989.", "date")
        assert satisfies("It happened in March.", "date")
        assert not satisfies("Chris works at Glyphh AI.", "date")

    def test_occupation(self):
        assert satisfies("Brandi is a nurse.", "occupation")
        assert not satisfies("Brandi likes coffee.", "occupation")

    def test_open_classes_need_proper_noun(self):
        assert satisfies("Chris is married to Brandi.", "person")
        assert satisfies("Chris works at Glyphh AI.", "org")
        assert not satisfies("the user likes it.", "person")


class TestAnswerabilityFactor:
    def test_leak_is_penalized(self):
        # The canonical leak: topically about Chris, but no color.
        f = answerability_factor(
            "what is Chris's favorite color?", "The user's name is Chris.",
        )
        assert f == 0.5  # closed-class miss → full penalty

    def test_real_answer_not_penalized(self):
        assert answerability_factor(
            "what does Brandi do for a living?", "Brandi is a nurse.",
        ) == 1.0

    def test_no_type_not_penalized(self):
        assert answerability_factor(
            "tell me about the project", "anything at all",
        ) == 1.0

    def test_open_class_miss_is_softer(self):
        # person expected, no proper noun present → softer than closed penalty
        f = answerability_factor("who is the user?", "the user likes it.")
        assert 0.5 < f < 1.0

    def test_closed_set_membership(self):
        assert "color" in CLOSED and "number" in CLOSED and "date" in CLOSED
        assert "person" not in CLOSED and "org" not in CLOSED
