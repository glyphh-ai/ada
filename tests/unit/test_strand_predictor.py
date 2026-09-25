"""Unit tests for glyphh.strand.predictor."""

import numpy as np
import pytest

from glyphh.core.ops import generate_symbol
from glyphh.strand import Strand
from glyphh.strand.predictor import StrandPredictor

DIM = 2048
SEED = 42


def sym(key: str) -> np.ndarray:
    return generate_symbol(SEED, key, DIM)


def test_empty_predictor_predicts_nothing():
    assert StrandPredictor(DIM).predict(sym("x")) == []


def test_learns_history_to_label_mapping():
    predictor = StrandPredictor(DIM)
    greet = Strand([sym("hello")]).state()
    order = Strand([sym("hello"), sym("i want food")]).state()
    predictor.observe(greet, "ASK_INTENT")
    predictor.observe(order, "REQUEST_SLOTS")
    assert predictor.predict(greet)[0][0] == "ASK_INTENT"
    assert predictor.predict(order)[0][0] == "REQUEST_SLOTS"


def test_surprise_orders_continuations():
    """The label that usually follows a state is less surprising there."""
    predictor = StrandPredictor(DIM)
    state_a = Strand([sym("a1"), sym("a2")]).state()
    state_b = Strand([sym("b1"), sym("b2")]).state()
    for _ in range(3):
        predictor.observe(state_a, "next-after-a")
        predictor.observe(state_b, "next-after-b")
    assert predictor.surprise(state_a, "next-after-a") < \
        predictor.surprise(state_a, "next-after-b")
    assert predictor.surprise(state_a, "never-seen") == 2.0


def test_session_surprise_mean():
    predictor = StrandPredictor(DIM)
    state = Strand([sym("s")]).state()
    predictor.observe(state, "known")
    known = predictor.surprise(state, "known")
    mixed = predictor.session_surprise([state, state], ["known", "unseen"])
    assert mixed == pytest.approx((known + 2.0) / 2)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        StrandPredictor(DIM).observe(np.ones(DIM + 1, dtype=np.int8), "x")
