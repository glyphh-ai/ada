"""Tests for the strand Anticipator wired into CognitiveLoop."""

import numpy as np
import pytest

from glyphh.cognitive.domain import DomainConfig
from glyphh.cognitive.loop import CognitiveLoop
from glyphh.core.ops import generate_symbol

from .conftest import DOMAIN_DICT, FUNC_SCHEMAS, make_state

DIM = 1000
SEED = 42


def sym(key: str) -> np.ndarray:
    return generate_symbol(SEED, key, DIM)


@pytest.fixture
def loop():
    config = DomainConfig.from_dict(DOMAIN_DICT)
    return CognitiveLoop(packs=[], domain_config=config, dimension=DIM)


def begin(loop):
    loop.begin(
        functions=FUNC_SCHEMAS,
        initial_state=make_state(primary="root.workspace"),
    )


def run_session(loop, turns):
    """Simulate a session as (idea-vector-key, resolved-functions) turns."""
    begin(loop)
    for key, funcs in turns:
        loop._record_turn(funcs, sym(key))


SESSION = [
    ("open the workspace", ["navigate"]),
    ("find the report", ["lookup"]),
    ("show it to me", ["display"]),
]


def test_fresh_loop_has_no_anticipation(loop):
    begin(loop)
    assert loop.anticipate() == []
    assert loop.drift() == 0.0


def test_anticipates_learned_next_step(loop):
    """After watching a workflow, the loop expects its next step."""
    for _ in range(3):
        run_session(loop, SESSION)
    begin(loop)
    loop._record_turn(["navigate"], sym("open the workspace"))
    predicted = loop.anticipate(top_k=1)
    assert predicted and predicted[0][0] == ("lookup",)
    loop._record_turn(["lookup"], sym("find the report"))
    predicted = loop.anticipate(top_k=1)
    assert predicted and predicted[0][0] == ("display",)


def test_begin_resets_session_not_learning(loop):
    run_session(loop, SESSION)
    assert loop.anticipator.turns == len(SESSION)
    begin(loop)
    assert loop.anticipator.turns == 0
    assert loop.anticipator.predictor.labels  # learning survived


def test_drift_flags_off_pattern_session(loop):
    """A session that breaks the learned order drifts more than one that follows it."""
    for _ in range(5):
        run_session(loop, SESSION)

    run_session(loop, SESSION)
    normal_drift = loop.drift()

    run_session(loop, [
        ("open the workspace", ["navigate"]),
        ("show it to me", ["display"]),
        ("find the report", ["lookup"]),
    ])
    off_pattern_drift = loop.drift()
    assert off_pattern_drift > normal_drift
