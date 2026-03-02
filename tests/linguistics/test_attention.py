"""
Tests for HDCAttention (Layer 5).

Key properties:
  - attend() returns bipolar int8 vector of correct shape
  - attend_roles() weights sum to ~1.0 (normalised)
  - Query near key → that key's value gets higher weight
  - top_role() returns the closest role key
  - Empty key_value_pairs raises ValueError
"""

import numpy as np
import pytest

from glyphh.core.ops import cosine_similarity, generate_symbol
from glyphh.linguistics.attention import HDCAttention


DIM  = 1000   # smaller for speed in tests
SEED = 42


@pytest.fixture(scope="module")
def attn():
    return HDCAttention(dimension=DIM)


def _sym(key: str) -> np.ndarray:
    return generate_symbol(SEED, key, DIM)


# ── attend() output shape and type ────────────────────────────────────────

def test_attend_output_bipolar(attn):
    q = _sym("query")
    kv = [(_sym("key_a"), _sym("val_a")), (_sym("key_b"), _sym("val_b"))]
    out = attn.attend(q, kv)
    assert out.dtype == np.int8
    assert set(np.unique(out)).issubset({-1, 1})
    assert len(out) == DIM


def test_attend_output_dimension(attn):
    q = _sym("q")
    kv = [(_sym("k1"), _sym("v1"))]
    out = attn.attend(q, kv)
    assert len(out) == DIM


def test_attend_empty_raises(attn):
    with pytest.raises(ValueError):
        attn.attend(_sym("q"), [])


# ── Attention focuses on relevant key ─────────────────────────────────────

def test_attend_weights_near_key(attn):
    """When query matches key_a, output should be near val_a."""
    val_a = _sym("val_a")
    val_b = _sym("val_b")

    # Make query identical to key_a
    key_a = _sym("key_a")
    key_b = _sym("key_b")
    query = key_a   # exact match with key_a

    out = attn.attend(query, [(key_a, val_a), (key_b, val_b)], relu=True)

    sim_a = float(cosine_similarity(out, val_a))
    sim_b = float(cosine_similarity(out, val_b))

    assert sim_a > sim_b, (
        f"Output should be closer to val_a ({sim_a:.3f}) than val_b ({sim_b:.3f})"
    )


# ── attend_roles() ─────────────────────────────────────────────────────────

def test_attend_roles_sums_to_one(attn):
    query = _sym("action_query")
    roles = {
        "action": _sym("action_proto"),
        "target": _sym("target_proto"),
        "domain": _sym("domain_proto"),
    }
    weights = attn.attend_roles(query, roles)
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.05, f"Weights sum {total:.4f} ≠ 1.0"


def test_attend_roles_returns_all_keys(attn):
    query = _sym("q")
    roles = {"a": _sym("a"), "b": _sym("b"), "c": _sym("c")}
    weights = attn.attend_roles(query, roles)
    assert set(weights.keys()) == {"a", "b", "c"}


def test_attend_roles_empty_returns_empty(attn):
    result = attn.attend_roles(_sym("q"), {})
    assert result == {}


def test_attend_roles_near_query_gets_high_weight(attn):
    """Role that matches query gets highest weight."""
    action_proto = _sym("action_proto")
    target_proto = _sym("target_proto")
    query = action_proto  # identical to action_proto

    weights = attn.attend_roles(query, {
        "action": action_proto,
        "target": target_proto,
    })

    assert weights["action"] > weights["target"], (
        f"action ({weights['action']:.3f}) should > target ({weights['target']:.3f})"
    )


# ── top_role() ─────────────────────────────────────────────────────────────

def test_top_role_returns_nearest(attn):
    query = _sym("verb_proto")
    roles = {
        "verb": _sym("verb_proto"),
        "noun": _sym("noun_proto"),
        "adj":  _sym("adj_proto"),
    }
    top, score = attn.top_role(query, roles)
    assert top == "verb", f"Expected 'verb', got '{top}'"
    assert score > 0.9  # should be ~1.0 (same vector)


def test_top_role_empty_raises(attn):
    with pytest.raises(ValueError):
        attn.top_role(_sym("q"), {})


# ── No-relu mode ───────────────────────────────────────────────────────────

def test_attend_no_relu(attn):
    q = _sym("q")
    kv = [(_sym("k1"), _sym("v1")), (_sym("k2"), _sym("v2"))]
    out = attn.attend(q, kv, relu=False)
    assert len(out) == DIM
    assert out.dtype == np.int8
