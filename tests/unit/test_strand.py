"""Unit tests for glyphh.strand — DNA-shaped sequence operations."""

import numpy as np
import pytest

from glyphh.core.ops import cosine_similarity, generate_symbol
from glyphh.strand import Strand, duplex, permute, splice, transcribe

DIM = 2048
SEED = 42


def sym(key: str) -> np.ndarray:
    return generate_symbol(SEED, key, DIM)


def test_permute_is_invertible():
    v = sym("a")
    assert np.array_equal(permute(permute(v, 3), -3), v)


def test_order_is_structural():
    """The lattice can't tell 'a then b' from 'b then a'; a strand can."""
    a, b = sym("a"), sym("b")
    ab = Strand([a, b]).state()
    ba = Strand([b, a]).state()
    assert abs(cosine_similarity(ab, ba)) < 0.2


def test_shared_recent_suffix_is_similar():
    """States anchor at the tail: shared recent history -> high similarity."""
    past1, past2 = sym("p1"), sym("p2")
    recent = [sym("r1"), sym("r2"), sym("r3")]
    s1 = Strand([past1] + recent).state()
    s2 = Strand([past2] + recent).state()
    assert cosine_similarity(s1, s2) > 0.5


def test_complement_is_exact():
    s = Strand([sym("a"), sym("b")])
    c = s.complement()
    for orig, comp in zip(s.codons, c.codons):
        assert np.array_equal(orig, -comp)


def test_duplex_transcription_is_lossless():
    """Given the duplex and the template, the coding strand is exactly recovered."""
    template = Strand([sym(f"t{i}") for i in range(5)])
    coding = Strand([sym(f"c{i}") for i in range(5)])
    pairs = duplex(template, coding)
    recovered = transcribe(pairs, template)
    for orig, rec in zip(coding.codons, recovered.codons):
        assert np.array_equal(orig, rec)


def test_duplex_length_mismatch_raises():
    with pytest.raises(ValueError):
        duplex(Strand([sym("a")]), Strand([sym("b"), sym("c")]))


def test_splice_crossover():
    a = Strand([sym(f"a{i}") for i in range(4)])
    b = Strand([sym(f"b{i}") for i in range(4)])
    child = splice(a, b, at_a=2, at_b=2)
    assert len(child) == 4
    assert np.array_equal(child.codons[0], a.codons[0])
    assert np.array_equal(child.codons[1], a.codons[1])
    assert np.array_equal(child.codons[2], b.codons[2])
    assert np.array_equal(child.codons[3], b.codons[3])


def test_kmer_profile_survives_offset():
    """Raw states of offset strands diverge; k-mer profiles stay close."""
    seq = [sym(f"x{i}") for i in range(8)]
    full = Strand(seq)
    shifted = Strand(seq[1:])
    profile_sim = cosine_similarity(full.kmer_profile(2), shifted.kmer_profile(2))
    assert profile_sim > 0.5


def test_empty_strand_state_raises():
    with pytest.raises(ValueError):
        Strand().state()
