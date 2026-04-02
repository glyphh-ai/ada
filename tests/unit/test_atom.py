"""
Tests for glyphh.memory.atom — neuron-level concept primitives.
"""

import numpy as np
import pytest

from glyphh.memory.atom import Atom, AtomForge


class TestAtom:
    def test_reinforce(self):
        a = Atom(name="test", vector=np.ones(10, dtype=np.int8))
        a.strength = 0.5
        a.reinforce(0.2)
        assert a.strength > 0.5

    def test_reinforce_cap(self):
        a = Atom(name="test", vector=np.ones(10, dtype=np.int8))
        a.strength = 2.9
        a.reinforce(1.0)
        assert a.strength <= 3.0

    def test_weaken(self):
        a = Atom(name="test", vector=np.ones(10, dtype=np.int8))
        a.weaken(0.5)
        assert a.strength == pytest.approx(0.5)


class TestAtomForge:
    @pytest.fixture
    def forge(self):
        return AtomForge(dimension=2048)

    def test_create_atom(self, forge):
        vec = forge.atom("chris")
        assert vec.shape == (2048,)
        assert set(np.unique(vec)).issubset({-1, 1})

    def test_deterministic(self, forge):
        a = forge.atom("hello")
        b = forge.atom("hello")
        assert np.array_equal(a, b)

    def test_different_atoms_different_vectors(self, forge):
        a = forge.atom("chris")
        b = forge.atom("glyphh")
        assert not np.array_equal(a, b)

    def test_similar_words_similar_vectors(self, forge):
        a = forge.atom("building")
        b = forge.atom("builds")
        sim = forge.similarity(a, b)
        assert sim > 0.3, f"Expected > 0.3, got {sim:.3f}"

    def test_unrelated_words_low_similarity(self, forge):
        a = forge.atom("chris")
        b = forge.atom("quantum")
        sim = forge.similarity(a, b)
        assert sim < 0.2, f"Expected < 0.2, got {sim:.3f}"

    def test_roles_exist(self, forge):
        for role in AtomForge.ROLES:
            vec = forge.role(role)
            assert vec.shape == (2048,)

    def test_role_unknown_raises(self, forge):
        with pytest.raises(ValueError, match="Unknown role"):
            forge.role("nonexistent")

    def test_bind_unbind_identity(self, forge):
        a = forge.atom("chris")
        b = forge.atom("name")
        bound = forge.bind(a, b)
        recovered = forge.unbind(bound, a)
        sim = forge.similarity(recovered, b)
        assert sim > 0.99, f"Expected > 0.99, got {sim:.3f}"

    def test_bundle(self, forge):
        a = forge.atom("chris")
        b = forge.atom("builds")
        c = forge.atom("glyphh")
        bundled = forge.bundle([a, b, c])
        assert bundled.shape == (2048,)
        # Bundle should be somewhat similar to each component
        for vec in [a, b, c]:
            sim = forge.similarity(bundled, vec)
            assert sim > 0.1

    def test_nearest_cleanup(self, forge):
        forge.atom("chris")
        forge.atom("glyphh")
        forge.atom("hdc")

        # Create a noisy version of "chris"
        chris = forge.atom("chris")
        noise = np.random.default_rng(0).choice([-1, 1], size=2048).astype(np.int8)
        noisy = np.where(np.random.default_rng(1).random(2048) > 0.2, chris, noise)
        noisy = noisy.astype(np.int8)

        matches = forge.nearest(noisy, top_k=3)
        assert len(matches) >= 1
        assert matches[0][0].name == "chris"

    def test_count(self, forge):
        initial = forge.count  # roles are pre-seeded
        forge.atom("test1")
        forge.atom("test2")
        assert forge.count == initial + 2

    def test_reinforce_on_access(self, forge):
        forge.atom("test")
        initial_strength = forge.get("test").strength
        forge.atom("test")  # second access should reinforce
        assert forge.get("test").strength > initial_strength

    def test_persistence(self, tmp_path, forge):
        forge.atom("chris")
        forge.atom("glyphh")
        forge.save(tmp_path)

        forge2 = AtomForge(dimension=2048)
        forge2.load(tmp_path)
        assert forge2.has("chris")
        assert forge2.has("glyphh")

        # Vectors should match
        sim = forge2.similarity(forge2.atom("chris"), forge.atom("chris"))
        assert sim > 0.99

    def test_all_atoms_filter(self, forge):
        forge.atom("chris", kind="entity")
        forge.atom("builds", kind="relation")
        entities = forge.all_atoms(kind="entity")
        assert any(a.name == "chris" for a in entities)
        assert not any(a.name == "builds" for a in entities)
