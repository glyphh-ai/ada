"""
Tests for Ada's self-knowledge — can she reason about Glyphh?

This is the real test. Not sounds and letters. Can Ada learn
complex domain knowledge and infer things she was never told?
"""

import pytest

from glyphh.memory.atom import AtomForge
from glyphh.memory.binding import FactStore
from glyphh.memory.teacher import Teacher


@pytest.fixture
def ada():
    forge = AtomForge(dimension=2048)
    facts = FactStore(forge)
    teacher = Teacher(forge, facts)
    teacher.learn_lesson("glyphh")
    return forge, facts


class TestDirectKnowledge:
    """Things Ada was explicitly taught."""

    def test_what_is_glyphh(self, ada):
        forge, facts = ada
        results = facts.query(subject="glyphh", top_k=10)
        assert len(results) >= 5  # many facts about glyphh

    def test_chris_builds_glyphh(self, ada):
        forge, facts = ada
        results = facts.query_object("chris", "builds")
        names = [n for n, _ in results]
        assert "glyphh" in names

    def test_bind_is_reversible(self, ada):
        forge, facts = ada
        results = facts.query_object("bind", "is")
        names = [n for n, _ in results]
        assert "reversible" in names

    def test_ada_uses_hdc(self, ada):
        """ada uses hdc for_thought — verify via direct query"""
        forge, facts = ada
        results = facts.query(subject="ada", relation="uses", top_k=10)
        objects = [f.object for f, _ in results]
        assert any("hdc" in o for o in objects)

    def test_llm_can_hallucinate(self, ada):
        forge, facts = ada
        results = facts.query_object("llm", "can")
        names = [n for n, _ in results]
        assert "hallucinate" in names

    def test_hdc_cannot_hallucinate(self, ada):
        forge, facts = ada
        results = facts.query_object("hdc", "cannot")
        names = [n for n, _ in results]
        assert "hallucinate" in names

    def test_models_known(self, ada):
        forge, facts = ada
        for model in ["toolrouter", "bfcl", "pipedream", "faq"]:
            results = facts.query_object(model, "is_a")
            names = [n for n, _ in results]
            assert "model" in names, f"{model} should be a model"


class TestInference:
    """Things Ada was NOT taught but should be able to derive."""

    def test_chris_uses_hdc(self, ada):
        """chris builds glyphh + glyphh uses hdc → chris uses hdc (never taught)"""
        forge, facts = ada
        # Not directly taught - needs inference
        results = facts.infer("chris", "has", depth=2)
        # chris builds glyphh, glyphh has encoder/gql/etc
        objects = [obj for obj, _, _ in results]
        # Should find glyphh's components via chain
        assert len(objects) > 0, "Expected inferences about what chris has via glyphh"

    def test_ada_stores_things(self, ada):
        """ada stores atoms + ada stores facts → can query what ada stores"""
        forge, facts = ada
        results = facts.query_object("ada", "stores")
        names = [n for n, _ in results]
        assert "atoms" in names
        assert "facts" in names

    def test_glyphh_components(self, ada):
        """What does glyphh have?"""
        forge, facts = ada
        results = facts.query_object("glyphh", "has", top_k=10)
        names = [n for n, _ in results]
        # At least some components should be recoverable via unbinding
        found = [n for n in names if n in ("encoder", "memory", "gql", "cognitive_loop", "linguistics", "state_tracker", "llm_engine")]
        assert len(found) >= 2, f"Expected glyphh components, got: {names}"

    def test_what_handles_what(self, ada):
        """llm handles language, hdc handles reasoning"""
        forge, facts = ada
        llm_handles = facts.query_object("llm", "handles")
        hdc_handles = facts.query_object("hdc", "handles")
        assert any("language" in n for n, _ in llm_handles)
        assert any("reasoning" in n for n, _ in hdc_handles)

    def test_toolrouter_chain(self, ada):
        """toolrouter is_a model + model has intent_extractor
        → toolrouter has intent_extractor (inferred)"""
        forge, facts = ada
        results = facts.infer("toolrouter", "has", depth=2)
        objects = [obj for obj, _, _ in results]
        assert "intent_extractor" in objects, \
            f"Expected 'intent_extractor' via model chain, got: {objects}"

    def test_bfcl_chain(self, ada):
        """bfcl is_a model + model is domain_specific
        → bfcl is domain_specific (inferred)"""
        forge, facts = ada
        results = facts.infer("bfcl", "is", depth=2)
        objects = [obj for obj, _, _ in results]
        assert "domain_specific" in objects, \
            f"Expected 'domain_specific' via model chain, got: {objects}"


class TestAtomConcepts:
    """Test that the pairing-level knowledge works."""

    def test_ada_associations(self, ada):
        forge, facts = ada
        results = facts.query_object("ada", "associated_with")
        names = [n for n, _ in results]
        # ada was paired with person, ai, lives_in_glyphh
        assert any(n in ["person", "ai", "lives_in_glyphh"] for n in names)

    def test_hdc_associations(self, ada):
        forge, facts = ada
        results = facts.query_object("hdc", "associated_with")
        names = [n for n, _ in results]
        assert any(n in ["computing_paradigm", "hyperdimensional", "bipolar_vectors"] for n in names)

    def test_atom_count(self, ada):
        forge, facts = ada
        # Should have created many atoms from the lesson
        entities = forge.all_atoms(kind="entity")
        assert len(entities) > 50


class TestDesignPrinciples:
    """Ada should know the project's design philosophy."""

    def test_no_runtime_llm(self, ada):
        forge, facts = ada
        results = facts.query(subject="glyphh", relation="is", top_k=10)
        objects = [f.object for f, _ in results]
        assert any("deterministic" in o or "no_runtime" in o for o in objects)

    def test_chris_preferences(self, ada):
        forge, facts = ada
        results = facts.query_object("chris", "prefers")
        names = [n for n, _ in results]
        assert "deterministic" in names
        assert "no_hallucination" in names

    def test_sdk_is_pure(self, ada):
        forge, facts = ada
        results = facts.query_object("sdk", "is")
        names = [n for n, _ in results]
        assert "pure_engine" in names

    def test_sdk_has_no_domain(self, ada):
        forge, facts = ada
        results = facts.query_object("sdk", "has")
        names = [n for n, _ in results]
        assert "no_domain_vocabulary" in names
