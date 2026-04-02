"""
Tests for glyphh.memory.binding — structured facts from atoms.
"""

import numpy as np
import pytest

from glyphh.memory.atom import AtomForge
from glyphh.memory.binding import Fact, FactStore


class TestFact:
    def test_properties(self):
        f = Fact(
            roles={"subject": "chris", "relation": "builds", "object": "glyphh"},
            vector=np.ones(10, dtype=np.int8),
        )
        assert f.subject == "chris"
        assert f.relation == "builds"
        assert f.object == "glyphh"

    def test_reinforce(self):
        f = Fact(
            roles={"subject": "x", "relation": "y", "object": "z"},
            vector=np.ones(10, dtype=np.int8),
        )
        initial = f.strength
        f.reinforce()
        assert f.strength > initial


class TestFactStore:
    @pytest.fixture
    def forge(self):
        return AtomForge(dimension=2048)

    @pytest.fixture
    def store(self, forge):
        return FactStore(forge)

    def test_teach(self, store):
        fact = store.teach("chris", "builds", "glyphh")
        assert store.count == 1
        assert fact.subject == "chris"
        assert fact.relation == "builds"
        assert fact.object == "glyphh"
        assert fact.vector.shape == (2048,)

    def test_teach_multiple(self, store):
        store.teach("chris", "builds", "glyphh")
        store.teach("glyphh", "uses", "hdc")
        store.teach("hdc", "is_a", "computing_paradigm")
        assert store.count == 3

    def test_query_by_subject(self, store):
        store.teach("chris", "builds", "glyphh")
        store.teach("alice", "builds", "something_else")
        results = store.query(subject="chris")
        assert len(results) >= 1
        assert results[0][0].subject == "chris"

    def test_query_by_subject_and_relation(self, store):
        store.teach("chris", "builds", "glyphh")
        store.teach("chris", "likes", "hdc")
        results = store.query(subject="chris", relation="builds")
        assert len(results) >= 1
        # The "builds" fact should score higher
        top_fact = results[0][0]
        assert top_fact.relation == "builds"

    def test_query_object(self, store):
        store.teach("chris", "builds", "glyphh")
        results = store.query_object("chris", "builds")
        assert len(results) >= 1
        # "glyphh" should be among the results
        names = [name for name, _ in results]
        assert "glyphh" in names

    def test_query_empty_store(self, store):
        results = store.query(subject="chris")
        assert results == []

    def test_infer_direct(self, store):
        store.teach("chris", "builds", "glyphh")
        results = store.infer("chris", "builds", depth=1)
        assert len(results) >= 1
        assert any(obj == "glyphh" for obj, _, _ in results)

    def test_infer_chain(self, store):
        """Test transitive inference: chris→builds→glyphh, glyphh→uses→hdc
        Therefore chris→uses→hdc (via chain)."""
        store.teach("chris", "builds", "glyphh")
        store.teach("glyphh", "uses", "hdc")

        results = store.infer("chris", "uses", depth=2)
        # Should find "hdc" via the chain
        found_hdc = any(obj == "hdc" for obj, _, _ in results)
        if not found_hdc:
            # Print debug info
            print(f"Infer results: {results}")
            connections = store._get_connections("chris")
            print(f"Connections from chris: {connections}")
        assert found_hdc, f"Expected to infer 'hdc' via chain, got: {results}"

    def test_infer_chain_with_derivation(self, store):
        """Verify the chain description is returned."""
        store.teach("chris", "builds", "glyphh")
        store.teach("glyphh", "uses", "hdc")

        results = store.infer("chris", "uses", depth=2)
        hdc_results = [(obj, score, chain) for obj, score, chain in results if obj == "hdc"]
        assert len(hdc_results) >= 1
        _, _, chain = hdc_results[0]
        assert len(chain) == 2  # two steps

    def test_extra_roles(self, store):
        fact = store.teach("chris", "builds", "glyphh", time="2026", location="home")
        assert fact.roles["time"] == "2026"
        assert fact.roles["location"] == "home"

    def test_persistence(self, tmp_path, forge):
        store1 = FactStore(forge)
        store1.teach("chris", "builds", "glyphh")
        store1.teach("glyphh", "uses", "hdc")
        store1.save(tmp_path)

        forge2 = AtomForge(dimension=2048)
        store2 = FactStore(forge2)
        store2.load(tmp_path)

        assert store2.count == 2
        facts = store2.facts
        subjects = [f.subject for f in facts]
        assert "chris" in subjects
        assert "glyphh" in subjects


class TestTeacher:
    @pytest.fixture
    def teacher(self):
        from glyphh.memory.teacher import Teacher
        forge = AtomForge(dimension=2048)
        facts = FactStore(forge)
        return Teacher(forge, facts), facts

    def test_parse_is_a(self, teacher):
        t, facts = teacher
        result = t.parse("Chris is a user")
        assert len(result) == 1
        assert result[0].subject == "chris"
        assert result[0].relation == "is_a"
        assert result[0].object == "user"

    def test_parse_svo(self, teacher):
        t, facts = teacher
        result = t.parse("Chris builds Glyphh")
        assert len(result) == 1
        assert result[0].subject == "chris"
        assert result[0].relation == "build"
        assert result[0].object == "glyphh"

    def test_parse_is(self, teacher):
        t, facts = teacher
        result = t.parse("HDC is cool")
        assert len(result) == 1
        assert result[0].subject == "hdc"
        assert result[0].relation == "is"
        assert result[0].object == "cool"

    def test_parse_has(self, teacher):
        t, facts = teacher
        result = t.parse("Chris's name is Christopher")
        assert len(result) == 1
        assert result[0].subject == "chris"
        assert result[0].relation == "name"
        assert result[0].object == "christopher"

    def test_parse_called(self, teacher):
        t, facts = teacher
        result = t.parse("The AI is called Ada")
        assert len(result) == 1
        assert result[0].relation == "named"

    def test_parse_unknown_returns_empty(self, teacher):
        t, facts = teacher
        result = t.parse("supercalifragilistic")
        assert result == []

    def test_parse_adds_to_factstore(self, teacher):
        t, facts = teacher
        t.parse("Chris builds Glyphh")
        assert facts.count == 1

    def test_llm_response_parsing(self, teacher):
        from glyphh.memory.teacher import Teacher
        response = '''Here are the facts:
[{"subject": "monetary_policy", "relation": "affects", "object": "inflation"},
 {"subject": "interest_rates", "relation": "influence", "object": "inflation"}]'''
        result = Teacher._parse_llm_response(response)
        assert len(result) == 2
        assert result[0]["subject"] == "monetary_policy"

    def test_llm_response_parsing_invalid(self, teacher):
        from glyphh.memory.teacher import Teacher
        result = Teacher._parse_llm_response("no json here")
        assert result == []
