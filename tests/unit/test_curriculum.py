"""
Tests for Ada's curriculum — learning language from the ground up.

Level 0: sounds exist (bare atoms)
Level 1: shape goes with sound (pairs)
Level 2: sounds combine into words (composition)
Level 3: word goes with experience (pairing)
Level 4: things connect (triples — only after enough experience)
"""

import pytest

from glyphh.memory.atom import AtomForge
from glyphh.memory.binding import FactStore
from glyphh.memory.teacher import Teacher


@pytest.fixture
def ada():
    """A fresh Ada brain."""
    forge = AtomForge(dimension=2048)
    facts = FactStore(forge)
    teacher = Teacher(forge, facts)
    return forge, facts, teacher


class TestLevel0Sounds:
    """Sounds exist. That's it. No meaning, no categories."""

    def test_sounds_loaded(self, ada):
        forge, facts, teacher = ada
        result = teacher.learn_lesson("01_sounds")
        assert result.atoms_created > 20

    def test_sound_atoms_exist(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("01_sounds")
        assert forge.has("ah")
        assert forge.has("buh")
        assert forge.has("ee")
        assert forge.has("shh")

    def test_no_facts_created(self, ada):
        """Level 0 creates atoms, not facts."""
        forge, facts, teacher = ada
        result = teacher.learn_lesson("01_sounds")
        assert result.facts_created == 0
        assert result.pairs_created == 0


class TestLevel1Letters:
    """Shape goes with sound. Pure pairing."""

    def test_letters_loaded(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("01_sounds")
        result = teacher.learn_lesson("02_letters")
        assert result.pairs_created == 26

    def test_letter_sound_association(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("01_sounds")
        teacher.learn_lesson("02_letters")
        # 'a' is associated with 'ay'
        results = facts.query_object("a", "associated_with")
        names = [name for name, _ in results]
        assert "ay" in names

    def test_all_letters_have_atoms(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("02_letters")
        for letter in "abcdefghijklmnopqrstuvwxyz":
            assert forge.has(letter), f"Missing atom: {letter}"


class TestLevel2FirstWords:
    """Sounds combine into words. Not spelling — sound composition."""

    def test_words_loaded(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("01_sounds")
        result = teacher.learn_lesson("03_first_words")
        assert result.compositions_created > 10

    def test_word_atoms_created(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("01_sounds")
        teacher.learn_lesson("03_first_words")
        assert forge.has("mama")
        assert forge.has("cat")
        assert forge.has("dog")

    def test_word_composition_stored(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("01_sounds")
        teacher.learn_lesson("03_first_words")
        # "mama" is composed of sounds
        results = facts.query_object("mama", "composed_of")
        assert len(results) >= 1

    def test_word_sound_parts(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("01_sounds")
        teacher.learn_lesson("03_first_words")
        # "cat" has individual sound components
        results = facts.query_object("cat", "sound_0")
        names = [name for name, _ in results]
        assert "kuh" in names


class TestLevel3Pointing:
    """Word goes with experience. Still just pairing."""

    def test_pointing_loaded(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("03_first_words")
        result = teacher.learn_lesson("04_pointing")
        assert result.pairs_created > 15

    def test_word_experience_association(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("03_first_words")
        teacher.learn_lesson("04_pointing")
        # "cat" is associated with "soft"
        results = facts.query_object("cat", "associated_with")
        names = [name for name, _ in results]
        assert "soft" in names or "furry" in names or "small" in names


class TestLevel4Connections:
    """NOW triples. Only because we have enough experience."""

    def test_connections_loaded(self, ada):
        forge, facts, teacher = ada
        result = teacher.learn_lesson("05_connections")
        assert result.facts_created > 10

    def test_triple_query(self, ada):
        forge, facts, teacher = ada
        teacher.learn_lesson("05_connections")
        results = facts.query_object("cat", "is")
        names = [name for name, _ in results]
        assert "soft" in names or "furry" in names


class TestFullCurriculum:
    def test_all_lessons_ordered(self, ada):
        forge, facts, teacher = ada
        lessons = ["01_sounds", "02_letters", "03_first_words", "04_pointing", "05_connections"]
        for lesson in lessons:
            teacher.learn_lesson(lesson)

        # After full curriculum, Ada knows a lot
        assert forge.count > 80
        assert facts.count > 50

    def test_cross_level_knowledge(self, ada):
        """After learning sounds, letters, and words:
        Ada knows 'a' goes with 'ay' (letter level)
        and 'kuh + ah + tuh = cat' (word level)
        and 'cat' goes with 'soft' (pointing level)"""
        forge, facts, teacher = ada
        for lesson in ["01_sounds", "02_letters", "03_first_words", "04_pointing"]:
            teacher.learn_lesson(lesson)

        # Letter → sound
        results = facts.query_object("a", "associated_with")
        assert len(results) >= 1

        # Word exists
        assert forge.has("cat")

        # Word → experience
        results = facts.query_object("cat", "associated_with")
        assert len(results) >= 1

    def test_persistence(self, ada, tmp_path):
        forge, facts, teacher = ada
        for lesson in ["01_sounds", "02_letters", "03_first_words"]:
            teacher.learn_lesson(lesson)

        forge.save(tmp_path)
        facts.save(tmp_path)

        forge2 = AtomForge(dimension=2048)
        facts2 = FactStore(forge2)
        forge2.load(tmp_path)
        facts2.load(tmp_path)

        assert forge2.has("mama")
        assert forge2.has("cat")
        assert facts2.count > 0
