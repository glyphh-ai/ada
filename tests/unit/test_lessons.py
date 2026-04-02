"""
Tests for the lesson system — teaching Ada from .teach files.
"""

import pytest

from glyphh.memory.atom import AtomForge
from glyphh.memory.binding import FactStore
from glyphh.memory.teacher import Teacher


class TestLessonFormat:
    @pytest.fixture
    def system(self):
        forge = AtomForge(dimension=2048)
        facts = FactStore(forge)
        teacher = Teacher(forge, facts)
        return forge, facts, teacher

    def test_level0_atoms(self, tmp_path, system):
        forge, facts, teacher = system
        lesson = tmp_path / "test.teach"
        lesson.write_text("# just atoms\nfoo\nbar\nbaz\n")
        result = teacher.learn_file(lesson)
        assert result.atoms_created == 3
        assert result.pairs_created == 0
        assert result.facts_created == 0
        assert forge.has("foo")
        assert forge.has("bar")

    def test_level1_pairs(self, tmp_path, system):
        forge, facts, teacher = system
        lesson = tmp_path / "test.teach"
        lesson.write_text("a : ay\nb : buh\n")
        result = teacher.learn_file(lesson)
        assert result.pairs_created == 2
        assert forge.has("a")
        assert forge.has("ay")

    def test_level2_composition(self, tmp_path, system):
        forge, facts, teacher = system
        lesson = tmp_path / "test.teach"
        lesson.write_text("kuh + ah + tuh = cat\n")
        result = teacher.learn_file(lesson)
        assert result.compositions_created == 1
        assert forge.has("cat")
        assert forge.has("kuh")

    def test_level3_triples(self, tmp_path, system):
        forge, facts, teacher = system
        lesson = tmp_path / "test.teach"
        lesson.write_text("cat is furry\ndog is big\n")
        result = teacher.learn_file(lesson)
        assert result.facts_created == 2

    def test_mixed_levels(self, tmp_path, system):
        forge, facts, teacher = system
        lesson = tmp_path / "test.teach"
        lesson.write_text(
            "# Level 0\nah\nee\n"
            "# Level 1\na : ay\n"
            "# Level 2\nkuh + ah + tuh = cat\n"
            "# Level 3\ncat is furry\n"
        )
        result = teacher.learn_file(lesson)
        assert result.atoms_created == 2
        assert result.pairs_created == 1
        assert result.compositions_created == 1
        assert result.facts_created == 1

    def test_comments_and_blanks_ignored(self, tmp_path, system):
        forge, facts, teacher = system
        lesson = tmp_path / "test.teach"
        lesson.write_text("# comment\n\n  \n# another\nfoo\n")
        result = teacher.learn_file(lesson)
        assert result.atoms_created == 1

    def test_file_not_found(self, system):
        forge, facts, teacher = system
        with pytest.raises(FileNotFoundError):
            teacher.learn_file("/nonexistent/path.teach")

    def test_lesson_not_found(self, system):
        forge, facts, teacher = system
        with pytest.raises(FileNotFoundError, match="not found"):
            teacher.learn_lesson("nonexistent_lesson")


class TestAlphabet:
    """Test the original alphabet lesson (now stored as alphabet.teach)."""

    @pytest.fixture
    def system(self):
        forge = AtomForge(dimension=2048)
        facts = FactStore(forge)
        teacher = Teacher(forge, facts)
        return forge, facts, teacher

    def test_alphabet_loads(self, system):
        forge, facts, teacher = system
        result = teacher.learn_lesson("alphabet")
        assert result.total > 50

    def test_alphabet_query_order(self, system):
        forge, facts, teacher = system
        teacher.learn_lesson("alphabet")
        results = facts.query_object("d", "comes_before")
        names = [name for name, _ in results]
        assert "e" in names

    def test_alphabet_infer_chain(self, system):
        forge, facts, teacher = system
        teacher.learn_lesson("alphabet")
        results = facts.infer("d", "comes_before", depth=2)
        objects = [obj for obj, _, _ in results]
        assert "e" in objects
        assert "f" in objects
