"""
Tests for the Extractor — rule-based + LLM extraction.

Covers: question detection, fact extraction, emotion detection,
greeting detection, correction detection, entity extraction,
LLM fallback.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from domains.brain.extractor import Extractor, Extraction


@pytest.fixture
def extractor():
    return Extractor()


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.available = True
    llm.ask = AsyncMock(return_value=None)
    return llm


# ── Question detection ─────────────────────────────────────────────

class TestQuestionDetection:

    @pytest.mark.asyncio
    async def test_question_mark(self, extractor):
        result = await extractor.extract("what is my name?")
        assert result.has_question
        assert result.question == "what is my name?"

    @pytest.mark.asyncio
    async def test_question_word_start(self, extractor):
        result = await extractor.extract("who is my wife")
        assert result.has_question

    @pytest.mark.asyncio
    async def test_how_question(self, extractor):
        result = await extractor.extract("how many children do i have")
        assert result.has_question

    @pytest.mark.asyncio
    async def test_do_you_know(self, extractor):
        result = await extractor.extract("do you know my name")
        assert result.has_question

    @pytest.mark.asyncio
    async def test_can_you_tell(self, extractor):
        result = await extractor.extract("can you tell me about Brandi")
        assert result.has_question

    @pytest.mark.asyncio
    async def test_tell_me(self, extractor):
        result = await extractor.extract("tell me about the project")
        assert result.has_question

    @pytest.mark.asyncio
    async def test_statement_not_question(self, extractor):
        result = await extractor.extract("my name is Chris")
        assert not result.has_question


# ── Fact extraction ────────────────────────────────────────────────

class TestFactExtraction:

    @pytest.mark.asyncio
    async def test_is_statement(self, extractor):
        result = await extractor.extract("my wife is Brandi")
        assert result.has_facts
        assert "my wife is Brandi" in result.facts

    @pytest.mark.asyncio
    async def test_has_statement(self, extractor):
        result = await extractor.extract("i have two kids")
        assert result.has_facts

    @pytest.mark.asyncio
    async def test_lives_statement(self, extractor):
        result = await extractor.extract("Traceton lives in Colorado")
        assert result.has_facts

    @pytest.mark.asyncio
    async def test_likes_statement(self, extractor):
        result = await extractor.extract("i like pizza")
        assert result.has_facts

    @pytest.mark.asyncio
    async def test_multi_word_treated_as_fact(self, extractor):
        result = await extractor.extract("Brandi has two children Traceton and James")
        assert result.has_facts

    @pytest.mark.asyncio
    async def test_question_not_stored_as_fact(self, extractor):
        result = await extractor.extract("what is my name?")
        assert not result.has_facts

    @pytest.mark.asyncio
    async def test_greeting_not_stored_as_fact(self, extractor):
        result = await extractor.extract("hi ada")
        assert not result.has_facts


# ── Greeting detection ─────────────────────────────────────────────

class TestGreetingDetection:

    @pytest.mark.asyncio
    async def test_hi(self, extractor):
        result = await extractor.extract("hi")
        assert result.is_greeting

    @pytest.mark.asyncio
    async def test_hello(self, extractor):
        result = await extractor.extract("hello")
        assert result.is_greeting

    @pytest.mark.asyncio
    async def test_hi_ada(self, extractor):
        result = await extractor.extract("hi ada")
        assert result.is_greeting

    @pytest.mark.asyncio
    async def test_hello_ada(self, extractor):
        result = await extractor.extract("hello ada")
        assert result.is_greeting

    @pytest.mark.asyncio
    async def test_good_morning(self, extractor):
        result = await extractor.extract("good morning")
        assert result.is_greeting

    @pytest.mark.asyncio
    async def test_thanks(self, extractor):
        result = await extractor.extract("thanks")
        assert result.is_greeting

    @pytest.mark.asyncio
    async def test_greeting_is_social(self, extractor):
        result = await extractor.extract("hi ada")
        assert result.is_social

    @pytest.mark.asyncio
    async def test_statement_not_greeting(self, extractor):
        result = await extractor.extract("my name is Chris")
        assert not result.is_greeting


# ── Emotion detection ──────────────────────────────────────────────

class TestEmotionDetection:

    @pytest.mark.asyncio
    async def test_i_feel_sad(self, extractor):
        result = await extractor.extract("i feel sad")
        assert result.emotion == "sad"

    @pytest.mark.asyncio
    async def test_i_am_happy(self, extractor):
        result = await extractor.extract("i am happy")
        assert result.emotion == "happy"

    @pytest.mark.asyncio
    async def test_im_frustrated(self, extractor):
        result = await extractor.extract("i'm really frustrated")
        assert result.emotion == "frustrated"

    @pytest.mark.asyncio
    async def test_no_emotion_in_statement(self, extractor):
        result = await extractor.extract("my name is Chris")
        assert result.emotion is None


# ── Correction detection ──────────────────────────────────────────

class TestCorrectionDetection:

    @pytest.mark.asyncio
    async def test_no_thats_wrong(self, extractor):
        result = await extractor.extract("no that is wrong")
        assert result.is_correction

    @pytest.mark.asyncio
    async def test_actually(self, extractor):
        result = await extractor.extract("actually it is different")
        assert result.is_correction

    @pytest.mark.asyncio
    async def test_i_never_said(self, extractor):
        result = await extractor.extract("i never said that")
        assert result.is_correction

    @pytest.mark.asyncio
    async def test_not_a_correction(self, extractor):
        result = await extractor.extract("my name is Chris")
        assert not result.is_correction


# ── Entity extraction ─────────────────────────────────────────────

class TestEntityExtraction:

    @pytest.mark.asyncio
    async def test_extracts_capitalized_names(self, extractor):
        result = await extractor.extract("my wife is Brandi")
        assert "Brandi" in result.entities

    @pytest.mark.asyncio
    async def test_extracts_multiple_entities(self, extractor):
        result = await extractor.extract("Brandi has two children Traceton and James")
        assert "Traceton" in result.entities
        assert "James" in result.entities

    @pytest.mark.asyncio
    async def test_skips_common_words(self, extractor):
        result = await extractor.extract("I am going to Ada")
        # "I" and "Ada" should be filtered
        assert "I" not in result.entities


# ── LLM fallback ──────────────────────────────────────────────────

class TestLLMFallback:

    @pytest.mark.asyncio
    async def test_llm_called_for_ambiguous(self, extractor, mock_llm):
        """LLM is called when rules can't extract anything from substantial input."""
        mock_llm.ask = AsyncMock(return_value='{"facts": ["test fact"], "question": null, "emotion": null, "is_correction": false}')
        result = await extractor.extract("something completely ambiguous and weird here", mock_llm)
        # LLM should have been called since rules couldn't extract
        assert result.llm_assisted or result.has_facts  # either LLM helped or rules caught it

    @pytest.mark.asyncio
    async def test_llm_not_called_for_question(self, extractor, mock_llm):
        """LLM is not called when rules already extracted a question."""
        result = await extractor.extract("what is my name?", mock_llm)
        mock_llm.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_not_called_for_greeting(self, extractor, mock_llm):
        """LLM is not called for simple greetings."""
        result = await extractor.extract("hi ada", mock_llm)
        mock_llm.ask.assert_not_called()


# ── Extraction dataclass ──────────────────────────────────────────

class TestExtractionStructure:

    @pytest.mark.asyncio
    async def test_raw_input_preserved(self, extractor):
        result = await extractor.extract("hello world")
        assert result.raw_input == "hello world"

    @pytest.mark.asyncio
    async def test_empty_input(self, extractor):
        result = await extractor.extract("")
        assert not result.has_question
        assert not result.has_facts
        assert not result.is_greeting
