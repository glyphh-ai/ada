"""
Input extraction — pulls structure from natural language.

Not classification. Extraction. What facts, questions, entities,
and emotions are IN this input?

Rule-based for obvious cases (fast, deterministic).
LLM for anything ambiguous (accurate, slower).

The extractor doesn't decide what to DO — it just tells you
what's there. The pipeline decides what to do with it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Question starters — if the input starts with one of these, it's a question
_Q_STARTERS = {
    "who", "what", "where", "when", "why", "how",
    "do", "does", "did", "is", "are", "was", "were",
    "can", "could", "will", "would", "should",
    "have", "has", "had",
    "tell", "explain", "describe",
}

# Correction signals
_CORRECTION_SIGNALS = {
    "no ", "no,", "nope", "wrong", "incorrect", "not right",
    "actually", "i meant", "i mean", "that's not",
    "thats not", "you're wrong", "youre wrong",
    "i never said", "i didn't say", "not what i said",
    "let me correct", "correction",
}

# Emotion words
_EMOTIONS = {
    "happy", "sad", "angry", "frustrated", "excited", "anxious",
    "nervous", "scared", "afraid", "worried", "stressed",
    "grateful", "thankful", "overwhelmed", "depressed",
    "furious", "annoyed", "disappointed", "proud", "lonely",
    "confused", "surprised", "hopeful", "jealous", "guilty",
    "ashamed", "disgusted", "love", "hate",
}

# Emotion patterns — "i feel X", "i am X", "i'm X"
_EMOTION_PATTERNS = [
    re.compile(r"\bi\s+(?:feel|am|'m)\s+(?:so\s+|really\s+|very\s+)?(\w+)", re.I),
    re.compile(r"\b(?:makes?\s+me|i'm\s+feeling)\s+(?:so\s+|really\s+)?(\w+)", re.I),
]

# Greeting patterns
_GREETINGS = {
    "hi", "hello", "hey", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "good night", "goodnight",
    "thanks", "thank you", "bye", "goodbye", "see you",
    "whats up", "what's up",
}


@dataclass
class Extraction:
    """What's in the input — facts, questions, entities, emotions."""

    raw_input: str

    # Extracted content
    facts: list[str] = field(default_factory=list)
    question: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    emotion: Optional[str] = None

    # Flags
    is_greeting: bool = False
    is_correction: bool = False

    # Whether LLM was needed for extraction
    llm_assisted: bool = False

    @property
    def has_question(self) -> bool:
        return self.question is not None

    @property
    def has_facts(self) -> bool:
        return len(self.facts) > 0

    @property
    def is_social(self) -> bool:
        """Greeting or thanks — no facts or questions, just social."""
        return self.is_greeting and not self.has_question and not self.has_facts


class Extractor:
    """Extracts structure from natural language input.

    Rule-based first (fast). Falls back to LLM for ambiguous inputs.
    """

    async def extract(self, input_text: str, llm: Any = None) -> Extraction:
        """Extract facts, questions, entities, emotions from input."""
        text = input_text.strip()
        result = Extraction(raw_input=text)

        if not text:
            return result

        text_lower = text.lower().strip()

        # ── Rule-based extraction ────────────────────────────

        # Greeting detection
        result.is_greeting = self._is_greeting(text_lower)

        # Correction detection
        result.is_correction = self._is_correction(text_lower)

        # Question detection
        result.question = self._extract_question(text, text_lower)

        # Emotion detection
        result.emotion = self._extract_emotion(text_lower)

        # Fact extraction — if not a question or greeting, try rules
        if not result.has_question and not result.is_greeting:
            result.facts = self._extract_facts(text)

        # Entity extraction — simple proper noun detection
        result.entities = self._extract_entities(text)

        # ── LLM fallback for ambiguous input ─────────────────
        # If we couldn't extract anything meaningful and input is
        # substantial, ask the LLM to help
        if (
            not result.has_question
            and not result.has_facts
            and not result.is_greeting
            and not result.emotion
            and not result.is_correction
            and len(text.split()) >= 3
            and llm and llm.available
        ):
            await self._llm_extract(text, result, llm)

        return result

    # ── Rule-based extractors ────────────────────────────────

    def _is_greeting(self, text_lower: str) -> bool:
        """Check if the input is a social greeting."""
        # Strip trailing punctuation for matching
        cleaned = text_lower.rstrip("!.?,")
        if cleaned in _GREETINGS:
            return True
        # "hi ada", "hello ada", "hey ada"
        for g in ("hi", "hello", "hey", "yo"):
            if cleaned.startswith(g + " ada") or cleaned == g + " ada":
                return True
        return False

    def _is_correction(self, text_lower: str) -> bool:
        """Check if the input is correcting previous information."""
        for signal in _CORRECTION_SIGNALS:
            if text_lower.startswith(signal) or f" {signal}" in text_lower:
                return True
        return False

    def _extract_question(self, text: str, text_lower: str) -> Optional[str]:
        """Extract the question from input, if any."""
        # Explicit question mark
        if "?" in text:
            return text

        # Starts with question word
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in _Q_STARTERS:
            return text

        # "do you know", "do you remember", "can you tell me"
        if re.match(r"(?:do you|can you|could you|would you)\s", text_lower):
            return text

        return None

    def _extract_emotion(self, text_lower: str) -> Optional[str]:
        """Extract emotion from input, if any."""
        for pattern in _EMOTION_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                word = match.group(1).lower()
                if word in _EMOTIONS:
                    return word

        # Direct emotion mentions: "that makes me sad"
        for emotion in _EMOTIONS:
            if f" {emotion}" in text_lower or text_lower.startswith(emotion):
                # Verify it's about the user, not a fact
                if re.search(rf"\bi\b.*\b{emotion}\b|\b{emotion}\b.*\bi\b", text_lower):
                    return emotion

        return None

    def _extract_facts(self, text: str) -> list[str]:
        """Extract factual statements from input.

        Returns the input text as a fact if it looks like a statement.
        The thought space handles the actual encoding and storage.
        """
        text_lower = text.lower().strip()

        # Short inputs without content — not facts
        words = text.split()
        if len(words) < 2:
            return []

        # If it contains assertion patterns, it's likely a fact
        # "X is Y", "my X is Y", "I have X", "X lives in Y", etc.
        # We don't need to parse the structure — just detect that
        # this is a statement and pass the whole thing to storage.
        fact_patterns = [
            r"\b(?:is|are|was|were)\b",       # "she is a doctor"
            r"\b(?:has|have|had)\b",           # "i have two kids"
            r"\b(?:lives?|works?|goes?)\b",    # "he lives in texas"
            r"\b(?:married|born|named)\b",     # "she married john"
            r"\bmy\s+\w+\s+is\b",             # "my name is chris"
            r"\b(?:likes?|loves?|hates?)\b",   # "i like pizza"
        ]

        for pattern in fact_patterns:
            if re.search(pattern, text_lower):
                return [text]

        # If nothing matched but it's a multi-word statement, treat as fact
        if len(words) >= 3:
            return [text]

        return []

    def _extract_entities(self, text: str) -> list[str]:
        """Extract proper nouns / entities from input.

        Simple capitalization heuristic — words that are capitalized
        mid-sentence are likely proper nouns.
        """
        entities = []
        words = text.split()
        for i, word in enumerate(words):
            # Skip first word (always capitalized) and common words
            if i == 0:
                continue
            cleaned = word.strip(".,!?;:'\"")
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in {
                "i", "ada", "a", "an", "the", "is", "am", "are",
            }:
                entities.append(cleaned)
        return entities

    # ── LLM fallback ─────────────────────────────────────────

    async def _llm_extract(
        self, text: str, result: Extraction, llm: Any
    ) -> None:
        """Use LLM to extract structure from ambiguous input."""
        prompt = (
            "Extract from this input. Reply with ONLY a JSON object.\n"
            f"Input: \"{text}\"\n\n"
            "Format: {\"facts\": [\"...\"], \"question\": \"...\" or null, "
            "\"emotion\": \"...\" or null, \"is_correction\": true/false}\n"
            "facts = new information the user is stating\n"
            "question = what the user is asking (null if not a question)\n"
            "emotion = emotion expressed (null if none)\n"
            "is_correction = whether user is correcting previous info"
        )
        response = await llm.ask(prompt, max_tokens=128)
        if not response:
            return

        try:
            import json
            # Try to parse JSON from response (LLM might wrap it)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(response)

            if data.get("facts"):
                result.facts = data["facts"]
            if data.get("question"):
                result.question = data["question"]
            if data.get("emotion"):
                result.emotion = data["emotion"]
            if data.get("is_correction"):
                result.is_correction = True

            result.llm_assisted = True
        except (json.JSONDecodeError, KeyError, TypeError):
            # LLM gave us garbage — treat input as a fact
            result.facts = [text]
            result.llm_assisted = True
