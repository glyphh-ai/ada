"""
AdaCognitive — Ada's cognitive infrastructure as a single SDK class.

Any model imports this to get memory, dreaming, routing, and confidence
gating. The model brings domain-specific recall (encoder, intent,
exemplars). Ada brings the cognitive loop.

Usage:
    from glyphh.memory import AdaCognitive

    ada = AdaCognitive()
    ada.absorb("my name is chris")
    result = ada.process("what is my name?")
    # result.gate == "DONE", result.facts has the match,
    # result.prompt is ready for LLM streaming
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop
from glyphh.memory.glyph_dream import GlyphDreamLoop, Insight
from glyphh.memory.cognitive_glyph import CognitiveGlyph, Action, CognitiveState
from glyphh.memory.ada_conversation import Conversation

logger = logging.getLogger(__name__)


# ── Default system prompt ────────────────────────────────────────────────────

ADA_SYSTEM_PROMPT = """\
You are Ada. 1 sentence max. Use ONLY facts given. Never guess.
"my" = user's things. "your" = user's things."""


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class CognitiveResult:
    """Everything a consumer needs from a single process() call."""
    state: CognitiveState
    gate: str                                    # "DONE" | "ASK"
    facts: list[tuple[str, str, float]]          # (content, speaker, similarity)
    prompt: str                                  # ready-to-stream LLM prompt


# ── Sentence splitting ───────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Returns at least one entry."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


# ── AdaCognitive ─────────────────────────────────────────────────────────────

class AdaCognitive:
    """Ada's cognitive infrastructure. One class, any model imports it.

    Owns: ThoughtGlyphSpace, GlyphCognitiveLoop, GlyphDreamLoop,
          CognitiveGlyph, Conversation.

    Args:
        system_prompt: LLM system prompt.
        recall_threshold: Minimum similarity for DONE gate.
        thought_space: Pre-built ThoughtGlyphSpace (or creates one).
        localized_interval: DreamLoop localized cycle interval (seconds).
        deep_interval: DreamLoop deep cycle interval (seconds).
        max_conversation_turns: Number of turns to keep in context.
    """

    def __init__(
        self,
        system_prompt: str = ADA_SYSTEM_PROMPT,
        recall_threshold: float = 0.35,
        thought_space: ThoughtGlyphSpace | None = None,
        localized_interval: float = 3.0,
        deep_interval: float = 30.0,
        max_conversation_turns: int = 3,
    ) -> None:
        self._recall_threshold = recall_threshold
        self._system_prompt = system_prompt

        # Core components
        self._space = thought_space or ThoughtGlyphSpace()
        self._glyph_loop = GlyphCognitiveLoop(self._space)
        self._cognitive = CognitiveGlyph(thought_space=self._space)
        self._conversation = Conversation(
            system_prompt=system_prompt,
            max_turns=max_conversation_turns,
        )

        # Dream loop (created lazily or on start_dreaming)
        self._localized_interval = localized_interval
        self._deep_interval = deep_interval
        self._dream: GlyphDreamLoop | None = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def thought_space(self) -> ThoughtGlyphSpace:
        return self._space

    @property
    def cognitive(self) -> CognitiveGlyph:
        return self._cognitive

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def glyph_loop(self) -> GlyphCognitiveLoop:
        return self._glyph_loop

    # ── Core API ──────────────────────────────────────────────────────────

    def absorb(self, text: str, speaker: str = "incoming") -> "StoredThought | None":
        """Absorb input sentence by sentence as thought glyphs.

        Returns the last StoredThought created (or None if all were
        duplicates / too short).
        """
        last = None
        for sentence in _split_sentences(text):
            if len(sentence) < 2:
                continue
            result = self._space.absorb(sentence, speaker=speaker)
            if result is not None:
                last = result
        return last

    def process(self, text: str) -> CognitiveResult:
        """Route + absorb + recall + gate + build prompt.

        Full pipeline: cognitive routing determines the action, then
        the confidence gate determines DONE/ASK, and finally the
        LLM prompt is built with the right context.

        Args:
            text: User input.

        Returns:
            CognitiveResult with state, gate, facts, and prompt.
        """
        state = self._cognitive.process(text)
        self.absorb(text)

        # Recall with confidence gate
        gate, facts = self.recall(text)

        # Override gate for non-RECALL actions
        if state.action == Action.CONTRADICT:
            gate, facts = self.recall(text)
        elif state.action != Action.RECALL:
            gate, facts = "ASK", []

        prompt = self.build_prompt(text, state, gate, facts)

        # Nudge dream loop
        if self._dream is not None:
            self._dream.notify_absorb()
            if not self._dream._running and self._space.count > 0:
                self._dream.start()

        return CognitiveResult(state=state, gate=gate, facts=facts, prompt=prompt)

    def recall(
        self,
        text: str,
        top_k: int = 5,
    ) -> tuple[str, list[tuple[str, str, float]]]:
        """Recall memories with confidence gate.

        Returns:
            (gate, facts) where gate is "DONE" or "ASK" and facts
            is a list of (content, speaker, similarity) tuples.
        """
        results = self._space.recall(text, top_k=top_k, speaker="incoming")

        if not results or results[0].global_similarity < self._recall_threshold:
            return "ASK", []

        facts = []
        for r in results:
            if r.global_similarity < self._recall_threshold:
                break
            facts.append((r.thought.content, r.thought.speaker, r.global_similarity))

        return "DONE", facts

    def build_prompt(
        self,
        text: str,
        state: CognitiveState,
        gate: str = "ASK",
        facts: list[tuple[str, str, float]] | None = None,
    ) -> str:
        """Build minimal LLM prompt from cognitive state + recall.

        DONE: "Facts: X. Y. Z. -> answer in 1 sentence"
        ASK:  "-> say I don't know"
        STORE/FEEL/etc: "-> acknowledge in 1 sentence"
        """
        if state.action == Action.RECALL:
            if gate == "DONE" and facts:
                fact_lines = "\n".join(f"- {content}" for content, _, _ in facts[:3])
                injection = f"You remember:\n{fact_lines}\nAnswer using ONLY these facts."
            else:
                injection = "Say: I don't know."
        elif state.action == Action.STORE:
            injection = "Say: Got it."
        elif state.action == Action.FEEL:
            injection = "Respond with empathy."
        elif state.action == Action.CONTRADICT:
            if facts:
                fact_lines = "\n".join(f"- {content}" for content, _, _ in facts[:2])
                injection = f"User corrected you. You remember:\n{fact_lines}"
            else:
                injection = "User corrected you. Say: OK."
        elif state.action == Action.WONDER:
            injection = "Ask a follow-up."
        else:
            injection = "Respond briefly."

        return self._conversation.build_prompt(text, recall=injection)

    def add_response(self, user_text: str, response: str) -> None:
        """Record a conversation turn and absorb Ada's response."""
        self._conversation.add_turn(user_text, response)
        self.absorb(response, speaker="outgoing")

    # ── Dream API ─────────────────────────────────────────────────────────

    def _ensure_dream(self) -> GlyphDreamLoop:
        if self._dream is None:
            self._dream = GlyphDreamLoop(
                self._glyph_loop,
                localized_interval=self._localized_interval,
                deep_interval=self._deep_interval,
            )
            self._dream.set_cognitive(self._cognitive)
        return self._dream

    def start_dreaming(self) -> None:
        self._ensure_dream().start()

    def stop_dreaming(self) -> None:
        if self._dream is not None:
            self._dream.stop()

    def drain_insights(self) -> list[Insight]:
        if self._dream is None:
            return []
        return self._dream.drain_insights()

    def dream_stats(self) -> dict:
        if self._dream is None:
            return {}
        return self._dream.stats

    @property
    def is_dreaming(self) -> bool:
        return self._dream is not None and self._dream.is_running

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """Save persistent state to disk."""
        if path and self._glyph_loop is not None:
            self._glyph_loop.save(path)

    def reset(self) -> None:
        """Clear all memory and conversation state."""
        if self._dream is not None:
            self._dream.stop()
            self._dream = None
        self._space.clear()
        self._conversation.clear()
