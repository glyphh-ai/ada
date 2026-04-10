"""
Ada's cognitive sub-agents — each action has its own brain.

Each sub-agent:
  - Has its own HDC model (exemplars encoded into the shared thought space)
  - Scores inputs against its exemplars for routing confidence
  - Has its own LLM prompt and response strategy
  - Controls its own hallucination policy

The LLMRouter breaks ties when HDC classification is ambiguous.

Architecture:
    CognitiveGlyph classifies (coarse) → Sub-agents refine → LLM tiebreaks
    Winner handles response with its own strategy + LLM prompt.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from glyphh.core.ops import cosine_similarity
from glyphh.memory.cognitive_glyph import Action, CognitiveState
from glyphh.memory.thought_space import ThoughtGlyphSpace

logger = logging.getLogger(__name__)


# ── Sub-agent HDC model ─────────────────────────────────────────────────

@dataclass
class AgentExemplar:
    """An encoded exemplar for a sub-agent's HDC model."""
    text: str
    vector: np.ndarray


class AgentModel:
    """Lightweight HDC model for a sub-agent.

    Each agent has its own exemplars encoded via the shared thought space
    encoder. Scoring an input against exemplars gives a routing confidence.
    """

    def __init__(self, thought_space: ThoughtGlyphSpace) -> None:
        self._encoder = thought_space.encoder
        self._exemplars: list[AgentExemplar] = []

    def add_exemplars(self, texts: list[str]) -> None:
        """Encode and store exemplar texts."""
        for text in texts:
            glyph = self._encoder.encode_thought(text, speaker="incoming")
            vec = glyph.metadata.get("_content_vector")
            if vec is not None:
                self._exemplars.append(AgentExemplar(
                    text=text,
                    vector=np.array(vec, dtype=np.float64),
                ))

    def score(self, input_text: str) -> float:
        """Score how well this input matches this agent's exemplars.

        Returns the max cosine similarity against any exemplar.
        """
        if not self._exemplars:
            return 0.0

        glyph = self._encoder.encode_thought(input_text, speaker="incoming")
        vec = glyph.metadata.get("_content_vector")
        if vec is None:
            return 0.0

        query_vec = np.array(vec, dtype=np.float64)
        best = 0.0
        for ex in self._exemplars:
            sim = float(cosine_similarity(query_vec, ex.vector))
            if sim > best:
                best = sim
        return max(0.0, best)


# ── Base sub-agent ──────────────────────────────────────────────────────

class SubAgent(ABC):
    """Base class for cognitive sub-agents."""

    action: Action
    name: str
    hallucination_gated: bool = False  # Only RecallAgent sets this True

    def __init__(self, thought_space: ThoughtGlyphSpace) -> None:
        self.model = AgentModel(thought_space)
        self.model.add_exemplars(self._exemplars())

    @abstractmethod
    def _exemplars(self) -> list[str]:
        """Exemplar texts for this agent's HDC model."""
        ...

    @abstractmethod
    async def respond(
        self,
        input_text: str,
        facts: list[tuple],
        confidence: float,
        memory_gate: str,
        llm: Any,
    ) -> str:
        """Generate a response for this cognitive action.

        Args:
            input_text: The user's input.
            facts: Recalled facts as (content, speaker, similarity) tuples.
            confidence: Overall confidence score (0-1).
            memory_gate: "DONE" or "ASK" from recall.
            llm: AdaLLM instance.
        """
        ...

    def score(self, input_text: str) -> float:
        """Score how well this input matches this agent."""
        return self.model.score(input_text)


# ── STORE agent ─────────────────────────────────────────────────────────

class StoreAgent(SubAgent):
    """Handles statements — acknowledges facts being stored."""

    action = Action.STORE
    name = "store"

    def _exemplars(self) -> list[str]:
        return [
            "my name is chris",
            "i like pizza",
            "we live in austin texas",
            "she is a doctor",
            "he works at the office",
            "my favorite color is blue",
            "i have two kids",
            "jake is seven years old",
            "i went to the store yesterday",
            "he was promoted to manager",
            "she joined the company last year",
            "the sky is blue",
            "i work as a software engineer",
        ]

    async def respond(self, input_text, facts, confidence, memory_gate, llm) -> str:
        if llm and llm.available:
            prompt = (
                f"The user told you a fact: \"{input_text}\"\n"
                f"Acknowledge naturally in 1 sentence. Be warm but brief. "
                f"Don't repeat the full fact back — just confirm you got it."
            )
            response = await llm.ask(prompt, max_tokens=64)
            if response:
                return response
        return "Got it."


# ── RECALL agent ────────────────────────────────────────────────────────

class RecallAgent(SubAgent):
    """Handles questions — searches memory, synthesizes answers.

    The ONLY agent with a hallucination gate. When confidence is low,
    the LLM is cut out entirely to prevent confabulation.
    """

    action = Action.RECALL
    name = "recall"
    hallucination_gated = True

    CONFIDENCE_THRESHOLD = 0.5

    def _exemplars(self) -> list[str]:
        return [
            "what is my name",
            "who am i",
            "who are you",
            "what do i like",
            "where do i live",
            "what does she do",
            "do you remember",
            "do you know my name",
            "what happened yesterday",
            "tell me about",
            "what time is it",
            "what is your name",
            "who is that",
            "how old is he",
            "when did i say that",
        ]

    async def respond(self, input_text, facts, confidence, memory_gate, llm) -> str:
        # ── Hallucination gate ──────────────────────────────
        # Below threshold: no LLM. Raw fact or "I don't know."
        if confidence < self.CONFIDENCE_THRESHOLD:
            if memory_gate == "DONE" and facts:
                return facts[0][0]
            return "I don't have information about that in my memory."

        # ── High confidence + grounded facts → LLM synthesizes
        if memory_gate == "DONE" and facts:
            if llm and llm.available:
                parts = [f"User asked: \"{input_text}\""]
                parts.append("\nFacts (from your memory — these are TRUE, use them):")
                for content, speaker, sim in facts[:5]:
                    parts.append(f"  - {content}")
                parts.append(
                    "\nRespond using ONLY these facts. 1-2 sentences. "
                    "Do not add information that isn't in the facts."
                )
                response = await llm.ask("\n".join(parts))
                if response:
                    return response
            return facts[0][0]

        # ── Partial match above threshold
        if facts and facts[0][2] >= self.CONFIDENCE_THRESHOLD:
            if llm and llm.available:
                parts = [f"User asked: \"{input_text}\""]
                parts.append("\nPossibly relevant memories:")
                for content, speaker, sim in facts[:3]:
                    parts.append(f"  - {content}")
                parts.append(
                    "\nRespond as Ada. Use these if relevant, "
                    "say what you're unsure about. 1-2 sentences."
                )
                response = await llm.ask("\n".join(parts))
                if response:
                    return response

        return "I don't have information about that in my memory."


# ── FEEL agent ──────────────────────────────────────────────────────────

class FeelAgent(SubAgent):
    """Handles emotional input — responds with empathy."""

    action = Action.FEEL
    name = "feel"

    def _exemplars(self) -> list[str]:
        return [
            "i am happy",
            "i feel sad",
            "i am angry",
            "i love that",
            "i hate this",
            "i am scared",
            "that makes me worried",
            "i am so frustrated",
            "i feel overwhelmed",
            "i am excited",
            "i am grateful",
            "i am nervous about this",
            "this is really stressful",
            "i feel great today",
        ]

    async def respond(self, input_text, facts, confidence, memory_gate, llm) -> str:
        if llm and llm.available:
            prompt = (
                f"The user expressed emotion: \"{input_text}\"\n"
                f"Respond with genuine empathy. 1 sentence. "
                f"Don't diagnose or advise — just acknowledge the feeling."
            )
            response = await llm.ask(prompt, max_tokens=64)
            if response:
                return response
        return "I hear you."


# ── CONTRADICT agent ────────────────────────────────────────────────────

class ContradictAgent(SubAgent):
    """Handles corrections — acknowledges the update."""

    action = Action.CONTRADICT
    name = "contradict"

    def _exemplars(self) -> list[str]:
        return [
            "no that is wrong",
            "that is not what i said",
            "actually it is different",
            "you are mistaken",
            "i never said that",
            "that contradicts what i told you",
            "wait that is not right",
            "you have it backwards",
            "no i meant something else",
            "let me correct you",
        ]

    async def respond(self, input_text, facts, confidence, memory_gate, llm) -> str:
        if llm and llm.available:
            parts = [f"The user is correcting you: \"{input_text}\""]
            if facts:
                parts.append(f"\nWhat you previously knew:")
                for content, speaker, sim in facts[:2]:
                    parts.append(f"  - {content}")
            parts.append(
                "\nAcknowledge the correction. Update your understanding. "
                "1 sentence. Don't apologize excessively."
            )
            response = await llm.ask("\n".join(parts), max_tokens=64)
            if response:
                return response
        return "OK, I'll update that."


# ── WONDER agent ────────────────────────────────────────────────────────

class WonderAgent(SubAgent):
    """Handles novel input — asks thoughtful follow-ups."""

    action = Action.WONDER
    name = "wonder"

    def _exemplars(self) -> list[str]:
        return [
            "tell me more about that",
            "that is interesting",
            "i wonder why",
            "i have never thought about that",
            "huh i did not know that",
            "how does that work",
            "what do you think about this",
            "have you ever considered",
        ]

    async def respond(self, input_text, facts, confidence, memory_gate, llm) -> str:
        if llm and llm.available:
            prompt = (
                f"The user said something intriguing: \"{input_text}\"\n"
                f"Ask one thoughtful follow-up question. Be curious, not interrogative. "
                f"1 sentence."
            )
            response = await llm.ask(prompt, max_tokens=64)
            if response:
                return response
        return "Tell me more about that."


# ── LLM Router ──────────────────────────────────────────────────────────

class LLMRouter:
    """Breaks routing ties when HDC classification is ambiguous.

    Uses sub-agent HDC models first (fast, deterministic), then
    falls back to LLM classification if still ambiguous.
    """

    ACTION_DESCRIPTIONS = {
        "STORE": "user telling you a fact or statement",
        "RECALL": "user asking a question or requesting information",
        "FEEL": "user expressing an emotion or feeling",
        "CONTRADICT": "user correcting something you said or know",
        "WONDER": "novel input that deserves a follow-up question",
    }

    async def resolve(
        self,
        input_text: str,
        coarse_state: CognitiveState,
        agents: dict[Action, SubAgent],
        llm: Any,
    ) -> Action:
        """Resolve the routing for an input.

        1. If CognitiveGlyph is confident (not ambiguous), trust it.
        2. If ambiguous, let sub-agent models compete.
        3. If still tied, ask the LLM.
        """
        # ── Trust confident CognitiveGlyph classification
        if not coarse_state.is_ambiguous:
            return coarse_state.action

        # ── Sub-agent model competition
        scores: list[tuple[Action, float]] = []
        for action, agent in agents.items():
            score = agent.score(input_text)
            scores.append((action, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_action, best_score = scores[0]
        runner_up_score = scores[1][1] if len(scores) > 1 else 0.0

        # Clear winner from sub-agent models
        if best_score > runner_up_score + 0.05:
            logger.info(
                f"Sub-agent routing: {input_text[:40]} → {best_action.name} "
                f"({best_score:.3f} vs {runner_up_score:.3f})"
            )
            return best_action

        # ── LLM tiebreaker
        if llm and llm.available:
            action_names = list(self.ACTION_DESCRIPTIONS.keys())
            descriptions = "\n".join(
                f"  {name}: {desc}" for name, desc in self.ACTION_DESCRIPTIONS.items()
            )
            prompt = (
                f"Classify this input into exactly one action.\n"
                f"Input: \"{input_text}\"\n\n"
                f"Actions:\n{descriptions}\n\n"
                f"Reply with ONLY the action name."
            )
            result = await llm.ask(prompt, max_tokens=16)
            if result:
                result = result.strip().upper()
                for action in Action:
                    if action.name == result:
                        logger.info(
                            f"LLM routing: {input_text[:40]} → {action.name}"
                        )
                        return action

        # ── Fallback to CognitiveGlyph's best guess
        return coarse_state.action


# ── Registry ────────────────────────────────────────────────────────────

class SubAgentRegistry:
    """Registry of all cognitive sub-agents."""

    def __init__(self, thought_space: ThoughtGlyphSpace) -> None:
        self.agents: dict[Action, SubAgent] = {
            Action.STORE: StoreAgent(thought_space),
            Action.RECALL: RecallAgent(thought_space),
            Action.FEEL: FeelAgent(thought_space),
            Action.CONTRADICT: ContradictAgent(thought_space),
            Action.WONDER: WonderAgent(thought_space),
        }
        self.router = LLMRouter()

    def get(self, action: Action) -> SubAgent:
        """Get the sub-agent for an action. Falls back to RecallAgent."""
        return self.agents.get(action, self.agents[Action.RECALL])

    async def route_and_respond(
        self,
        input_text: str,
        coarse_state: CognitiveState,
        facts: list[tuple],
        confidence: float,
        memory_gate: str,
        llm: Any,
    ) -> tuple[str, Action]:
        """Route to the right sub-agent and get a response.

        Returns (response_text, resolved_action).
        """
        # Resolve routing
        action = await self.router.resolve(
            input_text, coarse_state, self.agents, llm,
        )

        # Dispatch to sub-agent
        agent = self.get(action)
        response = await agent.respond(
            input_text=input_text,
            facts=facts,
            confidence=confidence,
            memory_gate=memory_gate,
            llm=llm,
        )

        return response, action
