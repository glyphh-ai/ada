"""
CognitiveGlyph — Ada's meta-cognition engine.

A glyph of glyphs. Each layer is a cognitive sub-agent that evaluates
every input in parallel: is this a question? a statement? a contradiction?
something novel? emotional? The winning layer determines what Ada DOES
with the thought — store, recall, dream, contradict.

The layers are weighted dynamically. During active conversation, question
and statement layers dominate. When idle, dream and curiosity layers
gain weight — Ada drifts into reflection. This is her attention system.

Each sub-agent has exemplar patterns (taught or crystallized) that it
matches against. The match score IS the layer's activation. The global
cortex of the cognitive glyph is Ada's state of mind — the bundle of
all cognitive activations, shifting in real time.

Architecture:
    CognitiveGlyph
    ├── Layer: question    (am I being asked something?)
    ├── Layer: statement   (am I being told something?)
    ├── Layer: recall      (does this relate to something I know?)
    ├── Layer: emotion     (does this carry feeling?)
    ├── Layer: contradict  (does this conflict with what I know?)
    ├── Layer: curiosity   (is this novel / surprising?)
    └── Layer: dream       (should I reflect on this?)

Each layer is an Encoder with its own exemplars. The meta-glyph is
built by bundling their outputs, weighted by activation strength.

Usage:
    from glyphh.memory.cognitive_glyph import CognitiveGlyph

    cog = CognitiveGlyph(thought_space=space)
    state = cog.process("what is my name?")
    # state.winner = "question", state.action = Action.RECALL
    # state.activations = {"question": 0.82, "statement": 0.15, ...}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.ops import bundle, cosine_similarity
from glyphh.core.types import Concept, Vector
from glyphh.encoder.base import Encoder
from glyphh.memory.thought_space import ThoughtGlyphSpace

logger = logging.getLogger(__name__)


# ── Cognitive actions ───────────────────────────────────────────────────

class Action(Enum):
    """What Ada does with a thought."""
    STORE = "store"           # absorb into long-term memory
    RECALL = "recall"         # search memory and respond
    DREAM = "dream"           # queue for background reflection
    CONTRADICT = "contradict" # flag conflict with known thoughts
    FEEL = "feel"             # update emotional state
    WONDER = "wonder"         # curiosity — explore further


# ── Cognitive sub-agent ─────────────────────────────────────────────────

@dataclass
class CognitiveExemplar:
    """An exemplar pattern for a cognitive sub-agent."""
    text: str
    vector: np.ndarray
    weight: float = 1.0


class CognitiveAgent:
    """A single cognitive sub-agent — one layer of the meta-glyph.

    Each agent has exemplar patterns it matches against. The activation
    score is the max cosine similarity between the input and any exemplar.
    Higher activation = this cognitive process is more relevant.

    Args:
        name: Agent name (e.g., "question", "statement").
        action: What cognitive operation this agent triggers.
        encoder: Shared encoder for encoding input text.
        exemplar_texts: Seed exemplar patterns.
        base_weight: Resting activation weight (how strongly this agent
            fires when idle). Dream and curiosity have higher base weights.
    """

    def __init__(
        self,
        name: str,
        action: Action,
        encoder: Encoder,
        exemplar_texts: list[str] | None = None,
        base_weight: float = 0.0,
    ) -> None:
        self.name = name
        self.action = action
        self._encoder = encoder
        self.base_weight = base_weight
        self._exemplars: list[CognitiveExemplar] = []
        self._activation: float = 0.0
        self._last_fired: float = 0.0

        # Encode seed exemplars
        if exemplar_texts:
            for text in exemplar_texts:
                self.add_exemplar(text)

    @property
    def activation(self) -> float:
        return self._activation

    def add_exemplar(self, text: str, weight: float = 1.0) -> None:
        """Add an exemplar pattern."""
        vec = self._encode_text(text)
        self._exemplars.append(CognitiveExemplar(
            text=text, vector=vec, weight=weight,
        ))

    def evaluate(self, input_vector: np.ndarray) -> float:
        """Score how strongly this agent fires for a given input.

        Returns max cosine across all exemplars, scaled by base_weight.
        """
        if not self._exemplars:
            self._activation = self.base_weight
            return self._activation

        max_sim = 0.0
        for ex in self._exemplars:
            sim = float(cosine_similarity(input_vector, ex.vector))
            weighted = sim * ex.weight
            if weighted > max_sim:
                max_sim = weighted

        self._activation = max(max_sim, self.base_weight)
        if max_sim > 0.1:
            self._last_fired = time.time()
        return self._activation

    def time_since_fired(self) -> float:
        """Seconds since this agent last fired above threshold."""
        if self._last_fired == 0:
            return float("inf")
        return time.time() - self._last_fired

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using the shared encoder's bag-of-words."""
        vec = self._encoder._encode_bag_of_words(text)
        return vec.data


# ── CognitiveState — snapshot of Ada's mind ─────────────────────────────

@dataclass
class CognitiveState:
    """Snapshot of Ada's cognitive state after processing an input."""
    text: str
    activations: dict[str, float]      # agent_name → activation score
    winner: str                         # name of winning agent
    action: Action                      # what to do with this thought
    confidence: float                   # winner's activation score
    runner_up: str | None = None        # second-strongest agent
    runner_up_score: float = 0.0
    state_vector: np.ndarray | None = None  # meta-glyph cortex

    @property
    def is_ambiguous(self) -> bool:
        """True if winner and runner-up are close (within 0.1)."""
        return self.runner_up_score > 0 and (
            self.confidence - self.runner_up_score < 0.1
        )


# ── CognitiveGlyph — the meta-glyph ────────────────────────────────────

# Default exemplars for each cognitive sub-agent.
# These are the SEED patterns — crystallization adds more over time.

_QUESTION_EXEMPLARS = [
    "what is my name",
    "who am i",
    "who are you",
    "what do i like",
    "where do i live",
    "how old is jake",
    "what does sarah do",
    "what is your favorite",
    "do you remember",
    "can you tell me",
    "what happened yesterday",
    "when did i say that",
    "why do you think that",
    "how do you know",
    "what time is it",
    "what do i do for work",
    "what does emma like",
    "what does jake like",
    "where do we live",
]

_STATEMENT_EXEMPLARS = [
    "my name is chris",
    "i like pizza",
    "my wife is named sarah",
    "i work as a software engineer",
    "we live in austin texas",
    "jake is seven years old",
    "i went to the store yesterday",
    "my favorite color is blue",
    "i have two kids",
    "sarah works at the hospital",
    "i am tired today",
    "i feel happy when i code",
    "the sky is blue",
    "dogs are mammals",
    "i learned python ten years ago",
    "jake loves minecraft",
    "emma likes drawing",
    "he enjoys playing soccer",
    "she prefers reading books",
]

_EMOTION_EXEMPLARS = [
    "i am happy",
    "i feel sad",
    "i am angry",
    "i love you",
    "i hate that",
    "i am scared",
    "i am excited",
    "that makes me worried",
    "i feel calm",
    "i am proud of you",
    "that is frustrating",
    "i am grateful",
]

_CONTRADICTION_EXEMPLARS = [
    "no that is wrong",
    "that is not what i said",
    "actually it is different",
    "you are mistaken",
    "i never said that",
    "that contradicts what i told you",
    "wait that is not right",
    "you have it backwards",
]

_CURIOSITY_EXEMPLARS = [
    "tell me more about that",
    "that is interesting",
    "i wonder why",
    "what do you think about",
    "i have never thought about that",
    "that is a good question",
    "huh i did not know that",
    "how does that work",
]


class CognitiveGlyph:
    """Ada's meta-cognition — a glyph of glyphs.

    Runs all cognitive sub-agents in parallel on every input. The winning
    agent determines what Ada does. The combined activation state IS Ada's
    current state of mind.

    Layer weights shift dynamically:
      - Active conversation → question/statement dominate
      - Idle → dream/curiosity weights rise (base_weight)
      - Emotional input → emotion layer spikes

    Args:
        thought_space: Ada's long-term memory.
        dimension: HDC vector dimension.
        seed: Random seed for deterministic encoding.
    """

    def __init__(
        self,
        thought_space: ThoughtGlyphSpace,
        dimension: int = 2000,
        seed: int = 42,
    ) -> None:
        self._space = thought_space
        self._dim = dimension
        self._seed = seed

        # Shared encoder for all sub-agents
        self._encoder = Encoder(EncoderConfig(
            dimension=dimension,
            seed=seed,
        ))

        # Build cognitive sub-agents
        self._agents: dict[str, CognitiveAgent] = {}
        self._build_agents()

        # State tracking
        self._last_state: CognitiveState | None = None
        self._idle_since: float = time.time()
        self._process_count: int = 0

    def _build_agents(self) -> None:
        """Initialize the cognitive sub-agents with seed exemplars."""
        agent_defs = [
            ("question",    Action.RECALL,     _QUESTION_EXEMPLARS,     0.0),
            ("statement",   Action.STORE,      _STATEMENT_EXEMPLARS,    0.0),
            ("emotion",     Action.FEEL,       _EMOTION_EXEMPLARS,      0.0),
            ("contradict",  Action.CONTRADICT, _CONTRADICTION_EXEMPLARS, 0.0),
            ("curiosity",   Action.WONDER,     _CURIOSITY_EXEMPLARS,    0.05),
            ("dream",       Action.DREAM,      [],                      0.1),
        ]

        for name, action, exemplars, base_weight in agent_defs:
            self._agents[name] = CognitiveAgent(
                name=name,
                action=action,
                encoder=self._encoder,
                exemplar_texts=exemplars,
                base_weight=base_weight,
            )

    # ── Core processing ──────────────────────────────────────────────────

    def process(self, text: str) -> CognitiveState:
        """Process an input through all cognitive sub-agents.

        Every agent evaluates the input in parallel. The winning agent
        (highest activation) determines what Ada does with the thought.
        The combined state is Ada's current cognitive snapshot.

        Args:
            text: Input text (from user or Ada's own output).

        Returns:
            CognitiveState with activations, winner, and action.
        """
        self._process_count += 1
        self._idle_since = time.time()

        # Encode input once, shared across all agents
        input_vector = self._encoder._encode_bag_of_words(text).data

        # Evaluate all agents
        activations: dict[str, float] = {}
        for name, agent in self._agents.items():
            # Apply idle boost: dream/curiosity gain weight when idle
            idle_boost = self._idle_boost(agent)
            raw_score = agent.evaluate(input_vector)
            activations[name] = raw_score + idle_boost

        # Find winner and runner-up
        sorted_agents = sorted(
            activations.items(), key=lambda x: x[1], reverse=True,
        )
        winner_name, winner_score = sorted_agents[0]
        runner_name, runner_score = sorted_agents[1] if len(sorted_agents) > 1 else (None, 0.0)

        winner_agent = self._agents[winner_name]

        # Build meta-glyph state vector (bundle of all agent activations)
        state_vector = self._build_state_vector(activations, input_vector)

        state = CognitiveState(
            text=text,
            activations=activations,
            winner=winner_name,
            action=winner_agent.action,
            confidence=winner_score,
            runner_up=runner_name,
            runner_up_score=runner_score,
            state_vector=state_vector,
        )

        self._last_state = state
        logger.info(
            "Cognitive: %s → %s (%.3f) | %s",
            text[:40], winner_name, winner_score,
            " ".join(f"{k}={v:.2f}" for k, v in sorted_agents),
        )
        return state

    def _idle_boost(self, agent: CognitiveAgent) -> float:
        """Boost dream/curiosity agents based on idle time.

        The longer Ada is idle, the more dream and curiosity
        gain weight. Active conversation resets the idle timer.
        """
        if agent.base_weight <= 0:
            return 0.0

        idle_seconds = time.time() - self._idle_since
        # Ramp up over 30 seconds: 0 → base_weight
        ramp = min(1.0, idle_seconds / 30.0)
        return agent.base_weight * ramp

    def _build_state_vector(
        self,
        activations: dict[str, float],
        input_vector: np.ndarray,
    ) -> np.ndarray:
        """Build the meta-glyph cortex — Ada's state of mind.

        Bundle of input vector weighted by each agent's activation.
        Agents that fire strongly contribute more to the state.
        """
        weighted_vecs = []
        for name, score in activations.items():
            if score > 0.05:
                # Weight the input by activation strength
                agent_vec = self._agents[name]._exemplars[0].vector if self._agents[name]._exemplars else input_vector
                weighted = (agent_vec.astype(np.float64) * score)
                weighted_vecs.append(weighted)

        if not weighted_vecs:
            return input_vector.copy()

        summed = np.sum(weighted_vecs, axis=0)
        return np.sign(summed).astype(np.int8)

    # ── Idle / dream transition ──────────────────────────────────────────

    def idle_tick(self) -> CognitiveState | None:
        """Called periodically when Ada has no input.

        As idle time increases, dream layer gains weight. If dream
        wins, return a state indicating Ada should dream.
        """
        idle_seconds = time.time() - self._idle_since
        if idle_seconds < 5.0:
            return None  # too soon

        # Simulate a "nothing happening" input
        activations: dict[str, float] = {}
        for name, agent in self._agents.items():
            activations[name] = self._idle_boost(agent)

        # Check if dream wins
        sorted_agents = sorted(
            activations.items(), key=lambda x: x[1], reverse=True,
        )
        winner_name, winner_score = sorted_agents[0]

        if winner_name == "dream" and winner_score > 0.05:
            return CognitiveState(
                text="(idle)",
                activations=activations,
                winner="dream",
                action=Action.DREAM,
                confidence=winner_score,
                state_vector=None,
            )
        return None

    # ── Exemplar management ──────────────────────────────────────────────

    def add_exemplar(self, agent_name: str, text: str, weight: float = 1.0) -> None:
        """Add an exemplar to a cognitive sub-agent.

        Called by crystallization when the DreamLoop discovers a new
        cognitive pattern.
        """
        if agent_name in self._agents:
            self._agents[agent_name].add_exemplar(text, weight)
            logger.debug("Added exemplar to %s: %s", agent_name, text[:40])

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def last_state(self) -> CognitiveState | None:
        return self._last_state

    @property
    def agents(self) -> dict[str, CognitiveAgent]:
        return self._agents

    def stats(self) -> dict:
        return {
            "process_count": self._process_count,
            "agents": {
                name: {
                    "exemplars": len(agent._exemplars),
                    "activation": agent.activation,
                    "base_weight": agent.base_weight,
                    "last_fired": agent.time_since_fired(),
                }
                for name, agent in self._agents.items()
            },
            "idle_seconds": time.time() - self._idle_since,
        }

    def format_state(self, state: CognitiveState | None = None) -> str:
        """Format cognitive state for display."""
        state = state or self._last_state
        if state is None:
            return "No cognitive state."

        lines = [f"Input: \"{state.text}\""]
        lines.append(f"Action: {state.action.value} (via {state.winner}, {state.confidence:.3f})")

        if state.is_ambiguous:
            lines.append(f"  (ambiguous: {state.runner_up} at {state.runner_up_score:.3f})")

        lines.append("Activations:")
        for name, score in sorted(
            state.activations.items(), key=lambda x: x[1], reverse=True,
        ):
            bar = "#" * int(score * 20)
            lines.append(f"  {name:12s} {score:.3f} {bar}")

        return "\n".join(lines)
