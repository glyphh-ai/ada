"""
Ada's cognitive thought process — runs for every request.

Not a linear pipeline. A loop that perceives, recalls, reasons,
acts, and evaluates. Can loop back if the first attempt doesn't work.

Steps per cycle:
  1. PERCEIVE  — What is this input? (CognitiveGlyph classification)
  2. GUARD     — Is this safe? (firewall, always)
  3. RECALL    — What do I know? (memory search + capability routing)
  4. REASON    — What should I do? (route decision + disambiguation)
  5. ACT       — Execute (capability query or LLM response)
  6. EVALUATE  — Was that good? (confidence check, retry if low)
  7. RESPOND   — Formulate natural response from results

Max 3 cycles. Each cycle can refine the previous one's results.
The thought process is the same whether called from MCP or CLI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_CYCLES = 3
CONFIDENCE_THRESHOLD = 0.5


@dataclass
class Thought:
    """One cycle of Ada's thought process."""
    cycle: int = 0
    # Perception
    input_text: str = ""
    cognitive_state: str = ""  # RECALL, STORE, FEEL, CONTRADICT, etc.
    # Guard
    firewall_blocked: bool = False
    firewall_result: str = ""
    # Recall
    facts: list[tuple] = field(default_factory=list)  # (content, speaker, similarity)
    memory_gate: str = "ASK"  # DONE or ASK from memory recall
    # Reasoning
    capability: Optional[str] = None
    route_confidence: float = 0.0
    route_method: str = ""  # "deterministic", "llm", "memory"
    # Action
    action_result: Optional[str] = None
    action_success: bool = False
    # Evaluation
    confidence: float = 0.0
    should_retry: bool = False
    retry_reason: str = ""


@dataclass
class ThoughtResult:
    """Final output of the thought process."""
    response: str
    capability: Optional[str] = None
    confidence: float = 0.0
    cognitive_state: str = ""
    gate: str = "ASK"
    facts: list[tuple] = field(default_factory=list)
    llm_assisted: bool = False
    firewall_pass: bool = True
    cycles: int = 1
    elapsed_ms: float = 0.0
    thoughts: list[Thought] = field(default_factory=list)


class ThoughtProcess:
    """Ada's cognitive thought process.

    Runs a perceive-guard-recall-reason-act-evaluate loop for every request.
    Each component is injected — the thought process is the orchestrator.
    """

    def __init__(
        self,
        cognitive,       # AdaCognitive
        router,          # BrainRouter
        llm,             # AdaLLM
        model_manager,   # ModelManager
        session_factory, # async_session_maker
        firewall_fn,     # async callable(str) -> Optional[str]
    ):
        self._cognitive = cognitive
        self._router = router
        self._llm = llm
        self._model_manager = model_manager
        self._session_factory = session_factory
        self._firewall = firewall_fn

        # Shared gradient pattern library — learns shortcuts over time
        from domains.brain.gradient_patterns import GradientPatternLibrary
        self._patterns = GradientPatternLibrary()

    async def think(self, input_text: str) -> ThoughtResult:
        """Run the full thought process."""
        start = time.monotonic()
        thoughts: list[Thought] = []
        final_response = ""
        final_thought = Thought(input_text=input_text)

        for cycle in range(MAX_CYCLES):
            thought = Thought(cycle=cycle, input_text=input_text)

            # ── 1. PERCEIVE ──────────────────────────────────────
            state = self._cognitive.cognitive.process(input_text)
            thought.cognitive_state = state.action.name if state else "UNKNOWN"

            # ── 2. GUARD ─────────────────────────────────────────
            if cycle == 0:  # Only firewall on first cycle
                blocked = await self._firewall(input_text)
                if blocked:
                    thought.firewall_blocked = True
                    thought.firewall_result = blocked
                    thoughts.append(thought)
                    return ThoughtResult(
                        response=blocked,
                        capability="firewall",
                        confidence=1.0,
                        cognitive_state=thought.cognitive_state,
                        gate="DONE",
                        firewall_pass=False,
                        cycles=cycle + 1,
                        elapsed_ms=(time.monotonic() - start) * 1000,
                        thoughts=thoughts,
                    )

            # ── 3. RECALL — gradient search through thought space ──
            from domains.brain.cognitive_gradient import CognitiveGradient

            gradient = CognitiveGradient(
                self._cognitive.thought_space,
                patterns=self._patterns,
            )
            grad_result = gradient.search(input_text, top_k=5)

            facts = grad_result.facts  # already ranked by cognitive gradient
            thought.facts = facts

            # Gate: if gradient converged with strong matches, DONE
            if grad_result.converged and facts and facts[0][2] >= CONFIDENCE_THRESHOLD:
                thought.memory_gate = "DONE"
            else:
                thought.memory_gate = "ASK"

            gate = thought.memory_gate

            # ── 4. REASON ────────────────────────────────────────
            route = await self._router.route_with_fallback(input_text, self._llm)
            thought.capability = route.capability
            thought.route_confidence = route.confidence
            thought.route_method = "llm" if route.llm_fallback else "deterministic"

            # If memory has a strong match and no capability routed,
            # this is a memory question — use recall results
            if gate == "DONE" and facts and not route.capability:
                thought.route_method = "memory"

            # ── 5. ACT ──────────────────────────────────────────
            if route.capability:
                result = await self._execute_capability(route.capability, input_text)
                if result:
                    thought.action_result = result
                    thought.action_success = True

            # ── 6. EVALUATE ─────────────────────────────────────
            thought.confidence = self._evaluate_confidence(thought)

            if thought.confidence < CONFIDENCE_THRESHOLD and cycle < MAX_CYCLES - 1:
                # Low confidence — retry with more context
                thought.should_retry = True
                if not thought.action_success and thought.capability:
                    thought.retry_reason = f"{thought.capability} returned no result"
                elif thought.route_confidence < CONFIDENCE_THRESHOLD:
                    thought.retry_reason = "routing uncertain"
                else:
                    thought.should_retry = False  # Don't retry if we just don't know

            thoughts.append(thought)
            final_thought = thought

            if not thought.should_retry:
                break

            # For retry: add context from this cycle's findings
            if thought.facts:
                context_hint = f" (I know: {thoughts[-1].facts[0][0][:50]})"
                input_text = input_text + context_hint

        # ── 7. RESPOND ──────────────────────────────────────────
        final_response = await self._formulate_response(
            final_thought, input_text
        )

        # Absorb the interaction
        self._cognitive.absorb(final_thought.input_text)
        if final_response:
            self._cognitive.absorb(final_response, speaker="ada")

        # Record Q→A pattern — the question IS the gradient
        if final_response and final_thought.confidence >= CONFIDENCE_THRESHOLD:
            query_glyph = self._cognitive.thought_space.encoder.encode_thought(
                final_thought.input_text, speaker="incoming"
            )
            query_vec = query_glyph.metadata.get("_content_vector")
            if query_vec is not None:
                import numpy as np
                self._patterns.record(
                    query_vector=np.array(query_vec, dtype=np.float64),
                    answer_content=final_response,
                    answer_speaker="ada",
                    residual_steps=len(thoughts),
                )

        elapsed = (time.monotonic() - start) * 1000

        return ThoughtResult(
            response=final_response,
            capability=final_thought.capability,
            confidence=final_thought.confidence,
            cognitive_state=final_thought.cognitive_state,
            gate="DONE" if final_thought.confidence >= CONFIDENCE_THRESHOLD else "ASK",
            facts=final_thought.facts,
            llm_assisted=final_thought.route_method == "llm" or not final_thought.action_success,
            firewall_pass=True,
            cycles=len(thoughts),
            elapsed_ms=elapsed,
            thoughts=thoughts,
        )

    # ── Capability execution ─────────────────────────────────────

    async def _execute_capability(self, capability_name: str, query: str) -> Optional[str]:
        from domains.brain.loader import ADA_ORG_ID

        key = (ADA_ORG_ID, capability_name)
        loaded = self._model_manager._models.get(key)
        if not loaded:
            return None

        try:
            from domains.query.service import QueryService
            query_service = QueryService(self._model_manager, self._session_factory)
            result = await query_service.similarity_search(
                org_id=ADA_ORG_ID,
                model_id=capability_name,
                query=query,
            )
            if result and isinstance(result, dict):
                return self._extract_result_text(result, capability_name)
        except Exception as e:
            logger.error(f"Capability {capability_name} error: {e}")

        return None

    def _extract_result_text(self, result: dict, capability: str) -> str:
        """Extract meaningful text from a query result."""
        ft = result.get("fact_tree", {})
        children = ft.get("children", [])
        if not children:
            return ""

        lines = []
        for child in children[:5]:
            desc = child.get("description", "")
            val = child.get("value", 0)
            if desc:
                lines.append(f"{desc} ({val:.2f})")

        return "\n".join(lines) if lines else ""

    # ── Confidence evaluation ────────────────────────────────────

    def _evaluate_confidence(self, thought: Thought) -> float:
        """Score how confident Ada is in this cycle's result."""
        score = 0.0

        # Route confidence
        if thought.capability:
            score += thought.route_confidence * 0.4

        # Action success
        if thought.action_success:
            score += 0.3

        # Memory relevance
        if thought.facts:
            top_sim = thought.facts[0][2]  # similarity of best fact
            score += top_sim * 0.3

        # Memory gate says DONE
        if thought.memory_gate == "DONE":
            score += 0.1

        return min(1.0, score)

    # ── Response formulation ─────────────────────────────────────

    async def _formulate_response(self, thought: Thought, input_text: str) -> str:
        """Formulate a natural response using Haiku."""

        # If a capability returned a result, use it directly
        if thought.action_success and thought.action_result:
            # Capability gave us structured data — have Haiku make it natural
            if self._llm.available:
                prompt = (
                    f"User asked: \"{input_text}\"\n"
                    f"Capability [{thought.capability}] returned:\n"
                    f"{thought.action_result}\n\n"
                    f"Summarize this as Ada in 1-2 sentences. Be direct."
                )
                response = await self._llm.ask(prompt)
                if response:
                    return response
            return thought.action_result

        # ── Hallucination gate ─────────────────────────────────
        # If overall confidence is below threshold, the LLM is cut out
        # entirely. Ada returns facts raw or says "I don't know."
        # This prevents the LLM from confabulating plausible-sounding
        # responses from weakly-matched or wrong facts.

        if thought.confidence < CONFIDENCE_THRESHOLD:
            # Low confidence — no LLM, just facts or "I don't know"
            if thought.memory_gate == "DONE" and thought.facts:
                return thought.facts[0][0]
            return "I don't have information about that in my memory."

        # High confidence + gradient converged — LLM synthesizes from grounded facts
        if thought.memory_gate == "DONE" and thought.facts:
            if self._llm.available:
                parts = [f"User said: \"{input_text}\""]
                parts.append("\nFacts (from your memory — these are TRUE, use them):")
                for content, speaker, sim in thought.facts[:5]:
                    parts.append(f"  - {content}")
                parts.append(
                    "\nRespond using ONLY these facts. 1-2 sentences. "
                    "Do not add information that isn't in the facts."
                )
                response = await self._llm.ask("\n".join(parts))
                if response:
                    return response
            return thought.facts[0][0]

        # High confidence but gradient didn't converge — partial match
        if thought.facts and thought.facts[0][2] >= CONFIDENCE_THRESHOLD:
            if self._llm.available:
                parts = [f"User said: \"{input_text}\""]
                parts.append("\nPossibly relevant memories (not fully confirmed):")
                for content, speaker, sim in thought.facts[:3]:
                    parts.append(f"  - {content}")
                parts.append(
                    "\nRespond as Ada. Use these memories if they seem relevant, "
                    "but say what you're unsure about. 1-2 sentences."
                )
                response = await self._llm.ask("\n".join(parts))
                if response:
                    return response

        # No facts, no convergence — honest "I don't know"
        return "I don't have information about that in my memory."
