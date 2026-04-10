"""
Ada's thought process — runs for every request.

A linear pipeline, not a classification tree. Every input goes
through the same steps. The response varies based on what was
extracted and what was recalled — not which bucket the input
was classified into.

Steps:
  1. GUARD    — Is this safe? (firewall, always first)
  2. EXTRACT  — What's in this input? (facts, questions, entities, emotions)
  3. STORE    — Absorb any new facts into thought space
  4. RECALL   — Search thought space for relevant memories
  5. RESPOND  — LLM generates response grounded by recalled facts
  6. ABSORB   — Store the response for future context
  7. LEARN    — Encode interaction pattern into user derivative

The LLM handles language. The thought space handles memory.
The derivative learns who the user is.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.5


@dataclass
class ThoughtResult:
    """Final output of the thought process."""
    response: str
    confidence: float = 0.0
    gate: str = "ASK"
    facts: list[tuple] = field(default_factory=list)
    capability: Optional[str] = None
    llm_assisted: bool = False
    firewall_pass: bool = True
    elapsed_ms: float = 0.0
    # Extraction results
    extracted_question: Optional[str] = None
    extracted_facts: list[str] = field(default_factory=list)
    extracted_emotion: Optional[str] = None
    is_greeting: bool = False
    is_correction: bool = False


class ThoughtProcess:
    """Ada's thought process.

    Every input goes through the same pipeline:
    guard → extract → store → recall → respond → absorb → learn.

    No classification buckets. No sub-agent routing.
    The LLM handles language. Facts drive content.
    """

    def __init__(
        self,
        cognitive,       # AdaCognitive
        router,          # BrainRouter
        llm,             # AdaLLM
        model_manager,   # ModelManager
        session_factory, # async_session_maker
        firewall_fn,     # async callable(str) -> Optional[str]
        derivative=None, # UserDerivative (optional)
        thread_store=None,  # ThreadStore (optional)
    ):
        self._cognitive = cognitive
        self._router = router
        self._llm = llm
        self._model_manager = model_manager
        self._session_factory = session_factory
        self._firewall = firewall_fn
        self._derivative = derivative
        self._thread_store = thread_store

        # Thread managers per tool (lazy init)
        self._thread_managers: dict[str, Any] = {}

        # Gradient pattern library — learns search shortcuts
        from domains.brain.gradient_patterns import GradientPatternLibrary
        self._patterns = GradientPatternLibrary()

        # Extractor — pulls structure from natural language
        from domains.brain.extractor import Extractor
        self._extractor = Extractor()

    async def think(self, input_text: str, tool: str = "unknown") -> ThoughtResult:
        """Run the full thought process."""
        start = time.monotonic()

        # ── 1. GUARD ────────────────────────────────────────────
        blocked = await self._firewall(input_text)
        if blocked:
            return ThoughtResult(
                response=blocked,
                capability="firewall",
                confidence=1.0,
                gate="DONE",
                firewall_pass=False,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )

        # ── 2. EXTRACT ──────────────────────────────────────────
        extraction = await self._extractor.extract(input_text, llm=self._llm)

        # ── 3. STORE (threads) ──────────────────────────────────
        # Store facts into a context thread tagged by tool + topic.
        # Also absorb into thought space for HDC fallback.
        thread = None
        if self._thread_store and extraction.has_facts:
            manager = self._get_thread_manager(tool)
            thread = manager.process(
                input_text=input_text,
                facts=extraction.facts,
                entities=extraction.entities,
                question=extraction.question,
                emotion=extraction.emotion,
            )
        if extraction.has_facts:
            for fact in extraction.facts:
                self._cognitive.absorb(fact)

        # ── 4. RECALL ───────────────────────────────────────────
        # Primary: thread-based structured recall (entity + topic)
        # Fallback: HDC cosine similarity on thought space
        search_text = extraction.question or input_text
        facts, confidence = self._recall_threads(
            search_text, extraction, tool,
        )

        # If thread recall found nothing, fall back to HDC
        if not facts or confidence < CONFIDENCE_THRESHOLD:
            hdc_facts, hdc_confidence = self._recall_hdc(search_text)
            if hdc_confidence > confidence:
                facts = hdc_facts
                confidence = hdc_confidence

        # Also try capability routing
        capability = None
        capability_result = None
        route = await self._router.route_with_fallback(input_text, self._llm)
        if route.capability:
            capability = route.capability
            capability_result = await self._execute_capability(
                route.capability, input_text
            )

        # ── 5. RESPOND ──────────────────────────────────────────
        response = await self._respond(
            input_text=input_text,
            extraction=extraction,
            facts=facts,
            confidence=confidence,
            capability=capability,
            capability_result=capability_result,
        )

        # ── 6. ABSORB ───────────────────────────────────────────
        if response:
            self._cognitive.absorb(response, speaker="ada")

        # Record Q→A pattern for gradient search shortcuts
        if response and confidence >= CONFIDENCE_THRESHOLD:
            self._record_pattern(input_text, response)

        # If this was a question, also record it in the active thread
        if self._thread_store and extraction.has_question and not thread:
            manager = self._get_thread_manager(tool)
            manager.process(
                input_text=input_text,
                facts=[],
                entities=extraction.entities,
                question=extraction.question,
                emotion=extraction.emotion,
            )

        # ── 7. LEARN ────────────────────────────────────────────
        if self._derivative:
            self._derivative.observe(
                input_text=input_text,
                response_text=response,
                extracted_facts=extraction.facts,
                extracted_question=extraction.question,
                extracted_emotion=extraction.emotion,
                is_correction=extraction.is_correction,
                recalled_facts=facts,
            )

        elapsed = (time.monotonic() - start) * 1000

        return ThoughtResult(
            response=response,
            confidence=confidence,
            gate="DONE" if confidence >= CONFIDENCE_THRESHOLD else "ASK",
            facts=facts,
            capability=capability,
            llm_assisted=True,
            firewall_pass=True,
            elapsed_ms=elapsed,
            extracted_question=extraction.question,
            extracted_facts=extraction.facts,
            extracted_emotion=extraction.emotion,
            is_greeting=extraction.is_greeting,
            is_correction=extraction.is_correction,
        )

    def _get_thread_manager(self, tool: str):
        """Get or create a ThreadManager for a tool."""
        if tool not in self._thread_managers:
            from domains.brain.thread_manager import ThreadManager
            self._thread_managers[tool] = ThreadManager(
                store=self._thread_store,
                tool=tool,
            )
        return self._thread_managers[tool]

    # ── Recall ──────────────────────────────────────────────────

    # ── Recall — thread-based (primary) ──────────────────────────

    def _recall_threads(
        self,
        search_text: str,
        extraction,
        tool: str,
    ) -> tuple[list[tuple], float]:
        """Thread-based recall — glyph similarity search.

        Primary recall path. Same three-signal cosine similarity
        as thought space, but at the thread level.
        Returns (facts, confidence) tuples.
        """
        if not self._thread_store or self._thread_store.count == 0:
            return [], 0.0

        manager = self._get_thread_manager(tool)
        threads = manager.recall_for_query(
            query_text=search_text,
            query_entities=extraction.entities if extraction else None,
        )

        if not threads:
            return [], 0.0

        facts = manager.collect_facts(threads)
        confidence = min(1.0, facts[0][2]) if facts else 0.0

        return facts, confidence

    # ── Recall — HDC fallback ──────────────────────────────────

    def _recall_hdc(self, search_text: str) -> tuple[list[tuple], float]:
        """HDC cosine similarity fallback for when threads don't match.

        Uses CognitiveGradient search through the thought space.
        """
        from domains.brain.cognitive_gradient import CognitiveGradient

        gradient = CognitiveGradient(
            self._cognitive.thought_space,
            patterns=self._patterns,
        )
        result = gradient.search(search_text, top_k=5)
        facts = result.facts

        confidence = 0.0
        if facts:
            confidence = min(1.0, facts[0][2])

        return facts, confidence

    # ── Response generation ─────────────────────────────────────

    async def _respond(
        self,
        input_text: str,
        extraction,
        facts: list[tuple],
        confidence: float,
        capability: Optional[str],
        capability_result: Optional[str],
    ) -> str:
        """Generate a response. The LLM handles language, facts drive content.

        Hallucination gate: question + no relevant facts = "I don't know."
        Everything else: LLM responds naturally, grounded by available facts.
        """
        # ── Capability result takes priority
        if capability_result:
            return await self._respond_capability(
                input_text, capability, capability_result
            )

        # ── Hallucination gate: question + low confidence = block LLM
        if extraction.has_question and confidence < CONFIDENCE_THRESHOLD:
            return "I don't have information about that in my memory."

        # ── LLM response grounded by facts and context
        if self._llm and self._llm.available:
            prompt = self._build_prompt(
                input_text, extraction, facts, confidence
            )
            response = await self._llm.ask(prompt)
            if response:
                return response

        # ── Offline fallbacks
        if extraction.is_greeting:
            return "Hello."
        if extraction.has_question:
            return "I don't have information about that in my memory."
        return "Got it."

    def _build_prompt(
        self,
        input_text: str,
        extraction,
        facts: list[tuple],
        confidence: float,
    ) -> str:
        """Build the LLM prompt. Facts drive content, LLM drives language."""
        parts = []

        # Tone / style directive
        style = None
        if self._derivative:
            style = self._derivative.style_summary()
        if style:
            parts.append(f"You are Ada. {style}")
        else:
            parts.append("You are Ada. Be concise and direct.")

        # The input
        parts.append(f"\nThe user said: \"{input_text}\"")

        # Grounded facts from memory
        if facts and confidence >= CONFIDENCE_THRESHOLD:
            parts.append("\nFrom your memory (these are TRUE — use them):")
            for content, speaker, sim in facts[:5]:
                parts.append(f"  - {content}")

        elif facts and not extraction.has_question:
            # Some related memories but not answering a question
            parts.append("\nRelated memories:")
            for content, speaker, sim in facts[:3]:
                parts.append(f"  - {content}")

        # Derivative context — what Ada has learned about this user
        if self._derivative:
            context = self._derivative.get_context(input_text)
            if context:
                parts.append("\nWhat you know about this user:")
                for c in context:
                    parts.append(f"  - {c}")

        # Emotion acknowledgment
        if extraction.emotion:
            parts.append(
                f"\nThe user is expressing {extraction.emotion}. "
                "Acknowledge the feeling."
            )

        # Correction handling
        if extraction.is_correction:
            parts.append(
                "\nThe user is correcting previous information. "
                "Acknowledge the correction briefly."
            )

        # Response instructions
        if extraction.has_question and facts:
            parts.append(
                "\nAnswer using ONLY the facts above. "
                "Do not add information that isn't in your memory. "
                "1-2 sentences."
            )
        elif extraction.is_greeting:
            parts.append(
                "\nThis is a greeting. Respond naturally. 1 sentence."
            )
        else:
            parts.append("\nRespond naturally. 1-2 sentences.")

        return "\n".join(parts)

    async def _respond_capability(
        self, input_text: str, capability: str, result: str,
    ) -> str:
        """Respond using capability results."""
        if self._llm and self._llm.available:
            prompt = (
                f"You are Ada.\n"
                f"User asked: \"{input_text}\"\n"
                f"Capability [{capability}] returned:\n{result}\n\n"
                f"Summarize in 1-2 sentences. Be direct."
            )
            response = await self._llm.ask(prompt)
            if response:
                return response
        return result

    # ── Capability execution ────────────────────────────────────

    async def _execute_capability(
        self, capability_name: str, query: str,
    ) -> Optional[str]:
        """Execute a query against a capability's vector space."""
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
                return self._extract_result_text(result)
        except Exception as e:
            logger.error(f"Capability {capability_name} error: {e}")

        return None

    def _extract_result_text(self, result: dict) -> str:
        """Extract meaningful text from a capability query result."""
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

    # ── Pattern recording ───────────────────────────────────────

    def _record_pattern(self, input_text: str, response: str) -> None:
        """Record a successful Q→A pattern for gradient search shortcuts."""
        try:
            query_glyph = self._cognitive.thought_space.encoder.encode_thought(
                input_text, speaker="incoming"
            )
            query_vec = query_glyph.metadata.get("_content_vector")
            if query_vec is not None:
                import numpy as np
                self._patterns.record(
                    query_vector=np.array(query_vec, dtype=np.float64),
                    answer_content=response,
                    answer_speaker="ada",
                    residual_steps=1,
                )
        except Exception as e:
            logger.warning(f"Pattern recording failed: {e}")
