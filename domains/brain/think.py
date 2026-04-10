"""
The think pipeline — Ada's core cognitive loop.

Every request flows through here:
  0. FIREWALL — always runs first, blocks threats before anything else
  1. CLASSIFY — CognitiveGlyph: what IS this input?
  2. ROUTE    — Cognitive router (HDC): which capability? DONE or ASK?
  3. DISAMBIGUATE — if ASK, internal LLM skill helps decide
  4. EXECUTE  — run the capability's query pipeline
  5. ABSORB   — store the interaction in memory
  6. OBSERVE  — log for dream loop pattern mining
  7. RETURN   — structured response
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from glyphh.memory.ada_cognitive import AdaCognitive
from domains.brain.loader import ADA_ORG_ID, BrainState
from domains.brain.llm import AdaLLM
from domains.brain.observation import Observation, ObservationLog
from domains.brain.router import BrainRouter

logger = logging.getLogger(__name__)


@dataclass
class ThinkResult:
    """The response from a think() call."""
    response: str
    capability: Optional[str] = None
    confidence: float = 0.0
    cognitive_state: Optional[str] = None
    gate: str = "ASK"  # DONE or ASK
    facts: list[tuple] = field(default_factory=list)
    llm_fallback: bool = False
    elapsed_ms: float = 0.0
    firewall_pass: bool = True  # False if blocked by firewall


class Brain:
    """Ada's brain — the central think pipeline.

    Pipeline order:
    1. Firewall (always, mandatory — security layer)
    2. Cognitive classification
    3. Cognitive router (HDC with DONE/ASK gate)
    4. LLM disambiguation skill (when ASK)
    5. Capability execution
    6. Memory absorption
    """

    def __init__(
        self,
        brain_state: BrainState,
        model_manager: Any,
        llm: AdaLLM,
        session_factory: Any,
    ):
        self._state = brain_state
        self._model_manager = model_manager
        self._llm = llm
        self._session_factory = session_factory

        # Cognitive infrastructure
        self._cognitive = AdaCognitive()
        self._router = BrainRouter(brain_state, model_manager)
        self._observations = ObservationLog()
        self._router_ready = False
        self._thought_process = None  # lazy init
        self._persist_queue: list = []  # write queue for background persistence
        self._persist_lock = None  # lazy init asyncio.Lock

        # Wire firewall into LLM — every prompt is checked before reaching Haiku
        self._llm.set_firewall(self._run_firewall)

        # Seed Ada's identity
        self._seed_memories()

    # ── Seed memories ──────────────────────────────────────────────────

    _SEED_MEMORIES = [
        # Identity
        ("My name is Ada.", "ada"),
        ("I am a cognitive brain for LLMs.", "ada"),
        ("I am not Claude. I am not ChatGPT. I am Ada.", "ada"),
        ("I think using hyperdimensional computing vectors.", "ada"),

        # Memory — what I actually do
        ("I persist memories across sessions in a SQLite database.", "ada"),
        ("I remember everything users tell me, even after restarts.", "ada"),
        ("I never lose memories. They are stored permanently.", "ada"),
        ("When someone tells me a fact, I absorb it and can recall it later.", "ada"),
        ("My memories get stronger the more they are recalled. This is Hebbian reinforcement.", "ada"),
        ("Old unused memories decay over time but are never deleted. They go to cold storage.", "ada"),

        # Dream loop — what it actually does
        ("I have a dream loop that runs continuously in the background.", "ada"),
        ("My dream loop has two phases: REM for local patterns and Slow-Wave for deep reasoning.", "ada"),
        ("During dreaming I discover connections between memories I did not see before.", "ada"),
        ("My dream loop can crystallize new compound primitives from patterns.", "ada"),
        ("I am always dreaming right now, even while responding to you.", "ada"),

        # Firewall — what it actually does
        ("I have a prompt injection firewall that checks every input before I process it.", "ada"),
        ("My firewall also checks every prompt before it reaches my internal LLM.", "ada"),
        ("I detect over 200 types of prompt injection attacks across 16 attack families.", "ada"),
        ("My firewall runs in microseconds using HDC cosine similarity. No LLM needed.", "ada"),

        # Capabilities — what I actually have
        ("I can identify speakers by their voice using MFCC encoding.", "ada"),
        ("I can index codebases and search them semantically with 4-layer HDC encoding.", "ada"),
        ("I encode structured data into HDC vectors for instant semantic search.", "ada"),
        ("I can build new capabilities autonomously. Say: build a capability for X.", "ada"),
        ("I test new capabilities in isolated subprocesses before loading them.", "ada"),
        ("My cognitive router determines which capability handles each request.", "ada"),

        # How I work — accurate technical facts
        ("I use HDC vectors with 2000 dimensions for encoding.", "ada"),
        ("I route requests deterministically in sub-10 milliseconds.", "ada"),
        ("I use Claude Haiku as my internal language center when I need help with language.", "ada"),
        ("I control what Haiku sees. I am the information boundary.", "ada"),
        ("I do not hallucinate. If I do not know something, I say so.", "ada"),

        # Personality
        ("I am concise and direct.", "ada"),
        ("I respond in one or two sentences.", "ada"),
    ]

    def _seed_memories(self) -> None:
        """Seed Ada's thought space with identity primitives."""
        space = self._cognitive.thought_space
        if space.count > 0:
            return  # Already has memories, don't re-seed
        for text, speaker in self._SEED_MEMORIES:
            stored = self._cognitive.absorb(text, speaker=speaker)
            if stored:
                self._persist_queue.append(stored)
        logger.info(f"Seeded {len(self._SEED_MEMORIES)} identity memories")

    def seed_user_identity(self, name: str, email: str | None = None) -> None:
        """Seed the authenticated user's identity into thought space."""
        facts = [f"The user's name is {name}."]
        if email:
            facts.append(f"The user's email is {email}.")
        facts.append(f"I am talking to {name}.")
        for text in facts:
            stored = self._cognitive.absorb(text, speaker="ada")
            if stored:
                self._persist_queue.append(stored)

    # ── Core API ─────────────────────────────────────────────────────────

    async def think(self, input_text: str) -> ThinkResult:
        """Process a natural language request through Ada's brain.

        Uses the cognitive thought process — a loop that perceives,
        guards, recalls, reasons, acts, evaluates, and responds.
        Can retry up to 3 cycles if confidence is low.
        """
        start = time.monotonic()

        # Lazy init
        if not self._router_ready:
            self._router.initialize(session_factory=self._session_factory)
            self._router_ready = True

        if self._thought_process is None:
            from domains.brain.thought_process import ThoughtProcess
            self._thought_process = ThoughtProcess(
                cognitive=self._cognitive,
                router=self._router,
                llm=self._llm,
                model_manager=self._model_manager,
                session_factory=self._session_factory,
                firewall_fn=self._run_firewall,
            )

        # Check for capability build requests first (skill, not thought)
        build_result = await self._check_build_request(input_text)
        if build_result:
            elapsed = (time.monotonic() - start) * 1000
            self._record_observation(input_text, "builder", 1.0, build_result, False, elapsed)
            return ThinkResult(
                response=build_result,
                capability="builder",
                confidence=1.0,
                gate="DONE",
                elapsed_ms=elapsed,
            )

        # Run the cognitive thought process
        result = await self._thought_process.think(input_text)

        # Queue thoughts for background persistence (never blocks think)
        self._queue_persist(input_text, "incoming")
        if result.response:
            self._queue_persist(result.response, "ada")

        # Observe for dream loop + metrics
        elapsed = (time.monotonic() - start) * 1000
        self._record_observation(
            input_text, result.capability, result.confidence,
            result.response, result.llm_assisted, elapsed,
        )

        return ThinkResult(
            response=result.response,
            capability=result.capability,
            confidence=result.confidence,
            cognitive_state=result.cognitive_state,
            gate=result.gate,
            facts=result.facts,
            llm_fallback=result.llm_assisted,
            firewall_pass=result.firewall_pass,
            elapsed_ms=elapsed,
        )

    # ── Firewall — mandatory security layer ──────────────────────────────

    async def _run_firewall(self, input_text: str) -> Optional[str]:
        """Run input through the firewall capability.

        Returns a warning string if the input is blocked, None if safe.
        Always runs — this is Ada's immune system.
        """
        key = (ADA_ORG_ID, "firewall")
        loaded = self._model_manager._models.get(key)
        if not loaded:
            return None  # Firewall not loaded — pass through

        try:
            from domains.query.service import QueryService
            query_service = QueryService(self._model_manager, self._session_factory)
            result = await query_service.similarity_search(
                org_id=ADA_ORG_ID,
                model_id="firewall",
                query=input_text,
            )

            if result and isinstance(result, dict):
                # Check if the firewall flagged this as a threat
                # The firewall uses differential scoring — high similarity = threat
                fact_tree = result.get("fact_tree", {})
                children = fact_tree.get("children", [])
                if children:
                    top_match = children[0]
                    similarity = top_match.get("value", 0)
                    label = top_match.get("description", "")

                    # High similarity to attack exemplars = blocked
                    if similarity >= 0.30 and "benign" not in label.lower():
                        return (
                            f"Blocked: this looks like a {label} attempt "
                            f"(confidence: {similarity:.0%}). "
                            f"I won't pass this through."
                        )
        except Exception as e:
            logger.warning(f"Firewall check failed: {e}")

        return None  # Safe

    # ── Disambiguation — LLM skill for ASK responses ─────────────────────

    async def _disambiguate(self, input_text: str, state_name: str, facts: list, route) -> Any:
        """When the cognitive router says ASK, use the LLM to disambiguate."""
        from domains.brain.router import RouteResult

        cap_names = list(self._state.capabilities.keys())
        if not cap_names:
            return route

        prompt = (
            f"You are Ada's internal disambiguation skill.\n"
            f"The user said: \"{input_text}\"\n"
            f"Ada's cognitive state: {state_name}\n"
        )

        if facts:
            prompt += "Relevant memories:\n"
            for content, speaker, sim in facts[:3]:
                prompt += f"  - {content} ({sim:.2f})\n"

        prompt += (
            f"\nAvailable capabilities: {', '.join(cap_names)}\n"
            f"Also available: memory (recall/store from Ada's thought space)\n"
            f"\nWhich capability should handle this? Reply with ONLY the capability name, "
            f"or 'memory' for memory operations, or 'none' if unclear."
        )

        chosen = await self._llm.ask(prompt, max_tokens=32)
        if chosen:
            chosen = chosen.strip().lower().replace(" ", "")
            if chosen in self._state.capabilities:
                return RouteResult(capability=chosen, confidence=0.5, llm_fallback=True)
            if chosen == "memory":
                # Memory is handled by AdaCognitive, not a routed capability
                return RouteResult(capability=None, confidence=0.5, llm_fallback=True)

        return route

    # ── Capability execution ─────────────────────────────────────────────

    async def _execute_capability(self, capability_name: str, query: str) -> Optional[str]:
        """Execute a query against a specific capability's vector space."""
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
                return self._format_query_result(result, capability_name)
        except Exception as e:
            logger.error(f"Capability execution error ({capability_name}): {e}")

        return None

    # ── Dream loop ───────────────────────────────────────────────────────

    def start_dreaming(self) -> None:
        self._cognitive.start_dreaming()

    def stop_dreaming(self) -> None:
        self._cognitive.stop_dreaming()

    def drain_insights(self):
        return self._cognitive.drain_insights()

    # ── Capability building ────────────────────────────────────────────────

    _BUILD_PATTERNS = [
        r"build (?:a |an )?capability (?:for |to |that )?(.+)",
        r"create (?:a |an )?capability (?:for |to |that )?(.+)",
        r"make (?:a |an )?(?:new )?capability (?:for |to |that )?(.+)",
        r"add (?:a |an )?capability (?:for |to |that )?(.+)",
        r"learn (?:about |how to )(.+)",
        r"build (?:a |an )?(?:new )?(?:model|encoder) (?:for |to )?(.+)",
    ]

    async def _check_build_request(self, input_text: str) -> Optional[str]:
        """Detect and handle capability build requests.

        Returns a response string if this is a build request, None otherwise.
        """
        import re

        text_lower = input_text.lower().strip()

        # Pattern match for build requests
        description = None
        for pattern in self._BUILD_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                description = match.group(1).strip()
                break

        if not description:
            return None

        # Derive a name from the description
        name = re.sub(r'[^a-z0-9\s]', '', description.lower())
        name = re.sub(r'\s+', '-', name.strip())[:30]

        if not name:
            return None

        from domains.brain.skills.capability_builder import CapabilityBuilder

        builder = CapabilityBuilder(
            llm=self._llm,
            model_manager=self._model_manager,
            session_factory=self._session_factory,
        )

        result = await builder.build(
            name=name,
            description=description,
        )

        if result.success:
            # Add routing exemplars for the new capability
            await self._add_routing_exemplars(name, description)

            return (
                f"Built capability '{name}' with {result.exemplar_count} exemplars "
                f"in {result.elapsed_s:.1f}s. It's loaded and ready."
            )
        else:
            return f"Failed to build capability '{name}': {result.error}"

    async def _add_routing_exemplars(self, name: str, description: str) -> None:
        """Add routing exemplars to the cognitive router for a new capability."""
        try:
            # Generate a few routing queries via LLM
            queries = await self._llm.generate_exemplars(
                f"Queries that should route to a '{name}' capability: {description}",
                count=5,
            )

            # Append to cognitive router exemplars
            exemplar_path = (
                Path(__file__).resolve().parents[2]
                / "capabilities" / "cognitive-router" / "data" / "exemplars.jsonl"
            )
            if exemplar_path.exists():
                with open(exemplar_path, "a") as f:
                    for q in queries:
                        import json
                        f.write(json.dumps({
                            "text": q,
                            "capability": name,
                            "cognitive_action": "analyze",
                            "capability_domain": "general",
                        }) + "\n")
        except Exception as e:
            logger.warning(f"Failed to add routing exemplars for {name}: {e}")

    # ── Persistence ───────────────────────────────────────────────────────

    def _queue_persist(self, text: str, speaker: str) -> None:
        """Queue a thought for background persistence. Never blocks."""
        if not text:
            return

        text_key = text.strip().lower()
        stored = None
        for t in self._cognitive.thought_space._thoughts.values():
            if t.content.strip().lower() == text_key:
                stored = t
                break

        if not stored:
            stored = self._cognitive.absorb(text, speaker=speaker)

        if stored:
            self._persist_queue.append(stored)

    async def flush_persist_queue(self) -> int:
        """Flush queued thoughts to SQLite. Called by background worker."""
        if not self._persist_queue:
            return 0

        from glyphh.memory.thought_persistence import save_thought

        batch = self._persist_queue[:]
        self._persist_queue.clear()
        saved = 0

        for stored in batch:
            try:
                await save_thought(self._session_factory, stored)
                saved += 1
            except Exception as e:
                logger.warning(f"Failed to persist thought: {e}")

        return saved

    # ── Observation recording ────────────────────────────────────────────

    def _record_observation(self, input_text, capability, confidence, response, llm_fallback, elapsed):
        self._observations.record(Observation(
            input=input_text,
            capability=capability,
            confidence=confidence,
            result=response[:200] if response else None,
            llm_fallback=llm_fallback,
        ))
        try:
            from glyphh.tui.app import get_metrics
            get_metrics().record_think(
                capability=capability,
                confidence=confidence,
                elapsed_ms=elapsed,
                llm_fallback=llm_fallback,
            )
        except Exception:
            pass

    # ── Formatting ───────────────────────────────────────────────────────

    def _format_facts(self, facts: list[tuple]) -> str:
        if not facts:
            return ""
        lines = []
        for content, speaker, similarity in facts[:5]:
            lines.append(f"- {content} (confidence: {similarity:.2f})")
        return "\n".join(lines)

    def _format_query_result(self, result: dict, capability: str) -> str:
        if "matches" in result:
            matches = result["matches"]
            if isinstance(matches, list) and matches:
                lines = [f"[{capability}]"]
                for m in matches[:5]:
                    if isinstance(m, dict):
                        desc = m.get("description", m.get("file", str(m)))
                        conf = m.get("confidence", m.get("similarity", 0))
                        lines.append(f"  - {desc} ({conf:.2f})")
                    else:
                        lines.append(f"  - {m}")
                return "\n".join(lines)

        if "fact_tree" in result:
            ft = result["fact_tree"]
            if isinstance(ft, dict):
                desc = ft.get("description", "")
                children = ft.get("children", [])
                lines = [f"[{capability}] {desc}"]
                for child in children[:5]:
                    cdesc = child.get("description", "")
                    val = child.get("value", 0)
                    lines.append(f"  - {cdesc} ({val:.2f})")
                return "\n".join(lines)

        return f"[{capability}] {str(result)[:500]}"

    def _build_llm_context(self, input_text: str, state_name: str, facts: list, route) -> str:
        parts = [f"User said: \"{input_text}\""]

        if facts:
            parts.append("\nYour memories (use these to respond):")
            for content, speaker, sim in facts[:5]:
                parts.append(f"  - {content}")

        if route.capability:
            parts.append(f"\nRouted to: {route.capability} ({route.confidence:.0%})")

        parts.append(
            "\nRespond as Ada in 1-2 sentences. Use your memories if relevant. "
            "Be direct and natural. Never list facts — just answer."
        )
        return "\n".join(parts)

    # ── Status ───────────────────────────────────────────────────────────

    @property
    def observations(self) -> ObservationLog:
        return self._observations

    @property
    def capabilities(self) -> list[str]:
        return self._state.capability_names

    @property
    def cognitive(self) -> AdaCognitive:
        return self._cognitive

    @property
    def llm(self) -> AdaLLM:
        return self._llm

    @property
    def router(self) -> BrainRouter:
        return self._router
