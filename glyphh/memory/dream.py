"""
DreamLoop — Ada's background reasoning engine.

Real cognition doesn't stop.  The DreamLoop runs continuously in a
background thread, wandering through Ada's knowledge graph, discovering
connections nobody asked about, hunting contradictions, and reinforcing
patterns that keep appearing from different starting points.

This is the "free will" layer — Ada chooses what to think about based on
curiosity (weak patterns, fresh atoms, contradictions), not user prompts.

Guardrails:
  - Background discoveries stay weak (max 0.5) — she can't self-confirm
  - Thought budget per cycle prevents obsessive loops
  - Decay still applies — idle thoughts fade
  - Insights are queued, not acted on — the user decides what matters

The DreamLoop is to CognitiveLoop what dreaming is to waking thought:
same machinery, different driver.

Usage:
    dream = DreamLoop(loop, forge, facts)
    dream.start()

    # Later...
    insights = dream.drain_insights()
    for insight in insights:
        print(f"Ada noticed: {insight}")

    dream.stop()
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .atom import AtomForge
    from .binding import FactStore
    from .cognitive import CognitiveLoop, ReasoningChain

logger = logging.getLogger(__name__)


# ── Insight types ─────────────────────────────────────────────────────────

class InsightKind(Enum):
    """What Ada discovered while thinking."""
    CONNECTION = "connection"       # found a new multi-hop chain
    CONTRADICTION = "contradiction" # found conflicting answers
    CONVERGENCE = "convergence"     # multiple paths → same conclusion
    QUESTION = "question"           # found a gap in knowledge


@dataclass
class Insight:
    """A single discovery from background reasoning."""
    kind: InsightKind
    summary: str                    # human-readable description
    atoms: list[str]                # atoms involved
    confidence: float = 0.0         # how confident the discovery is
    chain_depth: int = 0            # hop count if applicable
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"Insight({self.kind.value}: {self.summary})"


# ── Curiosity scoring ─────────────────────────────────────────────────────

@dataclass
class _CuriosityState:
    """Per-atom curiosity tracking."""
    name: str
    score: float = 1.0             # current curiosity (higher = more interesting)
    last_explored: float = 0.0     # timestamp of last exploration
    explore_count: int = 0         # how many times we've explored this
    connection_count: int = 0      # known connections (fewer = more curious)
    contradiction_count: int = 0   # contradictions involving this atom


# ── DreamLoop ─────────────────────────────────────────────────────────────

class DreamLoop:
    """Ada's background reasoning — continuous thought with guardrails.

    The loop cycles through three phases:
      1. Wander   — pick curious atoms, reason between them
      2. Hunt     — look for contradictions and gaps
      3. Converge — check if different paths reach the same conclusion

    Discoveries are queued as Insights for the user to review.
    Background patterns are capped at 0.5 strength — Ada can suspect
    but not self-confirm.

    Args:
        loop:           CognitiveLoop for reasoning.
        forge:          AtomForge for atom access.
        facts:          FactStore for fact access.
        cycle_budget:   Max reasoning chains per cycle.
        cycle_interval: Seconds between thinking cycles.
        max_insights:   Max queued insights before oldest are dropped.
        self_reinforce_cap: Max strength for self-discovered patterns.
    """

    # Max strength a background-discovered pattern can reach
    SELF_REINFORCE_CAP = 0.5

    def __init__(
        self,
        loop: CognitiveLoop,
        forge: AtomForge,
        facts: FactStore,
        cycle_budget: int = 10,
        cycle_interval: float = 2.0,
        max_insights: int = 50,
        self_reinforce_cap: float = 0.5,
    ) -> None:
        self._loop = loop
        self._forge = forge
        self._facts = facts

        self._cycle_budget = cycle_budget
        self._cycle_interval = cycle_interval
        self._max_insights = max_insights
        self.SELF_REINFORCE_CAP = self_reinforce_cap

        self._insights: Queue[Insight] = Queue(maxsize=max_insights)
        self._curiosity: dict[str, _CuriosityState] = {}

        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

        # Stats
        self._cycles = 0
        self._total_chains = 0
        self._total_insights = 0

        # Track what we've already discovered to avoid repeats
        self._seen_connections: set[tuple[str, str]] = set()
        self._convergence_map: dict[str, set[str]] = {}  # answer → set of starting atoms

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background reasoning."""
        if self._running:
            return
        self._running = True
        self._paused = False
        self._thread = threading.Thread(
            target=self._think_loop, daemon=True, name="ada-dream"
        )
        self._thread.start()
        logger.info("DreamLoop started")

    def stop(self) -> None:
        """Stop background reasoning."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("DreamLoop stopped after %d cycles", self._cycles)

    def pause(self) -> None:
        """Pause thinking (e.g., while user is talking)."""
        self._paused = True

    def resume(self) -> None:
        """Resume thinking."""
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def stats(self) -> dict:
        return {
            "cycles": self._cycles,
            "total_chains": self._total_chains,
            "total_insights": self._total_insights,
            "queued_insights": self._insights.qsize(),
            "atoms_tracked": len(self._curiosity),
        }

    # ── Insight access ────────────────────────────────────────────────

    def drain_insights(self) -> list[Insight]:
        """Drain all queued insights. Non-blocking."""
        insights = []
        while True:
            try:
                insights.append(self._insights.get_nowait())
            except Empty:
                break
        return insights

    def peek_insights(self) -> int:
        """How many insights are waiting."""
        return self._insights.qsize()

    def _surface(self, insight: Insight) -> None:
        """Queue an insight for the user."""
        try:
            self._insights.put_nowait(insight)
            self._total_insights += 1
            logger.debug("Insight: %s", insight)
        except Exception:
            # Queue full — drop oldest
            try:
                self._insights.get_nowait()
                self._insights.put_nowait(insight)
            except Exception:
                pass

    # ── Main loop ─────────────────────────────────────────────────────

    def _think_loop(self) -> None:
        """Background thinking thread."""
        while self._running:
            if self._paused:
                time.sleep(0.5)
                continue

            try:
                self._think_cycle()
            except Exception as e:
                logger.error("DreamLoop error: %s", e, exc_info=True)

            self._cycles += 1
            time.sleep(self._cycle_interval)

    def _think_cycle(self) -> None:
        """One cycle of background reasoning."""
        atoms = self._forge.all_atoms()
        if len(atoms) < 2:
            return

        # Update curiosity scores
        self._update_curiosity(atoms)

        # Allocate budget across phases
        wander_budget = max(1, self._cycle_budget // 2)
        hunt_budget = max(1, self._cycle_budget // 4)
        converge_budget = self._cycle_budget - wander_budget - hunt_budget

        # Phase 1: Wander — explore curious atoms
        with self._lock:
            self._wander(wander_budget)

        # Phase 2: Hunt — look for contradictions
        with self._lock:
            self._hunt_contradictions(hunt_budget)

        # Phase 3: Converge — check for convergent conclusions
        with self._lock:
            self._check_convergence(converge_budget)

    # ── Curiosity ─────────────────────────────────────────────────────

    def _update_curiosity(self, atoms) -> None:
        """Recalculate curiosity scores for all atoms."""
        now = time.time()

        for atom in atoms:
            if atom.kind == "role":
                continue  # skip structural role atoms

            name = atom.name
            if name not in self._curiosity:
                self._curiosity[name] = _CuriosityState(name=name)

            state = self._curiosity[name]

            # Factors that increase curiosity:
            # 1. Freshness — recently created atoms are interesting
            freshness = max(0, 1.0 - state.explore_count * 0.1)

            # 2. Low connectivity — isolated atoms need exploration
            connections = self._facts._get_connections(name)
            state.connection_count = len(connections)
            isolation = 1.0 / (1.0 + len(connections))

            # 3. Time since last explored — forgotten things become curious again
            time_factor = min(1.0, (now - state.last_explored) / 60.0) if state.last_explored > 0 else 1.0

            # 4. Contradiction involvement — unresolved conflict is interesting
            contradiction_bonus = min(0.5, state.contradiction_count * 0.2)

            # 5. Atom strength — strong atoms are more worth exploring
            strength_factor = atom.strength

            state.score = (
                freshness * 0.25
                + isolation * 0.20
                + time_factor * 0.25
                + contradiction_bonus * 0.15
                + strength_factor * 0.15
            )

    def _pick_curious(self, n: int) -> list[str]:
        """Pick the n most curious atoms."""
        candidates = [
            s for s in self._curiosity.values()
            if s.score > 0.1
        ]
        candidates.sort(key=lambda s: s.score, reverse=True)

        # Add some randomness — don't always pick the top
        top = candidates[:n * 2]
        if len(top) > n:
            random.shuffle(top)
            top = top[:n]

        return [s.name for s in top]

    # ── Phase 1: Wander ───────────────────────────────────────────────

    def _wander(self, budget: int) -> None:
        """Pick curious atoms, reason from them, discover connections."""
        curious = self._pick_curious(budget)

        for atom_name in curious:
            if not self._running:
                return

            # Mark explored
            if atom_name in self._curiosity:
                self._curiosity[atom_name].last_explored = time.time()
                self._curiosity[atom_name].explore_count += 1

            # Open reasoning — find direct connections
            chains = self._loop.reason(atom_name, top_k=5)
            self._total_chains += len(chains)

            # Also try targeted reasoning: from this atom's connections,
            # follow their relations (this is how we find multi-hop chains)
            connections = self._facts._get_connections(atom_name)
            seen_relations = set()
            for _, rel, _ in connections:
                if rel not in seen_relations:
                    seen_relations.add(rel)
                    targeted = self._loop.reason(atom_name, rel, top_k=3)
                    chains.extend(targeted)
                    self._total_chains += len(targeted)

            for chain in chains:
                # Track for convergence detection
                if chain.answer:
                    self._convergence_map.setdefault(chain.answer, set()).add(atom_name)

                # Cap background pattern strength
                self._cap_background_patterns(chain)

                if chain.depth < 2:
                    continue

                # New multi-hop connection?
                key = (atom_name, chain.answer or "")
                if key not in self._seen_connections:
                    self._seen_connections.add(key)
                    self._surface(Insight(
                        kind=InsightKind.CONNECTION,
                        summary=self._describe_chain(chain),
                        atoms=[atom_name, chain.answer or ""],
                        confidence=chain.confidence,
                        chain_depth=chain.depth,
                    ))

    # ── Phase 2: Hunt contradictions ──────────────────────────────────

    def _hunt_contradictions(self, budget: int) -> None:
        """Actively look for atoms with contradictory facts."""
        curious = self._pick_curious(budget)

        for atom_name in curious:
            if not self._running:
                return

            # Get all relations for this atom
            connections = self._facts._get_connections(atom_name)
            relations = set(rel for _, rel, _ in connections)

            for relation in relations:
                chains = self._loop.reason(atom_name, relation, top_k=5)
                self._total_chains += len(chains)

                contradicted = [c for c in chains if c.contradicted]
                if contradicted:
                    answers = [c.answer for c in chains if c.answer]
                    if atom_name in self._curiosity:
                        self._curiosity[atom_name].contradiction_count += 1

                    self._surface(Insight(
                        kind=InsightKind.CONTRADICTION,
                        summary=f"{atom_name} {relation} has conflicting answers: {', '.join(set(answers))}",
                        atoms=[atom_name] + list(set(answers)),
                        confidence=max(c.confidence for c in contradicted),
                    ))

    # ── Phase 3: Convergence ──────────────────────────────────────────

    def _check_convergence(self, budget: int) -> None:
        """Check if different starting points reach the same conclusion."""
        # Find answers reached from multiple starting atoms
        convergent = {
            answer: sources
            for answer, sources in self._convergence_map.items()
            if len(sources) >= 2
        }

        checked = 0
        for answer, sources in convergent.items():
            if checked >= budget or not self._running:
                break

            source_list = sorted(sources)
            key = (answer, tuple(source_list))

            # Self-reinforce: if multiple paths reach same conclusion,
            # slightly strengthen the patterns (but capped)
            for source in source_list:
                chains = self._loop.reason(source, top_k=3)
                self._total_chains += len(chains)
                checked += 1

                for chain in chains:
                    if chain.answer == answer and chain.pathway_name:
                        self._self_reinforce(chain.pathway_name)

            if len(source_list) >= 2:
                self._surface(Insight(
                    kind=InsightKind.CONVERGENCE,
                    summary=f"{', '.join(source_list[:3])} all lead to {answer}",
                    atoms=source_list[:3] + [answer],
                    confidence=0.5,  # convergence is medium confidence
                ))

    # ── Helpers ────────────────────────────────────────────────────────

    def _describe_chain(self, chain) -> str:
        """Human-readable description of a reasoning chain."""
        hops = " -> ".join(f"{s} {r} {o}" for s, r, o, _ in chain.hops)
        return f"{hops} (confidence: {chain.confidence:.3f})"

    def _cap_background_patterns(self, chain) -> None:
        """Ensure background-discovered patterns don't exceed the cap."""
        if chain.pathway_name and chain.pathway_name in self._loop.library._pathways:
            p = self._loop.library._pathways[chain.pathway_name]
            if p.strength > self.SELF_REINFORCE_CAP:
                p.strength = self.SELF_REINFORCE_CAP

    def _self_reinforce(self, pathway_name: str) -> None:
        """Slightly strengthen a pattern that keeps appearing — but capped."""
        if pathway_name in self._loop.library._pathways:
            p = self._loop.library._pathways[pathway_name]
            # Small self-reinforcement with hard cap
            if p.strength < self.SELF_REINFORCE_CAP:
                p.strength = min(self.SELF_REINFORCE_CAP, p.strength + 0.05)

    # ── Questions (gap detection) ─────────────────────────────────────

    def find_gaps(self) -> list[Insight]:
        """Find knowledge gaps — atoms connected to each other but
        missing an expected transitive link.

        E.g., if A→B and B→C exist but A has no connection to C,
        that's a gap worth asking about.

        This is called on-demand, not in the background loop.
        """
        gaps = []
        atoms = self._forge.all_atoms(kind=None)

        for atom in atoms[:20]:  # limit scope
            if atom.kind == "role":
                continue

            connections = self._facts._get_connections(atom.name)
            for obj, rel, score in connections:
                # Check if obj connects to things that atom doesn't
                obj_connections = self._facts._get_connections(obj)
                for obj2, rel2, score2 in obj_connections:
                    if obj2 == atom.name:
                        continue
                    # Does atom connect to obj2?
                    atom_connections = {o for o, _, _ in self._facts._get_connections(atom.name)}
                    if obj2 not in atom_connections:
                        gaps.append(Insight(
                            kind=InsightKind.QUESTION,
                            summary=f"Does {atom.name} {rel2} {obj2}? ({atom.name} -> {obj} -> {obj2})",
                            atoms=[atom.name, obj, obj2],
                            confidence=score * score2,
                            chain_depth=2,
                        ))

        return gaps[:10]  # limit results
