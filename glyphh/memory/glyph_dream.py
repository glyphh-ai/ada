"""
GlyphDreamLoop — Ada's dual-loop background reasoning engine.

Two concurrent loops, like REM and slow-wave sleep:

  Localized (REM — fast, reactive):
    Interval: ~3s during conversation
    Working set: ~50 recent thoughts
    Phases: Wander → Hunt → Converge
    Discovers local patterns and contradictions

  Deep (Slow-Wave — slow, reflective):
    Interval: ~30s, or when idle
    Working set: ~200 thoughts broadly sampled
    Phases: Survey → Connect → Generate → Crystallize → Prune
    Discovers cross-memory patterns, crystallizes compound primitives

Both loops share the ThoughtGlyphSpace (concurrent-safe: in-memory dict).
Localized has priority (serving conversation). Deep runs when idle.
Crystallized primitives from deep loop are immediately available.

Guardrails:
  - Background discoveries capped at 0.5 strength
  - Deep loop pauses during heavy conversation
  - Insights are queued, not acted on
  - Crystallization requires re-derivation from diverse starting pairs

Usage:
    dream = GlyphDreamLoop(glyph_loop)
    dream.start()
    insights = dream.drain_insights()
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

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity
from glyphh.memory.thought_space import ThoughtGlyphSpace, StoredThought, RecallResult
from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop, GlyphReasoningChain
from glyphh.memory.primitives import PrimitiveSpace

logger = logging.getLogger(__name__)


# ── Insight types (same as legacy — backward compatible) ─────────────────

class InsightKind(Enum):
    CONNECTION = "connection"
    CONTRADICTION = "contradiction"
    CONVERGENCE = "convergence"
    QUESTION = "question"
    CRYSTALLIZATION = "crystallization"  # new: deep loop minted a primitive


@dataclass
class Insight:
    kind: InsightKind
    summary: str
    atoms: list[str]
    confidence: float = 0.0
    chain_depth: int = 0
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"Insight({self.kind.value}: {self.summary})"


# ── Curiosity scoring ────────────────────────────────────────────────────

@dataclass
class _ThoughtCuriosity:
    """Per-thought curiosity tracking."""
    thought_id: str
    score: float = 1.0
    last_explored: float = 0.0
    explore_count: int = 0
    recall_hit_count: int = 0  # how often this thought appears in recalls


# ── GlyphDreamLoop ───────────────────────────────────────────────────────

class GlyphDreamLoop:
    """Ada's dual-loop background reasoning over thought glyphs.

    Args:
        glyph_loop: GlyphCognitiveLoop for reasoning.
        localized_interval: Seconds between localized cycles.
        deep_interval: Seconds between deep cycles.
        localized_size: Working set size for localized loop.
        deep_size: Working set size for deep loop.
        self_reinforce_cap: Max strength for self-discovered patterns.
    """

    SELF_REINFORCE_CAP = 0.5

    def __init__(
        self,
        glyph_loop: GlyphCognitiveLoop,
        localized_interval: float = 3.0,
        deep_interval: float = 30.0,
        localized_size: int = 50,
        deep_size: int = 200,
        max_insights: int = 50,
        self_reinforce_cap: float = 0.5,
    ) -> None:
        self._loop = glyph_loop
        self._space = glyph_loop.thought_space

        self._localized_interval = localized_interval
        self._deep_interval = deep_interval
        self._localized_size = localized_size
        self._deep_size = deep_size
        self._max_insights = max_insights
        self.SELF_REINFORCE_CAP = self_reinforce_cap

        # Insight queue + persistent log
        self._insights: Queue[Insight] = Queue(maxsize=max_insights)
        self._insight_log: list[Insight] = []
        self._max_log = 200
        self._seen_insights: set[str] = set()

        # Curiosity
        self._curiosity: dict[str, _ThoughtCuriosity] = {}

        # Deep loop state
        self._crystallization_candidates: dict[str, int] = {}  # content_hash → count
        self._convergence_map: dict[str, set[str]] = {}  # answer_content → set of query_contents

        # Threads
        self._localized_thread: threading.Thread | None = None
        self._deep_thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

        # Stats
        self._localized_cycles = 0
        self._deep_cycles = 0
        self._total_chains = 0
        self._total_insights = 0
        self._last_absorb_time = 0.0  # tracks conversation activity

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._paused = False

        self._localized_thread = threading.Thread(
            target=self._localized_loop, daemon=True, name="ada-rem",
        )
        self._deep_thread = threading.Thread(
            target=self._deep_loop, daemon=True, name="ada-slowwave",
        )
        self._localized_thread.start()
        self._deep_thread.start()
        logger.info("GlyphDreamLoop started (localized + deep)")

    def stop(self) -> None:
        self._running = False
        for thread in (self._localized_thread, self._deep_thread):
            if thread:
                thread.join(timeout=5.0)
        self._localized_thread = None
        self._deep_thread = None
        logger.info(
            "GlyphDreamLoop stopped: %d localized + %d deep cycles",
            self._localized_cycles, self._deep_cycles,
        )

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def stats(self) -> dict:
        return {
            "cycles": self._localized_cycles + self._deep_cycles,
            "localized_cycles": self._localized_cycles,
            "deep_cycles": self._deep_cycles,
            "total_chains": self._total_chains,
            "total_insights": self._total_insights,
            "queued_insights": self._insights.qsize(),
            "thoughts": self._space.count,
            "pathways": len(self._loop.library),
            "crystallization_candidates": len(self._crystallization_candidates),
        }

    def notify_absorb(self) -> None:
        """Called when a new thought is absorbed — signals conversation activity."""
        self._last_absorb_time = time.time()

    # ── Insight access ────────────────────────────────────────────────

    def drain_insights(self) -> list[Insight]:
        insights = []
        while True:
            try:
                insights.append(self._insights.get_nowait())
            except Empty:
                break
        return insights

    def peek_insights(self) -> int:
        return self._insights.qsize()

    def recall_insights(self, words: set[str], max_results: int = 5) -> list[Insight]:
        if not words or not self._insight_log:
            return []
        scored = []
        for insight in self._insight_log:
            matches = sum(1 for w in words if any(w in a for a in insight.atoms))
            if matches > 0:
                scored.append((insight, matches))
        scored.sort(key=lambda x: (-x[1], -x[0].confidence))
        return [i for i, _ in scored[:max_results]]

    def _surface(self, insight: Insight) -> None:
        if insight.summary in self._seen_insights:
            return
        self._seen_insights.add(insight.summary)
        self._insight_log.append(insight)
        if len(self._insight_log) > self._max_log:
            self._insight_log = self._insight_log[-self._max_log:]
        try:
            self._insights.put_nowait(insight)
            self._total_insights += 1
        except Exception:
            try:
                self._insights.get_nowait()
                self._insights.put_nowait(insight)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════
    # LOCALIZED LOOP (REM — fast, reactive)
    # ══════════════════════════════════════════════════════════════════════

    def _localized_loop(self) -> None:
        while self._running:
            if self._paused or self._space.count < 2:
                time.sleep(0.5)
                continue
            try:
                self._localized_cycle()
            except Exception as e:
                logger.error("Localized loop error: %s", e, exc_info=True)
            self._localized_cycles += 1
            time.sleep(self._localized_interval)

    def _localized_cycle(self) -> None:
        """One cycle: Wander → Hunt → Converge over recent thoughts."""
        working_set = self._space.load_working_set(
            strategy="recent", size=self._localized_size,
        )
        if len(working_set) < 2:
            return

        self._update_curiosity(working_set)

        # Phase 1: Wander — explore curious thoughts
        with self._lock:
            self._wander(working_set)

        # Phase 2: Hunt — look for contradictions
        with self._lock:
            self._hunt_contradictions(working_set)

        # Phase 3: Converge — check if different queries reach same thought
        with self._lock:
            self._check_convergence()

    # ── Phase 1: Wander ──────────────────────────────────────────────

    def _wander(self, working_set: list[StoredThought]) -> None:
        """Pick curious thoughts, use them as queries, discover connections."""
        curious = self._pick_curious(3)
        if not curious:
            # Fallback: random from working set
            curious = random.sample(working_set, min(3, len(working_set)))

        for thought in curious:
            if not self._running:
                return

            # Mark explored
            tid = thought.thought_id
            if tid in self._curiosity:
                self._curiosity[tid].last_explored = time.time()
                self._curiosity[tid].explore_count += 1

            # Reason from this thought's content
            chains = self._loop.reason(thought.content, top_k=5)
            self._total_chains += len(chains)

            for chain in chains:
                # Cap background-discovered pattern strength
                self._cap_pattern(chain)

                # Track for convergence
                if chain.answer:
                    self._convergence_map.setdefault(
                        chain.answer.content, set()
                    ).add(thought.content)

                # Surface multi-hop discoveries
                if chain.depth >= 2 and not chain.contradicted:
                    hop_str = " -> ".join(
                        h.result.thought.content[:30] for h in chain.hops
                    )
                    self._surface(Insight(
                        kind=InsightKind.CONNECTION,
                        summary=hop_str,
                        atoms=[h.result.thought.content[:30] for h in chain.hops],
                        confidence=chain.confidence,
                        chain_depth=chain.depth,
                    ))

    # ── Phase 2: Hunt contradictions ─────────────────────────────────

    def _hunt_contradictions(self, working_set: list[StoredThought]) -> None:
        """Find thoughts that give contradictory answers to similar queries."""
        if len(working_set) < 3:
            return

        # Sample pairs and check for contradictions
        sample = random.sample(working_set, min(4, len(working_set)))
        for thought in sample:
            if not self._running:
                return

            chains = self._loop.reason(thought.content, top_k=5)
            self._total_chains += len(chains)

            contradicted = [c for c in chains if c.contradicted]
            if contradicted:
                answers = [c.answer_text for c in chains if c.answer_text]
                self._surface(Insight(
                    kind=InsightKind.CONTRADICTION,
                    summary=f"Conflicting answers for '{thought.content[:30]}': {', '.join(set(a[:20] for a in answers[:3]))}",
                    atoms=[thought.content[:30]] + [a[:20] for a in answers[:3]],
                    confidence=max(c.confidence for c in contradicted),
                ))

    # ── Phase 3: Convergence ─────────────────────────────────────────

    def _check_convergence(self) -> None:
        """Check if different starting thoughts reach the same conclusion."""
        convergent = {
            answer: sources
            for answer, sources in self._convergence_map.items()
            if len(sources) >= 2
        }

        for answer, sources in list(convergent.items())[:3]:
            source_list = sorted(sources)[:3]

            # Self-reinforce converging pathways (capped)
            for source in source_list:
                chains = self._loop.reason(source, top_k=3)
                self._total_chains += len(chains)
                for chain in chains:
                    if chain.answer and chain.answer.content == answer:
                        self._self_reinforce(chain)

            self._surface(Insight(
                kind=InsightKind.CONVERGENCE,
                summary=f"'{', '.join(s[:20] for s in source_list)}' all recall '{answer[:30]}'",
                atoms=source_list[:3] + [answer[:30]],
                confidence=0.5,
            ))

    # ══════════════════════════════════════════════════════════════════════
    # DEEP LOOP (Slow-Wave — slow, reflective)
    # ══════════════════════════════════════════════════════════════════════

    def _deep_loop(self) -> None:
        # Wait for enough thoughts before starting deep reasoning
        while self._running:
            if self._paused or self._space.count < 5:
                time.sleep(2.0)
                continue

            # Yield to localized loop during active conversation
            if time.time() - self._last_absorb_time < 10.0:
                time.sleep(5.0)
                continue

            try:
                self._deep_cycle()
            except Exception as e:
                logger.error("Deep loop error: %s", e, exc_info=True)
            self._deep_cycles += 1
            time.sleep(self._deep_interval)

    def _deep_cycle(self) -> None:
        """One deep cycle: Survey → Connect → Generate → Crystallize → Prune."""

        # Phase 4: Survey — broadly sample from memory
        working_set = self._survey()
        if len(working_set) < 3:
            return

        # Phase 5: Connect — find structural similarities across distant thoughts
        with self._lock:
            connections = self._connect(working_set)

        # Phase 6: Generate — manifold interpolation between similar pairs
        with self._lock:
            self._generate(connections)

        # Phase 7: Crystallize — mint compound primitives from re-derived patterns
        with self._lock:
            self._crystallize()

        # Phase 8: Prune — global decay sweep
        with self._lock:
            self._prune()

    # ── Phase 4: Survey ──────────────────────────────────────────────

    def _survey(self) -> list[StoredThought]:
        """Sample broadly from memory using diverse strategies."""
        thoughts: list[StoredThought] = []
        n = self._deep_size

        # Mix strategies for diversity
        recent = self._space.load_working_set("recent", size=n // 3)
        weak = self._space.load_working_set("weak", size=n // 3)
        rand = self._space.load_working_set("random", size=n // 3)

        # Deduplicate
        seen: set[str] = set()
        for t in recent + weak + rand:
            if t.thought_id not in seen:
                seen.add(t.thought_id)
                thoughts.append(t)

        return thoughts

    # ── Phase 5: Connect ─────────────────────────────────────────────

    def _connect(
        self,
        working_set: list[StoredThought],
    ) -> list[tuple[StoredThought, StoredThought, float]]:
        """Find structurally similar but content-different thought pairs.

        These are the most interesting pairs for generative reasoning —
        same structure, different content. Like "my name is chris" and
        "my favorite color is blue" share perspective/self + relational/equals
        but differ in semantic content.
        """
        pairs: list[tuple[StoredThought, StoredThought, float]] = []

        # Sample pairs and compare layer structure
        sample = working_set[:min(20, len(working_set))]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                if not self._running:
                    return pairs

                a, b = sample[i], sample[j]
                global_sim = float(cosine_similarity(
                    a.glyph.global_cortex.data,
                    b.glyph.global_cortex.data,
                ))

                # Interesting: moderate global similarity (structurally related
                # but not identical)
                if 0.05 < global_sim < 0.6:
                    pairs.append((a, b, global_sim))

        # Sort by similarity — most structurally related first
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:10]

    # ── Phase 6: Generate ────────────────────────────────────────────

    def _generate(
        self,
        connections: list[tuple[StoredThought, StoredThought, float]],
    ) -> None:
        """Manifold interpolation between structurally similar thoughts.

        Interpolate between two thought cortex vectors. If the interpolated
        point is close to an existing thought, that's convergent evidence.
        If it's far from everything, it's a region Ada hasn't explored.
        """
        for a, b, sim in connections[:5]:
            if not self._running:
                return

            # Interpolate at midpoint
            interp = np.sign(
                0.5 * a.glyph.global_cortex.data.astype(np.float32)
                + 0.5 * b.glyph.global_cortex.data.astype(np.float32)
            ).astype(np.int8)
            # Fix zeros (sign(0) = 0, need bipolar)
            interp[interp == 0] = 1

            # Find nearest existing thought to the interpolated point
            best_sim = -1.0
            best_thought: StoredThought | None = None
            for stored in self._space.all_thoughts()[:50]:
                if stored.thought_id in (a.thought_id, b.thought_id):
                    continue
                s = float(cosine_similarity(interp, stored.glyph.global_cortex.data))
                if s > best_sim:
                    best_sim = s
                    best_thought = stored

            if best_thought and best_sim > 0.15:
                # The interpolated point is near an existing thought —
                # this is convergent evidence for a structural pattern
                key = f"{a.content[:20]}|{b.content[:20]}|{best_thought.content[:20]}"
                self._crystallization_candidates[key] = (
                    self._crystallization_candidates.get(key, 0) + 1
                )

                if self._crystallization_candidates[key] >= 2:
                    self._surface(Insight(
                        kind=InsightKind.CONNECTION,
                        summary=(
                            f"'{a.content[:25]}' and '{b.content[:25]}' "
                            f"structurally bridge to '{best_thought.content[:25]}'"
                        ),
                        atoms=[a.content[:20], b.content[:20], best_thought.content[:20]],
                        confidence=best_sim,
                    ))

    # ── Phase 7: Crystallize ─────────────────────────────────────────

    def _crystallize(self) -> None:
        """Mint compound primitives from patterns re-derived multiple times.

        When a structural pattern appears from diverse starting pairs
        across multiple deep cycles, it's stable enough to become a
        permanent compound primitive. The bridge thought's glyph IS
        the compound — register it as an exemplar and let HDC similarity
        do the rest.
        """
        to_crystallize = [
            (key, count) for key, count in self._crystallization_candidates.items()
            if count >= 3
        ]

        for key, count in to_crystallize:
            parts = key.split("|")
            if len(parts) != 3:
                del self._crystallization_candidates[key]
                continue

            # Find the bridge thought (parts[2]) by content prefix
            bridge_prefix = parts[2]
            bridge = self._find_thought_by_prefix(bridge_prefix)

            if bridge is None:
                del self._crystallization_candidates[key]
                continue

            # Extract content words from all three thoughts
            content_words = self._extract_content_words(parts)

            # The bridge glyph IS the compound — register it
            exemplar = self._space.primitives.add_compound(
                glyph=bridge.glyph,
                keywords=content_words,
                source=f"{parts[0]} ↔ {parts[1]} via {parts[2]} ({count}x)",
            )

            minted = f" → minted {exemplar.role}" if exemplar else ""
            self._surface(Insight(
                kind=InsightKind.CRYSTALLIZATION,
                summary=f"Crystallized: {parts[0]} ↔ {parts[1]} via {parts[2]} (seen {count}x){minted}",
                atoms=parts,
                confidence=min(1.0, count * 0.2),
            ))

            del self._crystallization_candidates[key]

    def _find_thought_by_prefix(self, prefix: str) -> StoredThought | None:
        """Find a thought whose content starts with the given prefix."""
        for thought in self._space.all_thoughts():
            if thought.content[:len(prefix)] == prefix:
                return thought
        return None

    def _extract_content_words(self, content_prefixes: list[str]) -> list[str]:
        """Extract content words (non-primitive) from crystallized thoughts."""
        from glyphh.memory.thought_glyph import _FILLER

        words: set[str] = set()
        primitives = self._space.primitives
        for prefix in content_prefixes:
            thought = self._find_thought_by_prefix(prefix)
            if thought is None:
                continue
            for w in thought.content.lower().split():
                w = w.strip("?.,!;:'\"()-")
                if not w or w in _FILLER:
                    continue
                # Content word = not already a known primitive
                if not primitives.match_roles(w):
                    words.add(w)
        return sorted(words)

    # ── Phase 8: Prune ───────────────────────────────────────────────

    def _prune(self) -> None:
        """Global decay sweep — weaken unused thoughts, prune dead ones."""
        pruned = self._space.decay_all(factor=0.98)
        if pruned > 0:
            logger.info("Deep loop pruned %d thoughts below threshold", pruned)

        # Also prune weak pathways
        if len(self._loop.library) > 0:
            removed = self._loop.library.prune(min_strength=0.1)
            if removed > 0:
                logger.info("Deep loop pruned %d weak pathways", removed)

    # ── Shared helpers ────────────────────────────────────────────────

    def _update_curiosity(self, working_set: list[StoredThought]) -> None:
        now = time.time()
        for thought in working_set:
            tid = thought.thought_id
            if tid not in self._curiosity:
                self._curiosity[tid] = _ThoughtCuriosity(thought_id=tid)

            state = self._curiosity[tid]
            freshness = max(0, 1.0 - state.explore_count * 0.1)
            time_factor = (
                min(1.0, (now - state.last_explored) / 60.0)
                if state.last_explored > 0 else 1.0
            )
            strength_factor = thought.strength

            state.score = (
                freshness * 0.30
                + time_factor * 0.30
                + strength_factor * 0.20
                + (1.0 / (1.0 + state.recall_hit_count)) * 0.20
            )

    def _pick_curious(self, n: int) -> list[StoredThought]:
        """Pick the n most curious thoughts."""
        candidates = sorted(
            self._curiosity.values(),
            key=lambda s: s.score,
            reverse=True,
        )
        # Add randomness
        top = candidates[:n * 2]
        if len(top) > n:
            random.shuffle(top)
            top = top[:n]

        result = []
        for state in top:
            thought = self._space.get_thought(state.thought_id)
            if thought:
                result.append(thought)
        return result

    def _cap_pattern(self, chain: GlyphReasoningChain) -> None:
        """Ensure background patterns don't exceed the self-reinforce cap."""
        if chain.pathway_name and chain.pathway_name in self._loop.library._pathways:
            p = self._loop.library._pathways[chain.pathway_name]
            if p.strength > self.SELF_REINFORCE_CAP:
                p.strength = self.SELF_REINFORCE_CAP

    def _self_reinforce(self, chain: GlyphReasoningChain) -> None:
        """Slightly strengthen a converging pattern — capped."""
        if chain.pathway_name and chain.pathway_name in self._loop.library._pathways:
            p = self._loop.library._pathways[chain.pathway_name]
            if p.strength < self.SELF_REINFORCE_CAP:
                p.strength = min(self.SELF_REINFORCE_CAP, p.strength + 0.05)

    # ── Questions (gap detection) ─────────────────────────────────────

    def find_gaps(self) -> list[Insight]:
        """Find knowledge gaps — thoughts that are structurally adjacent
        but missing an expected connection.

        If A→B and B→C exist (by layer similarity) but A and C are
        dissimilar, that's a gap worth asking about.
        """
        gaps = []
        thoughts = self._space.all_thoughts()[:20]

        for thought in thoughts:
            # Find thoughts similar to this one
            results = self._space.recall(thought.content, top_k=3)
            for r in results:
                if r.thought.thought_id == thought.thought_id:
                    continue
                # Find thoughts similar to the recalled thought
                sub_results = self._space.recall(r.thought.content, top_k=3)
                for sr in sub_results:
                    if sr.thought.thought_id in (thought.thought_id, r.thought.thought_id):
                        continue
                    # Is the original thought similar to this 2-hop thought?
                    hop2_sim = float(cosine_similarity(
                        thought.glyph.global_cortex.data,
                        sr.thought.glyph.global_cortex.data,
                    ))
                    if hop2_sim < 0.05:
                        gaps.append(Insight(
                            kind=InsightKind.QUESTION,
                            summary=(
                                f"'{thought.content[:25]}' connects to "
                                f"'{r.thought.content[:25]}' connects to "
                                f"'{sr.thought.content[:25]}' — but no direct link"
                            ),
                            atoms=[thought.content[:20], r.thought.content[:20], sr.thought.content[:20]],
                            confidence=r.global_similarity * sr.global_similarity,
                            chain_depth=2,
                        ))

        return gaps[:10]
