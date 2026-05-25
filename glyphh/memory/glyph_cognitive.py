"""
GlyphCognitiveLoop — Ada's glyph-based reasoning engine.

Replaces the old CognitiveLoop (atom/fact bind chains) with reasoning
over structured Thought Glyphs. Instead of following S-R-O hops through
a fact graph, reasoning is layered similarity search with multi-hop
associative recall.

Pathways shift from traversal routes to activation patterns — which
layers co-activated during successful reasoning. These patterns are
generative: a pathway learned from "what is my name?" (perspective/self
+ semantic/identity → relational/equals) also fires for "what is my
favorite color?" (perspective/self + semantic/emotion → relational/equals).

Three mechanisms:
  1. Reason — multi-hop glyph recall through layer associations
  2. Pattern — encode layer activation patterns as pathways
  3. Reinforce — Hebbian strengthening of confirmed patterns

Usage:
    from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop

    loop = GlyphCognitiveLoop(thought_space)
    results = loop.reason("what is my name")
    loop.confirm()  # strengthens the activation pathway
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity
from glyphh.state.pathway import Pathway, PathwayEncoder, PathwayLibrary
from glyphh.memory.thought_space import ThoughtGlyphSpace, RecallResult, StoredThought

logger = logging.getLogger(__name__)


# ── Layer activation snapshot ────────────────────────────────────────────

def _activation_vector(
    recall: RecallResult,
    dimension: int,
    space_id: str,
) -> np.ndarray:
    """Encode which layers fired during a recall hit as a single HDC vector.

    Binds each active layer's cortex together. This is the "activation
    pattern" — the structural fingerprint of how this recall matched.
    Active = layer similarity above threshold (positive signal).
    """
    active_cortices = []
    for layer_name, sim in recall.layer_similarities.items():
        if sim > 0.05 and layer_name in recall.thought.glyph.layers:
            active_cortices.append(
                recall.thought.glyph.layers[layer_name].cortex.data
            )

    if not active_cortices:
        # Fallback: use global cortex
        return recall.thought.glyph.global_cortex.data.copy()

    if len(active_cortices) == 1:
        return active_cortices[0].copy()

    # Bind active layers together — order doesn't matter for activation
    # patterns (bind is commutative for bipolar vectors)
    result = active_cortices[0]
    for cortex in active_cortices[1:]:
        result = bind(result, cortex)
    return result


# ── GlyphReasoningChain ─────────────────────────────────────────────────

@dataclass
class GlyphHop:
    """A single hop in glyph-based reasoning."""
    query: str                          # what was searched
    result: RecallResult                # what was found
    matching_layers: dict[str, float]   # layer → similarity


@dataclass
class GlyphReasoningChain:
    """A reasoning trace through thought glyph space.

    Each hop is a recall from thought space. The chain represents
    the associative path Ada followed to reach an answer.
    """
    hops: list[GlyphHop] = field(default_factory=list)
    confidence: float = 0.0
    pathway_vector: np.ndarray | None = None
    pathway_name: str | None = None
    pattern_boost: float = 0.0
    contradicted: bool = False

    @property
    def depth(self) -> int:
        return len(self.hops)

    @property
    def answer(self) -> StoredThought | None:
        return self.hops[-1].result.thought if self.hops else None

    @property
    def answer_text(self) -> str | None:
        return self.answer.content if self.answer else None

    def __repr__(self) -> str:
        chain_str = " → ".join(
            f"[{h.result.global_similarity:.2f}] {h.result.thought.content[:30]}"
            for h in self.hops
        )
        return f"GlyphChain({chain_str} | conf={self.confidence:.3f})"


# ── GlyphCognitiveLoop ──────────────────────────────────────────────────

class GlyphCognitiveLoop:
    """Ada's glyph-based reasoning engine.

    Reasons over ThoughtGlyphSpace using layered similarity. Pathways
    encode layer activation patterns — which layers co-activated during
    successful reasoning — not traversal routes through facts.

    Args:
        thought_space: ThoughtGlyphSpace for recall.
        dimension: Vector dimension (must match thought space).
        max_depth: Maximum reasoning hops.
        min_similarity: Minimum similarity to follow a hop.
    """

    def __init__(
        self,
        thought_space: ThoughtGlyphSpace,
        dimension: int = 2000,
        seed: int = 42,
        decay: float = 0.75,
        max_depth: int = 2,
        min_similarity: float = 0.05,
    ) -> None:
        self._space = thought_space
        self._dim = dimension
        self._seed = seed
        self._max_depth = max_depth
        self._min_similarity = min_similarity

        self._library = PathwayLibrary(
            dimension=dimension, seed=seed, decay=decay,
        )

        self._last_chains: list[GlyphReasoningChain] = []
        self._reason_count: int = 0

    @property
    def library(self) -> PathwayLibrary:
        return self._library

    @property
    def thought_space(self) -> ThoughtGlyphSpace:
        return self._space

    # ── Core reasoning ────────────────────────────────────────────────────

    def reason(
        self,
        query: str,
        top_k: int = 5,
        speaker: str = "incoming",
        semantic: bool | None = None,
    ) -> list[GlyphReasoningChain]:
        """Multi-hop glyph reasoning.

        1. Recall thoughts matching the query (hop 1)
        2. For top hits, derive secondary queries from their content
        3. Recall again from secondary queries (hop 2)
        4. Encode layer activation patterns as pathways
        5. Boost chains that match known activation patterns

        Args:
            query: Natural language query.
            top_k: Max chains to return.
            speaker: Speaker context.
            semantic: Override the space's semantic-recall flag (the dream
                loop passes False to stay HDC-only and skip Qwen3 inference).

        Returns:
            List of GlyphReasoningChain sorted by confidence.
        """
        self._reason_count += 1
        # Gentle pathway decay each reasoning cycle
        if len(self._library) > 0:
            self._library.decay_all(0.995)

        chains: list[GlyphReasoningChain] = []

        # Hop 1: direct recall
        results = self._space.recall(
            query, top_k=top_k * 2, speaker=speaker, semantic=semantic,
        )
        seen_thoughts: set[str] = set()

        for result in results:
            if result.global_similarity < self._min_similarity:
                continue

            tid = result.thought.thought_id
            seen_thoughts.add(tid)

            hop = GlyphHop(
                query=query,
                result=result,
                matching_layers=result.layer_similarities,
            )

            chain = GlyphReasoningChain(
                hops=[hop],
                confidence=result.global_similarity,
            )

            # Encode activation pattern as pathway vector
            chain.pathway_vector = _activation_vector(
                result, self._dim, self._space.encoder.space_id,
            )
            chain.pattern_boost = self._check_pattern(chain)

            chains.append(chain)

        # Hop 2: associative recall from top hits
        if self._max_depth >= 2 and chains:
            for parent_chain in chains[:3]:  # follow top 3
                parent = parent_chain.answer
                if parent is None:
                    continue

                # Use the recalled thought's content as a secondary query
                secondary_results = self._space.recall(
                    parent.content, top_k=3, speaker=parent.speaker,
                    semantic=semantic,
                )

                for result in secondary_results:
                    tid = result.thought.thought_id
                    if tid in seen_thoughts:
                        continue
                    if result.global_similarity < self._min_similarity:
                        continue
                    seen_thoughts.add(tid)

                    hop2 = GlyphHop(
                        query=parent.content,
                        result=result,
                        matching_layers=result.layer_similarities,
                    )

                    # Build 2-hop chain
                    encoder = PathwayEncoder(self._dim, self._seed, 0.75)
                    encoder.update(parent_chain.pathway_vector)
                    hop2_activation = _activation_vector(
                        result, self._dim, self._space.encoder.space_id,
                    )
                    encoder.update(hop2_activation)

                    multi_chain = GlyphReasoningChain(
                        hops=[parent_chain.hops[0], hop2],
                        confidence=(
                            parent_chain.confidence * 0.6
                            + result.global_similarity * 0.4
                        ),
                        pathway_vector=encoder.get_state(),
                    )
                    multi_chain.pattern_boost = self._check_pattern(multi_chain)
                    chains.append(multi_chain)

        # Detect contradictions
        self._detect_contradictions(chains)

        # Sort by confidence + pattern boost
        chains.sort(
            key=lambda c: c.confidence * (1.0 + c.pattern_boost),
            reverse=True,
        )

        self._last_chains = chains[:top_k]
        return self._last_chains

    # ── Pattern matching ──────────────────────────────────────────────────

    def _check_pattern(self, chain: GlyphReasoningChain) -> float:
        """Check if a chain's activation pattern matches a known pathway."""
        if chain.pathway_vector is None:
            return 0.0

        matches = self._library.match(chain.pathway_vector, top_k=1)
        if matches:
            pathway, score = matches[0]
            if score > 0.1:
                chain.pathway_name = pathway.name
                return score * pathway.strength
        return 0.0

    def _store_pattern(
        self,
        chain: GlyphReasoningChain,
        strength: float = 0.5,
    ) -> str:
        """Store a chain's activation pattern as a named pathway."""
        # Name from content hash — same content = same pattern name
        content_sig = "|".join(
            h.result.thought.content for h in chain.hops
        )
        name = f"glyph_{hashlib.md5(content_sig.encode()).hexdigest()[:8]}"

        if chain.pathway_vector is not None and name not in self._library:
            self._library._pathways[name] = Pathway(
                name=name,
                vector=chain.pathway_vector,
                strength=strength,
            )
            logger.debug("Stored activation pattern: %s (strength=%.2f)", name, strength)

        chain.pathway_name = name
        return name

    # ── Contradiction detection ───────────────────────────────────────────

    def _detect_contradictions(
        self,
        chains: list[GlyphReasoningChain],
    ) -> None:
        """Flag chains whose answers have low global cortex similarity.

        If two chains answer the same query with very different thoughts,
        one may contradict the other.
        """
        if len(chains) < 2:
            return

        # Compare top chains pairwise
        for i in range(min(5, len(chains))):
            for j in range(i + 1, min(5, len(chains))):
                a = chains[i].answer
                b = chains[j].answer
                if a is None or b is None:
                    continue

                sim = float(cosine_similarity(
                    a.glyph.global_cortex.data,
                    b.glyph.global_cortex.data,
                ))

                # Very different answers to the same query
                if sim < -0.1:
                    # Weaker chain is the contradiction
                    if chains[i].confidence < chains[j].confidence:
                        chains[i].contradicted = True
                    else:
                        chains[j].contradicted = True

    # ── Reinforcement ─────────────────────────────────────────────────────

    def confirm(self, chain: GlyphReasoningChain | None = None) -> None:
        """Confirm a reasoning chain — Hebbian strengthening.

        Strengthens:
          1. The activation pathway pattern
          2. The recalled thoughts (Hebbian reinforcement)
        """
        if chain is None:
            chain = self._last_chains[0] if self._last_chains else None
        if chain is None:
            return

        # Strengthen or create pathway pattern
        name = chain.pathway_name
        if name and name in self._library:
            p = self._library._pathways[name]
            if p.strength < 0.7:
                p.strength = 0.8
            self._library.strengthen(name, amount=0.15)
        else:
            name = self._store_pattern(chain, strength=0.8)

        # Reinforce the thoughts that were recalled
        for hop in chain.hops:
            self._space.reinforce(hop.result.thought.thought_id, amount=0.1)

        logger.info("Confirmed glyph chain: %s → %s", name, chain.answer_text)

    def reject(self, chain: GlyphReasoningChain | None = None) -> None:
        """Reject a reasoning chain — weaken the pathway."""
        if chain is None:
            chain = self._last_chains[0] if self._last_chains else None
        if chain is None:
            return

        name = chain.pathway_name
        if name and name in self._library:
            self._library.weaken(name, factor=0.7)
            logger.info("Rejected glyph chain: %s (weakened)", name)
        elif chain.pathway_vector is not None:
            name = self._store_pattern(chain, strength=0.2)
            logger.info("Rejected glyph chain: %s (stored weak)", name)

    # ── Recall for LLM prompt ────────────────────────────────────────────

    def recall(self, text: str, top_k: int = 5) -> str:
        """Reason over thought glyphs and format for LLM injection.

        This is the main entry point for the CLI — replaces the old
        CognitiveLoop.recall() that returned formatted fact lines.
        """
        chains = self.reason(text, top_k=top_k)
        if not chains:
            return ""

        # Deduplicate: each thought appears once (from its best chain)
        seen_ids: set[str] = set()
        unique_results: list[RecallResult] = []
        for chain in chains:
            tid = chain.hops[0].result.thought.thought_id
            if tid not in seen_ids:
                seen_ids.add(tid)
                unique_results.append(chain.hops[0].result)
        recall_str = self._space.format_recall(unique_results, max_results=top_k)

        # Add multi-hop inferences
        for chain in chains:
            if chain.depth > 1 and not chain.contradicted:
                hop_str = " → ".join(
                    h.result.thought.content[:40] for h in chain.hops
                )
                recall_str += (
                    f"\nAda inferred: {hop_str} "
                    f"(confidence: {chain.confidence:.2f})"
                )

        return recall_str

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the activation pathway library to disk."""
        self._library.prune(min_strength=0.15)
        self._library.save(Path(path))

    def load(self, path: str | Path) -> None:
        """Load the activation pathway library from disk."""
        self._library.load(Path(path))

    # ── Introspection ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "reason_count": self._reason_count,
            "pathways": len(self._library),
            "thoughts": self._space.count,
        }
