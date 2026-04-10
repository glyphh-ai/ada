"""
ThoughtGlyphSpace — Ada's long-term memory.

Orchestrates encoder + primitives + storage. Absorbs natural language
into structured Thought Glyphs, stores them, and recalls by layered
similarity search.

Two storage backends:
  - pgvector (production): HNSW-indexed, unlimited capacity
  - memory (dev/test): In-memory numpy, no external dependencies

Active recall always searches ALL stored thoughts (long-term memory).
The DreamLoop's working set is a separate in-memory subset.

Usage:
    from glyphh.memory.thought_space import ThoughtGlyphSpace

    space = ThoughtGlyphSpace()  # in-memory backend
    space.absorb("my name is chris", speaker="incoming")
    results = space.recall("what is my name", top_k=5)
    # → [("my name is chris", 0.37, glyph), ...]
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Glyph
from glyphh.memory.primitives import PrimitiveSpace
from glyphh.memory.thought_glyph import ThoughtGlyphEncoder

logger = logging.getLogger(__name__)


# ── Stored thought metadata ──────────────────────────────────────────────

@dataclass
class StoredThought:
    """A thought in long-term memory."""
    thought_id: str
    content: str
    speaker: str
    glyph: Glyph
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, amount: float = 0.1) -> None:
        """Hebbian strengthening with diminishing returns."""
        self.strength = min(3.0, self.strength + amount / (1.0 + 0.1 * self.strength))
        self.last_accessed = time.time()
        self.access_count += 1

    def decay(self, factor: float = 0.95) -> None:
        """Weaken unused thought."""
        self.strength = max(0.0, self.strength * factor)


@dataclass
class RecallResult:
    """A recall result with similarity scores."""
    thought: StoredThought
    global_similarity: float
    layer_similarities: dict[str, float] = field(default_factory=dict)


# ── ThoughtGlyphSpace ────────────────────────────────────────────────────

class ThoughtGlyphSpace:
    """Ada's long-term memory — absorb, store, recall.

    In-memory backend for now. pgvector backend will be added when
    Docker is running — the interface stays the same.

    Args:
        primitives: PrimitiveSpace (loaded). If None, creates and loads one.
        encoder: ThoughtGlyphEncoder. If None, creates one from primitives.
    """

    def __init__(
        self,
        primitives: PrimitiveSpace | None = None,
        encoder: ThoughtGlyphEncoder | None = None,
    ) -> None:
        # Primitives
        if primitives is None:
            primitives = PrimitiveSpace()
            primitives.load()
        self._primitives = primitives

        # Encoder
        if encoder is None:
            encoder = ThoughtGlyphEncoder(primitives=primitives)
        self._encoder = encoder

        # Storage (in-memory for now)
        self._thoughts: dict[str, StoredThought] = {}
        self._absorbed_texts: set[str] = set()  # dedup
        self._primitives_version: int = primitives.version  # track for re-encoding

    @property
    def primitives(self) -> PrimitiveSpace:
        return self._primitives

    @property
    def encoder(self) -> ThoughtGlyphEncoder:
        return self._encoder

    @property
    def count(self) -> int:
        return len(self._thoughts)

    # ── Absorb ────────────────────────────────────────────────────────────

    def absorb(
        self,
        text: str,
        speaker: str = "incoming",
        metadata: dict[str, Any] | None = None,
    ) -> StoredThought | None:
        """Encode a thought and store it in long-term memory.

        Deduplicates by exact text match. Returns None if already absorbed.

        Args:
            text: Natural language text.
            speaker: "incoming" (user) or "outgoing" (Ada).
            metadata: Optional metadata.

        Returns:
            StoredThought, or None if duplicate.
        """
        text_key = text.strip().lower()
        if not text_key:
            return None
        if text_key in self._absorbed_texts:
            return None

        glyph = self._encoder.encode_thought(text, speaker=speaker, metadata=metadata)
        thought_id = str(uuid.uuid4())[:12]

        stored = StoredThought(
            thought_id=thought_id,
            content=text,
            speaker=speaker,
            glyph=glyph,
            metadata=metadata or {},
        )

        self._thoughts[thought_id] = stored
        self._absorbed_texts.add(text_key)
        logger.debug("Absorbed thought %s: %s", thought_id, text[:50])
        return stored

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.01,
        speaker: str = "incoming",
    ) -> list[RecallResult]:
        """Search ALL long-term memory for thoughts matching a query.

        Three-signal search (following glyphh-code's layered re-ranking):
          1. Content cosine — pure content word match (primary signal, 0.50)
          2. Layer cosine — weighted per-layer comparison (structure, 0.35)
          3. Structural overlap — Jaccard of activated segments (bonus, 0.15)

        Content similarity is the dominant signal, like glyphh-code's
        content layer (0.50 weight). This avoids the global cortex noise
        problem where bundled vectors dilute specific word matches.

        Args:
            query: Natural language query.
            top_k: Max results to return.
            min_similarity: Minimum combined similarity threshold.
            speaker: Speaker context for the query encoding.

        Returns:
            List of RecallResult sorted by combined similarity.
        """
        if not self._thoughts:
            return []

        # Re-encode if primitives changed (crystallization added new compounds).
        # Keeps content vectors in sync so query↔stored similarity is symmetric.
        self._refresh_if_needed()

        query_glyph = self._encoder.encode_thought(query, speaker=speaker)

        # ── Signal 1: Content vector (primary — like glyphh-code content layer)
        query_content_vec = query_glyph.metadata.get("_content_vector")
        query_activated = set(query_glyph.metadata.get("_activated_attrs", []))

        _LAYER_WEIGHTS = {
            "perspective": 0.25,
            "semantic": 0.30,
            "relational": 0.25,
            "temporal": 0.10,
            "direction": 0.10,
        }

        results: list[RecallResult] = []
        for stored in self._thoughts.values():
            # ── Content similarity ──────────────────────────────
            content_sim = 0.0
            stored_content_vec = stored.glyph.metadata.get("_content_vector")
            if query_content_vec is not None and stored_content_vec is not None:
                content_sim = float(cosine_similarity(
                    query_content_vec, stored_content_vec,
                ))
                content_sim = max(0.0, content_sim)

            # ── Layer similarity (weighted per-layer role cosine) ──
            stored_activated = set(
                stored.glyph.metadata.get("_activated_attrs", [])
            )

            layer_sims: dict[str, float] = {}
            total_weighted_sim = 0.0
            total_weight = 0.0

            for layer_name, query_layer in query_glyph.layers.items():
                if layer_name not in stored.glyph.layers:
                    continue
                stored_layer = stored.glyph.layers[layer_name]
                layer_weight = _LAYER_WEIGHTS.get(layer_name, 0.1)

                layer_sim_sum = 0.0
                layer_seg_count = 0

                for seg_name, query_seg in query_layer.segments.items():
                    attr_key = f"{layer_name}_{seg_name}"
                    if attr_key not in query_activated:
                        continue
                    layer_seg_count += 1

                    if attr_key not in stored_activated:
                        continue
                    if seg_name not in stored_layer.segments:
                        continue
                    stored_seg = stored_layer.segments[seg_name]

                    for role_name, query_role_vec in query_seg.roles.items():
                        if role_name not in stored_seg.roles:
                            continue
                        stored_role_vec = stored_seg.roles[role_name]
                        rsim = float(cosine_similarity(
                            query_role_vec.data,
                            stored_role_vec.data,
                        ))
                        layer_sim_sum += max(0.0, rsim)

                if layer_seg_count > 0:
                    layer_sim = layer_sim_sum / layer_seg_count
                    layer_sims[layer_name] = layer_sim
                    total_weighted_sim += layer_sim * layer_weight
                    total_weight += layer_weight

            role_sim = total_weighted_sim / total_weight if total_weight > 0 else 0.0

            # ── Structural overlap (Jaccard of activated segments) ──
            seg_overlap = len(query_activated & stored_activated)
            seg_total = len(query_activated | stored_activated) or 1
            structural = seg_overlap / seg_total

            # ── Combine: adaptive weights (like glyphh-code re-ranking) ──
            # When query has distinctive content words, content dominates.
            # When query is all primitives ("who am i?"), lean on structure.
            n_content = len(query_glyph.metadata.get("_content_words", []))
            if n_content >= 2:
                w_content, w_role, w_struct = 0.55, 0.30, 0.15
            elif n_content == 1:
                w_content, w_role, w_struct = 0.40, 0.40, 0.20
            else:
                w_content, w_role, w_struct = 0.10, 0.65, 0.25

            combined = (
                content_sim * w_content
                + role_sim * w_role
                + structural * w_struct
            )

            if combined >= min_similarity:
                results.append(RecallResult(
                    thought=stored,
                    global_similarity=combined,
                    layer_similarities=layer_sims,
                ))

        results.sort(key=lambda r: r.global_similarity, reverse=True)
        return results[:top_k]

    def recall_by_layer(
        self,
        query: str,
        layer_name: str,
        top_k: int = 5,
        speaker: str = "incoming",
    ) -> list[RecallResult]:
        """Search by a specific layer only.

        Useful for targeted queries: "search perspective layer for
        thoughts about self".
        """
        if not self._thoughts:
            return []

        query_glyph = self._encoder.encode_thought(query, speaker=speaker)
        if layer_name not in query_glyph.layers:
            return []

        query_layer_cortex = query_glyph.layers[layer_name].cortex.data
        results: list[RecallResult] = []

        for stored in self._thoughts.values():
            if layer_name not in stored.glyph.layers:
                continue
            sim = float(cosine_similarity(
                query_layer_cortex,
                stored.glyph.layers[layer_name].cortex.data,
            ))
            results.append(RecallResult(
                thought=stored,
                global_similarity=sim,
                layer_similarities={layer_name: sim},
            ))

        results.sort(key=lambda r: r.global_similarity, reverse=True)
        return results[:top_k]

    # ── Primitive version tracking ─────────────────────────────────────────

    def _refresh_if_needed(self) -> None:
        """Re-encode all stored thoughts if primitives have changed.

        Crystallization adds new compound primitives, which changes how
        words are classified (content vs. structural). Without re-encoding,
        stored thoughts have stale content vectors that don't match queries
        encoded with the new primitives.

        Cheap for in-memory storage (<50ms for ~100 thoughts at 2000D).
        """
        current_version = self._primitives.version
        if current_version == self._primitives_version:
            return

        logger.info(
            "Primitives changed (v%d → v%d), re-encoding %d thoughts",
            self._primitives_version, current_version, len(self._thoughts),
        )

        for stored in self._thoughts.values():
            new_glyph = self._encoder.encode_thought(
                stored.content, speaker=stored.speaker, metadata=stored.metadata,
            )
            stored.glyph = new_glyph

        self._primitives_version = current_version

    # ── Working set for DreamLoop ─────────────────────────────────────────

    def load_working_set(
        self,
        strategy: str = "recent",
        size: int = 50,
    ) -> list[StoredThought]:
        """Load a subset of thoughts for the DreamLoop.

        Strategies:
          - "recent": Most recently absorbed
          - "weak": Lowest strength (need reinforcement or pruning)
          - "random": Random sample for diversity

        Args:
            strategy: Selection strategy.
            size: Number of thoughts to load.

        Returns:
            List of StoredThought.
        """
        thoughts = list(self._thoughts.values())
        if not thoughts:
            return []

        if strategy == "recent":
            thoughts.sort(key=lambda t: t.created_at, reverse=True)
        elif strategy == "weak":
            thoughts.sort(key=lambda t: t.strength)
        elif strategy == "random":
            np.random.shuffle(thoughts)

        return thoughts[:size]

    # ── Maintenance ───────────────────────────────────────────────────────

    def reinforce(self, thought_id: str, amount: float = 0.1) -> None:
        """Strengthen a thought (Hebbian reinforcement)."""
        if thought_id in self._thoughts:
            self._thoughts[thought_id].reinforce(amount)

    def decay_all(self, factor: float = 0.95) -> int:
        """Decay all thoughts. Returns count of pruned (strength ≤ 0)."""
        pruned = 0
        to_remove = []
        for tid, thought in self._thoughts.items():
            thought.decay(factor)
            if thought.strength <= 0.01:
                to_remove.append(tid)
                pruned += 1
        for tid in to_remove:
            text_key = self._thoughts[tid].content.strip().lower()
            self._absorbed_texts.discard(text_key)
            del self._thoughts[tid]
        return pruned

    def clear(self) -> None:
        """Clear all thoughts (hard reset)."""
        self._thoughts.clear()
        self._absorbed_texts.clear()

    # ── Introspection ─────────────────────────────────────────────────────

    def get_thought(self, thought_id: str) -> StoredThought | None:
        return self._thoughts.get(thought_id)

    def all_thoughts(self) -> list[StoredThought]:
        """All thoughts sorted by strength (strongest first)."""
        return sorted(self._thoughts.values(), key=lambda t: t.strength, reverse=True)

    def stats(self) -> dict:
        return {
            "count": len(self._thoughts),
            "primitives": self._primitives.stats(),
            "avg_strength": (
                sum(t.strength for t in self._thoughts.values()) / len(self._thoughts)
                if self._thoughts else 0.0
            ),
        }

    def format_recall(self, results: list[RecallResult], max_results: int = 5) -> str:
        """Format recall results for injection into the LLM prompt.

        Translates speaker tags into natural language so the LLM
        understands perspective: who said what.
        """
        if not results:
            return ""

        lines = []
        for r in results[:max_results]:
            t = r.thought
            who = "the user said" if t.speaker == "incoming" else "Ada said"
            lines.append(
                f"[{r.global_similarity:.2f}] {who}: \"{t.content}\""
            )

        return "Ada remembers:\n" + "\n".join(lines)
