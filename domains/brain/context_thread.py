"""
ContextThread — Ada's structured memory, encoded as glyphs.

A thread IS a glyph — same 5-layer structure as a thought
(perspective, semantic, relational, temporal, direction). The LLM
extracts entities, facts, topic, emotion. Those fill the layers.
Recall uses the same three-signal cosine similarity.

The thread's tool, topic, entities, and facts are encoded into the
glyph layers. Recall compares at whatever granularity you want —
global cortex, specific layer, segment, or role.

Temporal signal differentiates 10 discussions of "family" on the
same day. Each is a separate thread glyph with its own timestamp.
The dream loop consolidates across time.
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
from glyphh.memory.thought_glyph import ThoughtGlyphEncoder

logger = logging.getLogger(__name__)


@dataclass
class ContextThread:
    """A memory thread — a glyph with conversation context.

    The glyph encodes the thread's content through the same 5-layer
    pipeline as individual thoughts. Tool, topic, entities, and facts
    all contribute to the encoding. Similarity at any layer level.
    """

    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Source — which tool and session created this
    tool: str = "unknown"           # claude-code, claude-desktop, gemini, etc.
    session_id: str = ""            # ties to a single conversation session

    # Content
    topic: str = ""                 # inferred topic: "family", "code-refactor", etc.
    entities: list[str] = field(default_factory=list)   # Brandi, James, Traceton
    facts: list[str] = field(default_factory=list)      # the actual stored statements
    summary: str = ""               # LLM-generated thread summary (built by dream loop)

    # Temporal — differentiates 10 discussions of "family" on the same day
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0             # how many interactions contributed

    # Relationships — set by dream loop
    related_threads: list[str] = field(default_factory=list)  # thread_ids

    # State
    active: bool = True             # False = closed thread (topic shifted)

    # HDC encoding — the thread IS a glyph
    glyph: Glyph | None = field(default=None, repr=False)

    def add_fact(self, fact: str, entities: list[str] | None = None) -> None:
        """Add a fact to this thread. Marks glyph stale."""
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self.turn_count += 1
            self.updated_at = time.time()
            self.glyph = None  # stale — needs re-encoding
        if entities:
            for e in entities:
                if e not in self.entities:
                    self.entities.append(e)
                    self.glyph = None  # stale

    def close(self) -> None:
        """Close this thread — topic has shifted."""
        self.active = False
        self.updated_at = time.time()

    def encode(self, encoder: ThoughtGlyphEncoder) -> None:
        """Encode (or re-encode) this thread as a glyph.

        Combines tool, topic, entities, and facts into text and
        runs it through the same 5-layer thought encoding pipeline.
        The glyph captures the structure at every level — global
        cortex for broad matching, layers for structural matching,
        segments/roles for precise matching.
        """
        text = self._build_encoding_text()
        self.glyph = encoder.encode_thought(text, speaker="incoming")

    def _build_encoding_text(self) -> str:
        """Build text for glyph encoding from thread content.

        Combines all thread dimensions into natural text so the
        thought encoder maps them to the right layers/segments.
        Tool and topic become content words. Entities are content.
        Facts carry the full structural signal.
        """
        parts = []
        # Tool as content signal (will become a content word in the encoding)
        if self.tool and self.tool != "unknown":
            parts.append(self.tool)
        # Topic
        if self.topic:
            parts.append(self.topic)
        # Entities
        for entity in self.entities:
            parts.append(entity)
        # Facts — recent, capped to avoid dilution
        for fact in self.facts[-10:]:
            if not fact.startswith("[Q] "):
                parts.append(fact)
        return ". ".join(parts) if parts else "empty"

    def ensure_encoded(self, encoder: ThoughtGlyphEncoder) -> None:
        """Encode only if glyph is stale or missing."""
        if self.glyph is None:
            self.encode(encoder)

    def matches_tool(self, tool: str) -> bool:
        """Check if this thread came from a specific tool."""
        return self.tool.lower() == tool.lower()

    def matches_topic(self, topic: str) -> bool:
        """Check if this thread is about a specific topic."""
        return self.topic.lower() == topic.lower()


class ThreadStore:
    """Thread store with glyph-based recall.

    No inverted indexes. Recall uses the same three-signal cosine
    similarity as ThoughtGlyphSpace: content similarity, layer
    similarity, and structural overlap. Temporal signal (recency)
    acts as a multiplier.

    Tool is a metadata filter (you can restrict to one tool),
    not a similarity dimension.
    """

    def __init__(self, encoder: ThoughtGlyphEncoder | None = None):
        self._threads: dict[str, ContextThread] = {}
        self._encoder = encoder

    @property
    def count(self) -> int:
        return len(self._threads)

    @property
    def encoder(self) -> ThoughtGlyphEncoder | None:
        return self._encoder

    def set_encoder(self, encoder: ThoughtGlyphEncoder) -> None:
        """Set or update the encoder."""
        self._encoder = encoder

    def add(self, thread: ContextThread) -> None:
        """Add a thread to the store. Encodes if encoder available."""
        self._threads[thread.thread_id] = thread
        if self._encoder:
            thread.ensure_encoded(self._encoder)

    def get(self, thread_id: str) -> ContextThread | None:
        return self._threads.get(thread_id)

    def active_thread(self, tool: str, topic: str = "") -> ContextThread | None:
        """Find the currently active thread for a tool+topic."""
        candidates = []
        for t in self._threads.values():
            if not t.active or not t.matches_tool(tool):
                continue
            if topic and not t.matches_topic(topic):
                continue
            candidates.append(t)

        if not candidates:
            return None
        return max(candidates, key=lambda t: t.updated_at)

    def recall(
        self,
        query: str,
        tool: str = "",
        limit: int = 10,
        min_similarity: float = 0.01,
    ) -> list[tuple[ContextThread, float]]:
        """Glyph similarity recall — same three-signal search.

        Returns (thread, score) tuples sorted by combined score.
        Score = glyph_similarity * recency_weight.

        Tool is a metadata filter, not a similarity dimension.
        Temporal signal: recent threads get a recency boost.
        """
        if not self._threads or not self._encoder:
            return []

        query_glyph = self._encoder.encode_thought(query, speaker="incoming")
        query_content_vec = query_glyph.metadata.get("_content_vector")
        query_activated = set(query_glyph.metadata.get("_activated_attrs", []))

        _LAYER_WEIGHTS = {
            "perspective": 0.25,
            "semantic": 0.30,
            "relational": 0.25,
            "temporal": 0.10,
            "direction": 0.10,
        }

        scored: list[tuple[ContextThread, float]] = []

        for thread in self._threads.values():
            # Tool filter
            if tool and not thread.matches_tool(tool):
                continue

            # Ensure encoded
            if thread.glyph is None:
                thread.ensure_encoded(self._encoder)
            if thread.glyph is None:
                continue

            # ── Signal 1: Content similarity ──────────────────
            content_sim = 0.0
            stored_content_vec = thread.glyph.metadata.get("_content_vector")
            if query_content_vec is not None and stored_content_vec is not None:
                content_sim = max(0.0, float(cosine_similarity(
                    query_content_vec, stored_content_vec,
                )))

            # ── Signal 2: Layer similarity ────────────────────
            stored_activated = set(
                thread.glyph.metadata.get("_activated_attrs", [])
            )

            total_weighted_sim = 0.0
            total_weight = 0.0

            for layer_name, query_layer in query_glyph.layers.items():
                if layer_name not in thread.glyph.layers:
                    continue
                stored_layer = thread.glyph.layers[layer_name]
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
                        rsim = max(0.0, float(cosine_similarity(
                            query_role_vec.data,
                            stored_seg.roles[role_name].data,
                        )))
                        layer_sim_sum += rsim

                if layer_seg_count > 0:
                    layer_sim = layer_sim_sum / layer_seg_count
                    total_weighted_sim += layer_sim * layer_weight
                    total_weight += layer_weight

            role_sim = total_weighted_sim / total_weight if total_weight > 0 else 0.0

            # ── Signal 3: Structural overlap ──────────────────
            seg_overlap = len(query_activated & stored_activated)
            seg_total = len(query_activated | stored_activated) or 1
            structural = seg_overlap / seg_total

            # ── Combine: adaptive weights ─────────────────────
            n_content = len(query_glyph.metadata.get("_content_words", []))
            if n_content >= 2:
                w_content, w_role, w_struct = 0.55, 0.30, 0.15
            elif n_content == 1:
                w_content, w_role, w_struct = 0.40, 0.40, 0.20
            else:
                w_content, w_role, w_struct = 0.10, 0.65, 0.25

            glyph_sim = (
                content_sim * w_content
                + role_sim * w_role
                + structural * w_struct
            )

            # ── Temporal signal: recency weight ───────────────
            # Recent threads get a boost. Old ones still match
            # if the glyph similarity is strong.
            age_hours = (time.time() - thread.updated_at) / 3600
            recency = 1.0 / (1.0 + age_hours / 24.0)
            final_score = glyph_sim * (0.8 + 0.2 * recency)

            if final_score >= min_similarity:
                scored.append((thread, final_score))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def recall_by_layer(
        self,
        query: str,
        layer_name: str,
        tool: str = "",
        limit: int = 10,
    ) -> list[tuple[ContextThread, float]]:
        """Search threads by a specific glyph layer only.

        Useful for targeted queries: "search semantic layer for
        threads about family."
        """
        if not self._threads or not self._encoder:
            return []

        query_glyph = self._encoder.encode_thought(query, speaker="incoming")
        if layer_name not in query_glyph.layers:
            return []

        query_layer_cortex = query_glyph.layers[layer_name].cortex.data
        scored: list[tuple[ContextThread, float]] = []

        for thread in self._threads.values():
            if tool and not thread.matches_tool(tool):
                continue
            thread.ensure_encoded(self._encoder)
            if thread.glyph is None or layer_name not in thread.glyph.layers:
                continue

            sim = float(cosine_similarity(
                query_layer_cortex,
                thread.glyph.layers[layer_name].cortex.data,
            ))
            if sim > 0:
                scored.append((thread, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def find_by_tool(self, tool: str) -> list[ContextThread]:
        """Find all threads from a specific tool (metadata filter)."""
        results = [
            t for t in self._threads.values()
            if t.matches_tool(tool)
        ]
        return sorted(results, key=lambda t: t.updated_at, reverse=True)

    def all_threads(self) -> list[ContextThread]:
        """All threads, most recent first."""
        return sorted(
            self._threads.values(),
            key=lambda t: t.updated_at,
            reverse=True,
        )

    def reindex(self, thread: ContextThread) -> None:
        """Re-encode a thread after content changes."""
        if self._encoder:
            thread.encode(self._encoder)
