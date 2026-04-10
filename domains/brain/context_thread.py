"""
ContextThread — Ada's structured memory.

A thread is a conversation context: a topic being discussed, the entities
involved, the facts learned, which tool initiated it, and when.

Threads replace flat string matching for recall. Instead of cosine
similarity between "who is my wife?" and "my wifes name is brandi",
recall searches by entity ("wife") and topic ("family") — structured
lookup, not fuzzy vector math.

HDC still encodes threads for the dream loop, which finds connections
BETWEEN threads (cross-topic, cross-tool). That's the right abstraction
level for HDC — pattern matching across contexts, not across words.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ContextThread:
    """A memory thread — a conversation context with structure."""

    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Source — which tool and session created this
    tool: str = "unknown"           # claude-code, claude-desktop, gemini, etc.
    session_id: str = ""            # ties to a single conversation session

    # Content
    topic: str = ""                 # inferred topic: "family", "code-refactor", etc.
    entities: list[str] = field(default_factory=list)   # Brandi, James, Traceton
    facts: list[str] = field(default_factory=list)      # the actual stored statements
    summary: str = ""               # LLM-generated thread summary (built by dream loop)

    # Temporal
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0             # how many interactions contributed

    # Relationships — set by dream loop
    related_threads: list[str] = field(default_factory=list)  # thread_ids

    # State
    active: bool = True             # False = closed thread (topic shifted)

    def add_fact(self, fact: str, entities: list[str] | None = None) -> None:
        """Add a fact to this thread."""
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self.turn_count += 1
            self.updated_at = time.time()
        if entities:
            for e in entities:
                if e not in self.entities:
                    self.entities.append(e)

    def close(self) -> None:
        """Close this thread — topic has shifted."""
        self.active = False
        self.updated_at = time.time()

    def matches_entity(self, entity: str) -> bool:
        """Check if this thread involves a specific entity."""
        entity_lower = entity.lower()
        return any(e.lower() == entity_lower for e in self.entities)

    def matches_topic(self, topic: str) -> bool:
        """Check if this thread is about a specific topic."""
        return self.topic.lower() == topic.lower()

    def matches_tool(self, tool: str) -> bool:
        """Check if this thread came from a specific tool."""
        return self.tool.lower() == tool.lower()

    def relevance_to(self, query_entities: list[str], query_topic: str = "") -> float:
        """Score how relevant this thread is to a query.

        Entity matches are strong signals. Topic match is a boost.
        More recent threads get a slight edge.
        """
        score = 0.0

        # Entity overlap — primary signal
        if query_entities:
            matched = sum(
                1 for qe in query_entities
                if self.matches_entity(qe)
            )
            score += matched / len(query_entities)

        # Topic match — secondary signal
        if query_topic and self.matches_topic(query_topic):
            score += 0.3

        # Recency boost — threads updated recently get a small edge
        age_hours = (time.time() - self.updated_at) / 3600
        if age_hours < 1:
            score += 0.1
        elif age_hours < 24:
            score += 0.05

        return min(1.0, score)


class ThreadStore:
    """In-memory thread index with entity and topic lookups.

    Backed by SQL for persistence (loaded at boot, flushed on write).
    The in-memory index gives instant structured lookup — no vector
    search needed for recall.
    """

    def __init__(self):
        self._threads: dict[str, ContextThread] = {}
        # Inverted indexes for fast lookup
        self._by_entity: dict[str, list[str]] = {}  # entity -> [thread_ids]
        self._by_topic: dict[str, list[str]] = {}   # topic -> [thread_ids]
        self._by_tool: dict[str, list[str]] = {}    # tool -> [thread_ids]

    @property
    def count(self) -> int:
        return len(self._threads)

    def add(self, thread: ContextThread) -> None:
        """Add a thread to the store and update indexes."""
        self._threads[thread.thread_id] = thread
        self._index(thread)

    def get(self, thread_id: str) -> ContextThread | None:
        return self._threads.get(thread_id)

    def active_thread(self, tool: str, topic: str = "") -> ContextThread | None:
        """Find the currently active thread for a tool+topic.

        If topic is given, looks for an active thread with that topic.
        Otherwise returns the most recently updated active thread for the tool.
        """
        tool_threads = self._by_tool.get(tool.lower(), [])
        candidates = []
        for tid in tool_threads:
            t = self._threads.get(tid)
            if t and t.active:
                if topic and t.matches_topic(topic):
                    candidates.append(t)
                elif not topic:
                    candidates.append(t)

        if not candidates:
            return None
        # Most recently updated
        return max(candidates, key=lambda t: t.updated_at)

    def find_by_entity(self, entity: str, tool: str = "") -> list[ContextThread]:
        """Find all threads involving an entity, optionally filtered by tool."""
        thread_ids = self._by_entity.get(entity.lower(), [])
        results = []
        for tid in thread_ids:
            t = self._threads.get(tid)
            if t:
                if tool and not t.matches_tool(tool):
                    continue
                results.append(t)
        return sorted(results, key=lambda t: t.updated_at, reverse=True)

    def find_by_topic(self, topic: str, tool: str = "") -> list[ContextThread]:
        """Find all threads about a topic, optionally filtered by tool."""
        thread_ids = self._by_topic.get(topic.lower(), [])
        results = []
        for tid in thread_ids:
            t = self._threads.get(tid)
            if t:
                if tool and not t.matches_tool(tool):
                    continue
                results.append(t)
        return sorted(results, key=lambda t: t.updated_at, reverse=True)

    def find_by_tool(self, tool: str) -> list[ContextThread]:
        """Find all threads from a specific tool."""
        thread_ids = self._by_tool.get(tool.lower(), [])
        results = []
        for tid in thread_ids:
            t = self._threads.get(tid)
            if t:
                results.append(t)
        return sorted(results, key=lambda t: t.updated_at, reverse=True)

    def recall(
        self,
        entities: list[str] | None = None,
        topic: str = "",
        tool: str = "",
        limit: int = 10,
    ) -> list[ContextThread]:
        """Structured recall — find threads by entity, topic, and/or tool.

        This is the primary recall path. No cosine similarity,
        no vector math. Structured lookup.
        """
        if not entities and not topic and not tool:
            # Return most recent threads
            all_threads = sorted(
                self._threads.values(),
                key=lambda t: t.updated_at,
                reverse=True,
            )
            return all_threads[:limit]

        # Score every thread by relevance
        scored = []
        for thread in self._threads.values():
            score = 0.0

            # Entity matching
            if entities:
                for e in entities:
                    if thread.matches_entity(e):
                        score += 1.0

            # Topic matching
            if topic and thread.matches_topic(topic):
                score += 0.5

            # Tool filtering (not scoring — it's a filter)
            if tool and not thread.matches_tool(tool):
                continue

            if score > 0:
                scored.append((thread, score))

        scored.sort(key=lambda x: (-x[1], -x[0].updated_at))
        return [t for t, _ in scored[:limit]]

    def all_threads(self) -> list[ContextThread]:
        """All threads, most recent first."""
        return sorted(
            self._threads.values(),
            key=lambda t: t.updated_at,
            reverse=True,
        )

    def _index(self, thread: ContextThread) -> None:
        """Update inverted indexes for a thread."""
        # Entity index
        for entity in thread.entities:
            key = entity.lower()
            if key not in self._by_entity:
                self._by_entity[key] = []
            if thread.thread_id not in self._by_entity[key]:
                self._by_entity[key].append(thread.thread_id)

        # Topic index
        if thread.topic:
            key = thread.topic.lower()
            if key not in self._by_topic:
                self._by_topic[key] = []
            if thread.thread_id not in self._by_topic[key]:
                self._by_topic[key].append(thread.thread_id)

        # Tool index
        if thread.tool:
            key = thread.tool.lower()
            if key not in self._by_tool:
                self._by_tool[key] = []
            if thread.thread_id not in self._by_tool[key]:
                self._by_tool[key].append(thread.thread_id)

    def reindex(self, thread: ContextThread) -> None:
        """Re-index a thread after entities/topic change."""
        self._index(thread)
