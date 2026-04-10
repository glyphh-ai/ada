"""
ThreadManager — manages context threads during conversation.

Decides when to create a new thread vs extend an existing one.
Infers topics from entities and facts. Closes threads when the
topic shifts. Encodes threads as glyphs after each update.

The manager sits between the extractor (which pulls structure from
language) and the thread store (which does glyph-based recall).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from domains.brain.context_thread import ContextThread, ThreadStore

logger = logging.getLogger(__name__)

# If the active thread hasn't been updated in this many seconds,
# a new topic probably started.
THREAD_TIMEOUT_S = 300  # 5 minutes

# Topic keywords → topic labels. Simple rule-based for now.
# The dream loop can refine these later.
_TOPIC_SIGNALS = {
    "family": ["wife", "husband", "son", "daughter", "child", "children",
               "kids", "brother", "sister", "mother", "father", "mom",
               "dad", "parent", "married", "spouse", "family"],
    "work": ["job", "work", "company", "boss", "project", "team",
             "meeting", "deadline", "client", "office", "career"],
    "location": ["lives", "live", "moved", "address", "city", "state",
                 "country", "home", "house", "apartment", "neighborhood"],
    "preferences": ["like", "love", "hate", "prefer", "favorite",
                    "enjoy", "dislike"],
    "health": ["feel", "sick", "doctor", "hospital", "health",
               "exercise", "sleep", "diet", "pain", "medication"],
    "identity": ["name", "age", "birthday", "born", "email", "phone"],
}


def infer_topic(text: str, entities: list[str], facts: list[str]) -> str:
    """Infer a topic label from the content."""
    combined = " ".join([text] + entities + facts).lower()

    best_topic = ""
    best_count = 0

    for topic, signals in _TOPIC_SIGNALS.items():
        count = sum(1 for s in signals if s in combined)
        if count > best_count:
            best_count = count
            best_topic = topic

    return best_topic or "general"


class ThreadManager:
    """Manages context threads for a session.

    Usage:
        manager = ThreadManager(store, tool="claude-code")

        thread = manager.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )

        results = manager.recall_for_query("who is my wife?")
        # → [(thread, 0.72), ...]
    """

    def __init__(
        self,
        store: ThreadStore,
        tool: str = "unknown",
        session_id: str = "",
    ):
        self._store = store
        self._tool = tool
        self._session_id = session_id
        self._active_thread: Optional[ContextThread] = None

    @property
    def store(self) -> ThreadStore:
        return self._store

    @property
    def active_thread(self) -> Optional[ContextThread]:
        return self._active_thread

    def process(
        self,
        input_text: str,
        facts: list[str],
        entities: list[str],
        question: str | None = None,
        emotion: str | None = None,
    ) -> ContextThread:
        """Process an interaction — add to existing thread or create new.

        After adding facts, re-encodes the thread glyph so the
        encoding reflects the updated content.
        """
        topic = infer_topic(input_text, entities, facts)
        thread = self._resolve_thread(topic, entities)

        for fact in facts:
            thread.add_fact(fact, entities)

        if question:
            thread.add_fact(f"[Q] {question}")

        # Re-encode the glyph (add_fact marks it stale)
        self._store.reindex(thread)

        self._active_thread = thread
        return thread

    def recall_for_query(
        self,
        query_text: str,
        query_entities: list[str] | None = None,
        query_topic: str = "",
        tool_filter: str = "",
    ) -> list[ContextThread]:
        """Find threads relevant to a query using glyph similarity.

        The query is encoded as a glyph and compared against all
        thread glyphs using three-signal cosine similarity.
        """
        # Build the query text with any implicit entities
        implicit = _extract_implicit_entities(query_text)
        all_entities = list(set((query_entities or []) + implicit))

        # Enrich the query with entities for better encoding
        enriched = query_text
        if all_entities:
            enriched = query_text + " " + " ".join(all_entities)

        results = self._store.recall(
            query=enriched,
            tool=tool_filter,
            limit=5,
        )

        return [thread for thread, score in results]

    def collect_facts(
        self, threads: list[ContextThread],
    ) -> list[tuple[str, str, float]]:
        """Collect facts from threads in pipeline format.

        Returns (content, speaker, confidence) tuples.
        Confidence comes from thread recency.
        """
        facts = []
        seen = set()
        for thread in threads:
            age_hours = (time.time() - thread.updated_at) / 3600
            base_confidence = 0.9 if age_hours < 1 else 0.7 if age_hours < 24 else 0.5

            for fact in thread.facts:
                if fact in seen or fact.startswith("[Q] "):
                    continue
                seen.add(fact)
                facts.append((fact, "incoming", base_confidence))

        return facts

    def _resolve_thread(
        self, topic: str, entities: list[str],
    ) -> ContextThread:
        """Decide whether to extend the active thread or create a new one.

        Rules:
        1. Active + overlapping entities → extend
        2. Active + same topic + recent → extend
        3. Stale (>5min) → close, start new
        4. Different context → start new
        """
        if self._active_thread and self._active_thread.active:
            thread = self._active_thread

            age = time.time() - thread.updated_at
            if age > THREAD_TIMEOUT_S:
                thread.close()
                return self._create_thread(topic, entities)

            if entities:
                for e in entities:
                    if any(
                        existing.lower() == e.lower()
                        for existing in thread.entities
                    ):
                        if topic != "general" and thread.topic == "general":
                            thread.topic = topic
                        return thread

            if topic != "general" and thread.matches_topic(topic):
                return thread

            thread.close()

        return self._create_thread(topic, entities)

    def _create_thread(self, topic: str, entities: list[str]) -> ContextThread:
        """Create a new thread and add it to the store."""
        thread = ContextThread(
            tool=self._tool,
            session_id=self._session_id,
            topic=topic,
            entities=list(entities),
        )
        self._store.add(thread)
        logger.info(
            f"New thread: {thread.thread_id[:8]} "
            f"tool={self._tool} topic={topic} entities={entities}"
        )
        return thread


def _extract_implicit_entities(text: str) -> list[str]:
    """Extract implicit entity references from a query.

    "who is my wife?" → ["wife"]
    "tell me about Brandi" → ["Brandi"]
    """
    text_lower = text.lower()
    implicit = []

    _RELATIONSHIP_WORDS = [
        "wife", "husband", "spouse", "partner",
        "son", "daughter", "child", "children", "kids",
        "brother", "sister", "mother", "father", "mom", "dad",
        "friend", "boss", "coworker",
        "name", "email", "age", "birthday",
    ]

    for word in _RELATIONSHIP_WORDS:
        if word in text_lower:
            implicit.append(word)

    for word in text.split():
        cleaned = word.strip("?.,!\"'")
        if cleaned and cleaned[0].isupper() and cleaned.lower() not in (
            "i", "a", "the", "is", "are", "was", "were", "what",
            "who", "where", "when", "how", "why", "do", "does",
            "can", "could", "would", "should", "tell", "about",
            "my", "me", "you", "ada", "hello", "hi",
        ):
            implicit.append(cleaned)

    return implicit
