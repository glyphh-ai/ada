"""
Thread persistence — saves and loads context threads from SQLite.

ThreadStore stays in-memory for fast structured lookup. This module
syncs to/from the database:
  - Boot: load all active threads into memory
  - Write: save new/updated threads after each interaction
  - Shutdown: flush all thread state
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.brain.context_thread import ContextThread, ThreadStore

logger = logging.getLogger(__name__)


async def load_threads(session_factory: Any, store: ThreadStore) -> int:
    """Load all threads from SQLite into the thread store.

    Called once at boot. Returns count of loaded threads.
    """
    from domains.models.db_models import AdaThread

    count = 0
    async with session_factory() as session:
        result = await session.execute(select(AdaThread))
        rows = result.scalars().all()

        for row in rows:
            thread = ContextThread(
                thread_id=row.thread_id,
                tool=row.tool,
                session_id=row.session_id,
                topic=row.topic,
                entities=row.entities or [],
                facts=row.facts or [],
                summary=row.summary or "",
                turn_count=row.turn_count,
                created_at=row.created_at,
                updated_at=row.updated_at,
                related_threads=row.related_threads or [],
                active=bool(row.active),
            )
            store.add(thread)
            count += 1

    if count:
        logger.info(f"Loaded {count} threads from database")
    return count


async def save_thread(session_factory: Any, thread: ContextThread) -> None:
    """Save or update a thread in SQLite."""
    from domains.models.db_models import AdaThread

    async with session_factory() as session:
        # Check if it exists
        existing = await session.get(AdaThread, thread.thread_id)

        if existing:
            existing.topic = thread.topic
            existing.entities = thread.entities
            existing.facts = thread.facts
            existing.summary = thread.summary
            existing.turn_count = thread.turn_count
            existing.updated_at = thread.updated_at
            existing.related_threads = thread.related_threads
            existing.active = 1 if thread.active else 0
        else:
            row = AdaThread(
                thread_id=thread.thread_id,
                tool=thread.tool,
                session_id=thread.session_id,
                topic=thread.topic,
                entities=thread.entities,
                facts=thread.facts,
                summary=thread.summary,
                turn_count=thread.turn_count,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                related_threads=thread.related_threads,
                active=1 if thread.active else 0,
            )
            session.add(row)

        await session.commit()


async def clear_all_threads(session_factory: Any) -> int:
    """Delete all threads from SQLite. Called by 'memory reset'."""
    from domains.models.db_models import AdaThread
    from sqlalchemy import delete as sa_delete

    async with session_factory() as session:
        result = await session.execute(sa_delete(AdaThread))
        await session.commit()
        count = result.rowcount
        logger.info(f"Cleared {count} threads from database")
        return count
