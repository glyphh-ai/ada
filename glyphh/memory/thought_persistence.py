"""
Thought persistence — saves and loads Ada's memories from SQLite.

ThoughtGlyphSpace stays in-memory for fast recall. This module
syncs to/from the database:
  - Boot: load all non-archived thoughts into memory
  - Absorb: write new thought to DB
  - Reinforce/decay: update strength + last_accessed async
  - Archive: set archived=1, keep in DB for potential restoration
  - Shutdown: flush pending updates
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from glyphh.memory.thought_space import ThoughtGlyphSpace, StoredThought

logger = logging.getLogger(__name__)


def _serialize_glyph(glyph: Any) -> str:
    """Serialize a Glyph to JSON for storage."""
    data = {}

    # Serialize layers → segments → roles
    if hasattr(glyph, 'layers'):
        layers = {}
        for layer_name, layer in glyph.layers.items():
            segs = {}
            if hasattr(layer, 'segments'):
                for seg_name, seg in layer.segments.items():
                    roles = {}
                    if hasattr(seg, 'roles'):
                        for role_name, role_vec in seg.roles.items():
                            vec = role_vec.data if hasattr(role_vec, 'data') else role_vec
                            if isinstance(vec, np.ndarray):
                                roles[role_name] = vec.tolist()
                            elif isinstance(vec, list):
                                roles[role_name] = vec
                    segs[seg_name] = roles
            layers[layer_name] = segs
        data['layers'] = layers

    # Serialize cortex
    if hasattr(glyph, 'cortex'):
        cortex = glyph.cortex
        if isinstance(cortex, np.ndarray):
            data['cortex'] = cortex.tolist()
        elif isinstance(cortex, list):
            data['cortex'] = cortex

    return json.dumps(data)


def _serialize_content_vector(glyph: Any) -> str | None:
    """Serialize the content vector for fast recall."""
    vec = glyph.metadata.get("_content_vector") if hasattr(glyph, 'metadata') else None
    if vec is None:
        return None
    if isinstance(vec, np.ndarray):
        return json.dumps(vec.tolist())
    if isinstance(vec, list):
        return json.dumps(vec)
    return None


async def load_thoughts(
    session_factory: Any,
    space: ThoughtGlyphSpace,
) -> int:
    """Load all non-archived thoughts from SQLite into the thought space.

    Called once at boot. Overwrites any seed memories with persisted state
    (strength, access count, timestamps). Returns count of loaded thoughts.
    """
    from domains.models.db_models import AdaThought

    count = 0
    async with session_factory() as session:
        result = await session.execute(
            select(AdaThought).where(AdaThought.archived == 0)
        )
        rows = result.scalars().all()

        for row in rows:
            text_key = row.content.strip().lower()

            # Check if this thought already exists (e.g. from seeds)
            existing = None
            for t in space._thoughts.values():
                if t.content.strip().lower() == text_key:
                    existing = t
                    break

            if existing:
                # Overwrite seed with persisted state
                existing.thought_id = row.thought_id
                existing.strength = row.strength
                existing.created_at = row.created_at
                existing.last_accessed = row.last_accessed
                existing.access_count = row.access_count
                existing.metadata = row.extra_data or {}
                count += 1
            else:
                # New thought from DB — absorb and restore state
                stored = space.absorb(row.content, speaker=row.speaker)
                if stored:
                    stored.thought_id = row.thought_id
                    stored.strength = row.strength
                    stored.created_at = row.created_at
                    stored.last_accessed = row.last_accessed
                    stored.access_count = row.access_count
                    stored.metadata = row.extra_data or {}
                    count += 1

    if count:
        logger.info(f"Loaded {count} thoughts from database")
    return count


async def save_thought(
    session_factory: Any,
    thought: StoredThought,
) -> None:
    """Save a new thought to SQLite."""
    from domains.models.db_models import AdaThought

    async with session_factory() as session:
        row = AdaThought(
            thought_id=thought.thought_id,
            content=thought.content,
            speaker=thought.speaker,
            glyph_data=_serialize_glyph(thought.glyph),
            content_vector=_serialize_content_vector(thought.glyph),
            strength=thought.strength,
            access_count=thought.access_count,
            created_at=thought.created_at,
            last_accessed=thought.last_accessed,
            extra_data=thought.metadata,
            archived=0,
        )
        session.add(row)
        await session.commit()


async def update_thought_strength(
    session_factory: Any,
    thought_id: str,
    strength: float,
    last_accessed: float,
    access_count: int,
) -> None:
    """Update strength/access for a thought in SQLite."""
    from domains.models.db_models import AdaThought

    async with session_factory() as session:
        await session.execute(
            update(AdaThought)
            .where(AdaThought.thought_id == thought_id)
            .values(
                strength=strength,
                last_accessed=last_accessed,
                access_count=access_count,
            )
        )
        await session.commit()


async def archive_thought(
    session_factory: Any,
    thought_id: str,
) -> None:
    """Mark a thought as archived in SQLite."""
    from domains.models.db_models import AdaThought

    async with session_factory() as session:
        await session.execute(
            update(AdaThought)
            .where(AdaThought.thought_id == thought_id)
            .values(archived=1)
        )
        await session.commit()


async def clear_all_thoughts(session_factory: Any) -> int:
    """Delete all thoughts from SQLite. Called by 'memory reset'."""
    from domains.models.db_models import AdaThought
    from sqlalchemy import delete as sa_delete

    async with session_factory() as session:
        result = await session.execute(sa_delete(AdaThought))
        await session.commit()
        count = result.rowcount
        logger.info(f"Cleared {count} thoughts from database")
        return count


async def flush_all_strengths(
    session_factory: Any,
    space: ThoughtGlyphSpace,
) -> int:
    """Batch-update all thought strengths to SQLite. Called on shutdown."""
    from domains.models.db_models import AdaThought

    count = 0
    async with session_factory() as session:
        for thought in space._thoughts.values():
            await session.execute(
                update(AdaThought)
                .where(AdaThought.thought_id == thought.thought_id)
                .values(
                    strength=thought.strength,
                    last_accessed=thought.last_accessed,
                    access_count=thought.access_count,
                )
            )
            count += 1
        await session.commit()

    if count:
        logger.info(f"Flushed {count} thought strengths to database")
    return count
