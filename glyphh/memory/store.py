"""
ThoughtStore — persistent file-based storage for Ada's thoughts.

Storage format:
  ~/.glyphh/memory/thoughts.jsonl   — metadata (one JSON object per line)
  ~/.glyphh/memory/vectors.npy      — stacked numpy array of all thought vectors

On save, both files are rewritten atomically.  On load, they're read once
and cached in memory.  Cosine similarity search is a single matrix multiply
against the stacked vectors — fast even with thousands of thoughts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from glyphh.core.ops import cosine_similarity
from .thought import Thought, ThoughtEncoder

logger = logging.getLogger(__name__)

# ── Default storage path ───────────────────────────────────────────────────

_DEFAULT_DIR = os.path.expanduser("~/.glyphh/memory")


# ── ThoughtStore ───────────────────────────────────────────────────────────

class ThoughtStore:
    """Persistent thought storage with semantic recall.

    Usage:
        encoder = ThoughtEncoder()
        store = ThoughtStore(encoder)
        store.load()

        # Store a thought
        thought = store.remember("HDC uses bipolar vectors for encoding")

        # Recall by language
        results = store.recall("what are bipolar vectors?", top_k=3)
        for thought, score in results:
            print(f"{score:.3f}  {thought.content}")

        store.save()
    """

    def __init__(
        self,
        encoder: ThoughtEncoder,
        storage_dir: str | Path = _DEFAULT_DIR,
    ) -> None:
        self._encoder = encoder
        self._dir = Path(storage_dir)
        self._thoughts: list[Thought] = []
        self._vectors: np.ndarray | None = None  # stacked (N, dim) matrix
        self._dirty = False

    # ── Public API ─────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Thought:
        """Create and store a new thought. Returns the thought."""
        thought = self._encoder.create_thought(content, metadata)
        self._thoughts.append(thought)
        self._rebuild_matrix()
        self._dirty = True
        logger.info("Stored thought %s: %s", thought.id, content[:60])
        return thought

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[tuple[Thought, float]]:
        """Recall thoughts by semantic similarity to a natural language query.

        Returns list of (thought, score) pairs sorted by score descending.
        Recalled thoughts are reinforced (Hebbian learning).
        """
        if not self._thoughts:
            return []

        query_vec = self._encoder.encode(query)
        scores = self._cosine_scores(query_vec)

        # Sort by weighted score (cosine * strength)
        weighted = [(i, scores[i] * self._thoughts[i].strength) for i in range(len(self._thoughts))]
        weighted.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in weighted[:top_k]:
            if score < min_score:
                break
            thought = self._thoughts[idx]
            thought.reinforce()
            results.append((thought, float(score)))
            self._dirty = True

        return results

    def forget(self, thought_id: str) -> bool:
        """Remove a thought by ID. Returns True if found and removed."""
        for i, t in enumerate(self._thoughts):
            if t.id == thought_id:
                self._thoughts.pop(i)
                self._rebuild_matrix()
                self._dirty = True
                return True
        return False

    def decay_all(self, rate: float = 0.01) -> int:
        """Apply temporal decay to all thoughts. Returns count of forgotten (strength=0)."""
        forgotten = 0
        for t in self._thoughts:
            t.decay(rate)
            if t.strength <= 0:
                forgotten += 1
        # Remove fully decayed thoughts
        if forgotten:
            self._thoughts = [t for t in self._thoughts if t.strength > 0]
            self._rebuild_matrix()
        self._dirty = True
        return forgotten

    @property
    def count(self) -> int:
        return len(self._thoughts)

    @property
    def thoughts(self) -> list[Thought]:
        return list(self._thoughts)

    # ── Persistence ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load thoughts from disk. Safe to call if files don't exist."""
        meta_path = self._dir / "thoughts.jsonl"
        vec_path = self._dir / "vectors.npy"

        if not meta_path.exists() or not vec_path.exists():
            logger.debug("No stored thoughts found at %s", self._dir)
            return

        thoughts = []
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                thoughts.append(data)

        vectors = np.load(vec_path)

        if len(thoughts) != vectors.shape[0]:
            logger.warning(
                "Thought count mismatch: %d metadata, %d vectors — rebuilding",
                len(thoughts), vectors.shape[0],
            )
            return

        self._thoughts = []
        for i, data in enumerate(thoughts):
            self._thoughts.append(Thought(
                id=data["id"],
                content=data["content"],
                vector=vectors[i],
                strength=data.get("strength", 1.0),
                created_at=data.get("created_at", 0),
                recalled_at=data.get("recalled_at"),
                metadata=data.get("metadata", {}),
            ))

        self._vectors = vectors
        self._dirty = False
        logger.info("Loaded %d thoughts from %s", len(self._thoughts), self._dir)

    def save(self) -> None:
        """Persist thoughts to disk."""
        if not self._dirty and self._dir.exists():
            return

        self._dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._dir / "thoughts.jsonl"
        vec_path = self._dir / "vectors.npy"

        with open(meta_path, "w") as f:
            for t in self._thoughts:
                record = {
                    "id": t.id,
                    "content": t.content,
                    "strength": t.strength,
                    "created_at": t.created_at,
                    "recalled_at": t.recalled_at,
                    "metadata": t.metadata,
                }
                f.write(json.dumps(record) + "\n")

        if self._thoughts:
            np.save(vec_path, np.stack([t.vector for t in self._thoughts]))
        elif vec_path.exists():
            vec_path.unlink()

        self._dirty = False
        logger.info("Saved %d thoughts to %s", len(self._thoughts), self._dir)

    # ── Internals ──────────────────────────────────────────────────────────

    def _rebuild_matrix(self) -> None:
        """Rebuild the stacked vector matrix after mutations."""
        if self._thoughts:
            self._vectors = np.stack([t.vector for t in self._thoughts])
        else:
            self._vectors = None

    def _cosine_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Compute cosine similarity of query against all stored vectors.

        Returns array of shape (N,) with scores in [-1, 1].
        Uses vectorized dot product — O(N * dim) but fast with numpy.
        """
        if self._vectors is None:
            return np.array([])

        # For bipolar vectors: cosine = dot / dim  (since ||v|| = sqrt(dim))
        dots = self._vectors.astype(np.float32) @ query_vec.astype(np.float32)
        return dots / (self._encoder.dimension)
