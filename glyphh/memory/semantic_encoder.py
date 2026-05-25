"""
Semantic encoder — a local Qwen3 embedding signal for recall.

This is the fix for the lexical-overlap blind spot: HDC content/role/structure
matching is deterministic and fast but deaf to paraphrase ("employs" vs "works").
A real sentence embedding closes that gap while staying FULLY LOCAL — no API,
no data leaving the box, which is the whole point of the runtime.

Graceful by design: if sentence-transformers or the model is unavailable, this
returns no vectors and recall falls back to pure HDC. Ada still works without it.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Apple-Silicon-friendly default; ~600MB, fully local. Override via env.
DEFAULT_MODEL = os.environ.get("ADA_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Qwen3-Embedding is asymmetric: queries get an instruction, documents don't.
_QUERY_PROMPT = (
    "Instruct: Given a question, retrieve facts that answer it\nQuery: "
)


class SemanticEncoder:
    """Lazy-loading, cached wrapper around a local sentence-embedding model.

    Thread-safe lazy load. Encodings are cached by text so stored thoughts are
    embedded once, not on every recall.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._available: Optional[bool] = None  # None = not yet attempted
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, bool], np.ndarray] = {}

    def _ensure_model(self) -> bool:
        if self._available is not None:
            return self._available
        with self._lock:
            if self._available is not None:
                return self._available
            try:
                # Silence HF/transformers tqdm bars (e.g. "Loading weights")
                # so the model load doesn't print to the shell.
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                try:
                    from transformers.utils import logging as _hf_logging
                    _hf_logging.disable_progress_bar()
                except Exception:
                    pass
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._available = True
                logger.info(f"Semantic encoder online: {self._model_name}")
            except Exception as e:  # ImportError or model load failure
                self._available = False
                logger.warning(
                    f"Semantic encoder offline ({e}). "
                    f"Recall falls back to HDC-only."
                )
            return self._available

    @property
    def available(self) -> bool:
        return self._ensure_model()

    @property
    def dim(self) -> Optional[int]:
        if not self._ensure_model():
            return None
        return self._model.get_sentence_embedding_dimension()

    def encode(self, text: str, is_query: bool = False) -> Optional[np.ndarray]:
        """Encode text to a unit vector. Cached. Returns None if unavailable."""
        if not text or not self._ensure_model():
            return None
        key = (text, is_query)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        payload = (_QUERY_PROMPT + text) if is_query else text
        try:
            # show_progress_bar=False: otherwise sentence-transformers prints a
            # tqdm "Batches" bar on every encode — and the background dream loop
            # recalls constantly, flooding the shell.
            vec = self._model.encode(
                payload, normalize_embeddings=True, show_progress_bar=False,
            )
            vec = np.asarray(vec, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Semantic encode failed: {e}")
            return None
        self._cache[key] = vec
        return vec

    @staticmethod
    def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Cosine of two already-normalized vectors. 0.0 if either is missing."""
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))


# Module-level singleton so the model loads once per process.
_SINGLETON: Optional[SemanticEncoder] = None
_SINGLETON_LOCK = threading.Lock()


def get_semantic_encoder() -> SemanticEncoder:
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = SemanticEncoder()
    return _SINGLETON
