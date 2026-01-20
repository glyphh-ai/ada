from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_PATH = "models/intent/all-MiniLM-L6-v2"
DEFAULT_MIN_SCORE = 0.45


def _intent_texts(intent: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    name = intent.get("name")
    description = intent.get("description")
    if isinstance(name, str) and name.strip():
        parts.append(name.strip())
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    patterns = intent.get("patterns") or []
    if isinstance(patterns, list):
        parts.extend([p for p in patterns if isinstance(p, str) and p.strip()])
    examples = intent.get("examples") or []
    if isinstance(examples, list):
        parts.extend([e for e in examples if isinstance(e, str) and e.strip()])
    return parts


def _collect_intents(nl_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intents: List[Dict[str, Any]] = []
    for cfg in nl_configs:
        for intent in (cfg or {}).get("intents", []) or []:
            if isinstance(intent, dict) and intent.get("name"):
                intents.append(intent)
    return intents


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_path)


def infer_intent_with_model(
    text: str,
    nl_configs: List[Dict[str, Any]],
    *,
    model_path: Optional[str] = None,
    min_score: Optional[float] = None,
) -> Optional[Tuple[str, float]]:
    if not text or not text.strip():
        return None
    intents = _collect_intents(nl_configs)
    if not intents:
        return None
    model_path = model_path or os.getenv("GLYPH_INTENT_MODEL_PATH", DEFAULT_MODEL_PATH)
    min_score = float(os.getenv("GLYPH_INTENT_MIN_SCORE", min_score or DEFAULT_MIN_SCORE))
    model = _load_model(model_path)

    query_vec = model.encode([text], normalize_embeddings=True)[0]
    best_name = None
    best_score = -1.0
    for intent in intents:
        parts = _intent_texts(intent)
        if not parts:
            continue
        embeddings = model.encode(parts, normalize_embeddings=True)
        mean_vec = np.mean(embeddings, axis=0)
        score = float(np.dot(query_vec, mean_vec))
        if score > best_score:
            best_score = score
            best_name = intent.get("name")
    if best_name and best_score >= min_score:
        return best_name, best_score
    return None
