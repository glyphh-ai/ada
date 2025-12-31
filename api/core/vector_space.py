from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from glyphh.config import DEFAULT_VECTOR_DIM

DEFAULT_ENCODER_SEED = 42
DEFAULT_SPACE_VERSION = 1


def _normalize_roles_config(roles_config: Dict[str, Any] | None) -> Dict[str, Any]:
    if roles_config is None:
        return {}
    if not isinstance(roles_config, dict):
        return {}
    return roles_config


def compute_space_id(
    vector_dim: int,
    encoder_seed: int,
    roles_config: Dict[str, Any] | None,
) -> str:
    normalized = json.dumps(_normalize_roles_config(roles_config), sort_keys=True, separators=(",", ":"))
    payload = f"{vector_dim}:{encoder_seed}:{normalized}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest


def build_vector_space_metadata(
    *,
    roles_config: Dict[str, Any] | None,
    vector_dim: int | None = None,
    encoder_seed: int | None = None,
    space_version: int | None = None,
) -> dict[str, Any]:
    final_vector_dim = vector_dim if vector_dim is not None else DEFAULT_VECTOR_DIM
    final_encoder_seed = encoder_seed if encoder_seed is not None else DEFAULT_ENCODER_SEED
    final_space_version = space_version if space_version is not None else DEFAULT_SPACE_VERSION
    space_id = compute_space_id(final_vector_dim, final_encoder_seed, roles_config)
    return {
        "vector_dim": final_vector_dim,
        "encoder_seed": final_encoder_seed,
        "space_id": space_id,
        "space_version": final_space_version,
    }
