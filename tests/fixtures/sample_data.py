"""
Sample data fixtures for Glyphh Runtime tests.
"""

import jwt
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


# =============================================================================
# Sample Glyphs
# =============================================================================

def create_sample_embedding(seed: int = 42) -> List[float]:
    """Create a deterministic sample embedding."""
    np.random.seed(seed)
    embedding = np.random.randn(768).astype(float)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def create_sample_glyph(
    org_id: str = "test_org",
    model_id: str = "test_model",
    concept_text: str = "sample concept",
    glyph_id: Optional[UUID] = None,
    seed: int = 42,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a sample glyph for testing."""
    return {
        "id": glyph_id or uuid4(),
        "org_id": org_id,
        "model_id": model_id,
        "concept_text": concept_text,
        "embedding": create_sample_embedding(seed),
        "metadata": metadata or {"source": "test"},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


# Pre-defined sample glyphs
SAMPLE_GLYPHS = [
    create_sample_glyph(
        concept_text="The quick brown fox jumps over the lazy dog",
        seed=1,
    ),
    create_sample_glyph(
        concept_text="Machine learning is a subset of artificial intelligence",
        seed=2,
    ),
    create_sample_glyph(
        concept_text="Python is a popular programming language",
        seed=3,
    ),
    create_sample_glyph(
        concept_text="Data science involves statistics and programming",
        seed=4,
    ),
    create_sample_glyph(
        concept_text="Neural networks are inspired by biological neurons",
        seed=5,
    ),
]


# =============================================================================
# Sample Edges
# =============================================================================

def create_sample_edge(
    org_id: str = "test_org",
    model_id: str = "test_model",
    source_glyph_id: Optional[UUID] = None,
    target_glyph_id: Optional[UUID] = None,
    edge_type: str = "similarity",
    weight: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a sample edge for testing."""
    return {
        "id": uuid4(),
        "org_id": org_id,
        "model_id": model_id,
        "source_glyph_id": source_glyph_id or uuid4(),
        "target_glyph_id": target_glyph_id or uuid4(),
        "edge_type": edge_type,
        "weight": weight,
        "metadata": metadata or {},
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=24),
    }


# Pre-defined sample edges
SAMPLE_EDGES = [
    create_sample_edge(edge_type="similarity", weight=0.9),
    create_sample_edge(edge_type="contrast", weight=0.3),
    create_sample_edge(edge_type="analogy", weight=0.7),
    create_sample_edge(edge_type="precedes", weight=0.8),
    create_sample_edge(edge_type="causes", weight=0.6),
]


# =============================================================================
# Sample Model Configuration
# =============================================================================

SAMPLE_MODEL_CONFIG = {
    "org_id": "test_org",
    "model_id": "test_model",
    "model_path": "/path/to/test.glyphh",
    "model_version": "1.0.0",
    "sdk_version": "0.1.0",
    "similarity_weights": {
        "similarity": 1.0,
        "contrast": 0.5,
        "analogy": 0.7,
        "composition": 0.8,
        "precedes": 0.6,
        "follows": 0.6,
        "causes": 0.9,
        "prevents": 0.4,
    },
    "beam_width": 5,
    "max_tree_depth": 3,
    "resource_quotas": {
        "memory_mb": 1024,
        "storage_gb": 10,
        "max_glyphs": 1000000,
    },
    "resource_usage": {
        "memory_mb": 0,
        "storage_gb": 0,
        "glyph_count": 0,
    },
}


# =============================================================================
# Sample JWT Tokens
# =============================================================================

def create_sample_jwt(
    user_id: str = "test_user",
    org_ids: Optional[List[str]] = None,
    permissions: Optional[List[str]] = None,
    secret_key: str = "test_secret_key",
    expires_in_hours: int = 1,
) -> str:
    """Create a sample JWT token for testing."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(hours=expires_in_hours),
        "org_ids": org_ids or ["test_org"],
        "permissions": permissions or ["read", "write"],
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


# Pre-defined sample tokens
SAMPLE_JWT_TOKEN = create_sample_jwt()

SAMPLE_ADMIN_JWT_TOKEN = create_sample_jwt(
    user_id="admin_user",
    permissions=["read", "write", "admin"],
)

SAMPLE_READONLY_JWT_TOKEN = create_sample_jwt(
    user_id="readonly_user",
    permissions=["read"],
)

SAMPLE_EXPIRED_JWT_TOKEN = create_sample_jwt(
    expires_in_hours=-1,  # Already expired
)


# =============================================================================
# Sample API Requests
# =============================================================================

SAMPLE_SEARCH_REQUEST = {
    "query": "machine learning",
    "top_k": 10,
    "filters": None,
    "include_embeddings": False,
}

SAMPLE_FACT_TREE_REQUEST = {
    "claim": "Python is a programming language",
    "max_depth": 3,
    "branching_factor": 5,
}

SAMPLE_TEMPORAL_PREDICT_REQUEST = {
    "current_state": ["initial state", "context"],
    "steps_ahead": 3,
    "beam_width": 5,
    "direction": "forward",
}

SAMPLE_GLYPH_CREATE_REQUEST = {
    "concept": "New test concept",
    "metadata": {"source": "test", "version": 1},
}

SAMPLE_BATCH_CREATE_REQUEST = {
    "concepts": [
        "First concept",
        "Second concept",
        "Third concept",
    ],
    "metadata": {"batch": True},
}
