"""
Custom Hypothesis strategies for Glyphh Runtime property-based tests.

Provides generators for:
- Glyphs (concept text, embeddings, metadata)
- Models (namespaces, configurations)
- Users (permissions, tokens)
- Queries (search, fact tree, temporal)
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy


# =============================================================================
# Basic Strategies
# =============================================================================

@st.composite
def valid_namespace(draw) -> str:
    """Generate a valid namespace string."""
    prefix = draw(st.sampled_from(["model", "test", "prod", "dev"]))
    suffix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=4,
        max_size=8,
    ))
    return f"{prefix}_{suffix}"


@st.composite
def valid_concept_text(draw) -> str:
    """Generate valid concept text."""
    # Generate meaningful-ish text
    words = draw(st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz ",
            min_size=3,
            max_size=15,
        ),
        min_size=1,
        max_size=10,
    ))
    return " ".join(words).strip()


@st.composite
def valid_embedding(draw, dim: int = 768) -> List[float]:
    """Generate a valid embedding vector."""
    # Generate normalized random vector
    values = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim,
    ))
    # Normalize to unit length
    norm = np.linalg.norm(values)
    if norm > 0:
        values = [v / norm for v in values]
    return values


@st.composite
def valid_metadata(draw) -> Dict[str, Any]:
    """Generate valid metadata dictionary."""
    return draw(st.fixed_dictionaries({
        "source": st.sampled_from(["api", "listener", "batch", "import"]),
        "version": st.integers(min_value=1, max_value=100),
        "tags": st.lists(st.text(min_size=1, max_size=20), max_size=5),
    }, optional={
        "security_level": st.floats(min_value=0.0, max_value=1.0),
        "expires_at": st.datetimes(min_value=datetime.now()),
    }))


# =============================================================================
# Glyph Strategies
# =============================================================================

@st.composite
def glyph_create_request(draw) -> Dict[str, Any]:
    """Generate a valid glyph creation request."""
    return {
        "concept": draw(valid_concept_text()),
        "metadata": draw(st.one_of(st.none(), valid_metadata())),
    }


@st.composite
def glyph_data(draw, namespace: Optional[str] = None) -> Dict[str, Any]:
    """Generate complete glyph data for storage."""
    return {
        "namespace": namespace or draw(valid_namespace()),
        "concept_text": draw(valid_concept_text()),
        "embedding": draw(valid_embedding()),
        "metadata": draw(valid_metadata()),
    }


@st.composite
def batch_glyph_request(draw, max_size: int = 10) -> Dict[str, Any]:
    """Generate a batch glyph creation request."""
    concepts = draw(st.lists(
        valid_concept_text(),
        min_size=1,
        max_size=max_size,
    ))
    return {
        "concepts": concepts,
        "metadata": draw(st.one_of(st.none(), valid_metadata())),
    }


# =============================================================================
# Query Strategies
# =============================================================================

@st.composite
def similarity_search_request(draw) -> Dict[str, Any]:
    """Generate a similarity search request."""
    return {
        "query": draw(valid_concept_text()),
        "top_k": draw(st.integers(min_value=1, max_value=100)),
        "filters": draw(st.one_of(st.none(), st.fixed_dictionaries({
            "source": st.sampled_from(["api", "listener", "batch"]),
        }))),
        "include_embeddings": draw(st.booleans()),
    }


@st.composite
def fact_tree_request(draw) -> Dict[str, Any]:
    """Generate a fact tree request."""
    return {
        "claim": draw(valid_concept_text()),
        "max_depth": draw(st.integers(min_value=1, max_value=10)),
        "branching_factor": draw(st.integers(min_value=1, max_value=20)),
    }


@st.composite
def temporal_predict_request(draw) -> Dict[str, Any]:
    """Generate a temporal prediction request."""
    return {
        "current_state": draw(st.lists(
            valid_concept_text(),
            min_size=1,
            max_size=5,
        )),
        "steps_ahead": draw(st.integers(min_value=1, max_value=10)),
        "beam_width": draw(st.integers(min_value=1, max_value=20)),
        "direction": draw(st.sampled_from(["forward", "backward"])),
    }


# =============================================================================
# Model Configuration Strategies
# =============================================================================

@st.composite
def similarity_weights(draw) -> Dict[str, float]:
    """Generate similarity weights configuration."""
    return {
        "similarity": draw(st.floats(min_value=0.0, max_value=1.0)),
        "contrast": draw(st.floats(min_value=0.0, max_value=1.0)),
        "analogy": draw(st.floats(min_value=0.0, max_value=1.0)),
        "composition": draw(st.floats(min_value=0.0, max_value=1.0)),
        "precedes": draw(st.floats(min_value=0.0, max_value=1.0)),
        "follows": draw(st.floats(min_value=0.0, max_value=1.0)),
        "causes": draw(st.floats(min_value=0.0, max_value=1.0)),
        "prevents": draw(st.floats(min_value=0.0, max_value=1.0)),
    }


@st.composite
def model_config_update(draw) -> Dict[str, Any]:
    """Generate a model configuration update request."""
    return {
        "similarity_weights": draw(st.one_of(st.none(), similarity_weights())),
        "beam_width": draw(st.one_of(st.none(), st.integers(min_value=1, max_value=50))),
        "max_tree_depth": draw(st.one_of(st.none(), st.integers(min_value=1, max_value=20))),
    }


@st.composite
def resource_quota(draw) -> Dict[str, Any]:
    """Generate resource quota configuration."""
    return {
        "memory_mb": draw(st.integers(min_value=128, max_value=8192)),
        "storage_gb": draw(st.integers(min_value=1, max_value=100)),
        "max_glyphs": draw(st.integers(min_value=1000, max_value=10000000)),
        "max_edges": draw(st.integers(min_value=5000, max_value=50000000)),
        "max_requests_per_minute": draw(st.integers(min_value=10, max_value=1000)),
    }


# =============================================================================
# User/Auth Strategies
# =============================================================================

@st.composite
def user_permissions(draw) -> Dict[str, Any]:
    """Generate user permissions."""
    return {
        "user_id": str(uuid4()),
        "namespaces": draw(st.lists(valid_namespace(), min_size=1, max_size=5)),
        "security_level": draw(st.floats(min_value=0.0, max_value=1.0)),
    }


@st.composite
def jwt_claims(draw) -> Dict[str, Any]:
    """Generate JWT token claims."""
    return {
        "sub": str(uuid4()),
        "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        "iat": int(datetime.utcnow().timestamp()),
        "namespaces": draw(st.lists(valid_namespace(), min_size=1, max_size=3)),
        "permissions": draw(st.lists(
            st.sampled_from(["read", "write", "admin"]),
            min_size=1,
            max_size=3,
        )),
    }


# =============================================================================
# Edge Strategies
# =============================================================================

@st.composite
def edge_type(draw) -> str:
    """Generate a valid edge type."""
    return draw(st.sampled_from([
        "similarity", "contrast", "analogy", "composition",
        "precedes", "follows", "causes", "prevents",
    ]))


@st.composite
def edge_data(draw, namespace: Optional[str] = None) -> Dict[str, Any]:
    """Generate edge data."""
    return {
        "namespace": namespace or draw(valid_namespace()),
        "source_glyph_id": uuid4(),
        "target_glyph_id": uuid4(),
        "edge_type": draw(edge_type()),
        "weight": draw(st.floats(min_value=0.0, max_value=1.0)),
        "metadata": draw(st.one_of(st.none(), valid_metadata())),
    }
