"""
Python wrapper for Rust performance operations.

This module provides a fallback to pure Python implementations if the Rust
engine is not available, ensuring the SDK works even without Rust compilation.
"""

import numpy as np
from typing import List, Optional

# Try to import Rust engine, fall back to Python if not available
try:
    from rust_engine import (
        bind as rust_bind,
        bundle as rust_bundle,
        cosine_similarity as rust_cosine_similarity,
        hamming_similarity as rust_hamming_similarity,
        generate_symbol as rust_generate_symbol,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    rust_bind = None
    rust_bundle = None
    rust_cosine_similarity = None
    rust_hamming_similarity = None
    rust_generate_symbol = None


def bind(r: np.ndarray, v: np.ndarray, use_rust: bool = True) -> np.ndarray:
    """
    Bind operation: element-wise multiplication of two bipolar vectors.
    
    Args:
        r: Role vector (bipolar: {-1, +1})
        v: Value vector (bipolar: {-1, +1})
        use_rust: Use Rust implementation if available (default: True)
    
    Returns:
        Bound vector (element-wise product)
    
    Raises:
        ValueError: If dimensions don't match
    """
    if r.shape != v.shape:
        raise ValueError(f"Dimension mismatch: r has shape {r.shape}, v has shape {v.shape}")
    
    if use_rust and RUST_AVAILABLE:
        return rust_bind(r.astype(np.int8), v.astype(np.int8))
    else:
        # Pure Python fallback
        return (r * v).astype(np.int8)


def bundle(vectors: List[np.ndarray], use_rust: bool = True) -> np.ndarray:
    """
    Bundle operation: majority-vote aggregation of multiple bipolar vectors.
    
    Args:
        vectors: List of bipolar vectors to bundle
        use_rust: Use Rust implementation if available (default: True)
    
    Returns:
        Bundled vector (majority vote per dimension)
    
    Raises:
        ValueError: If vector list is empty or dimensions don't match
    """
    if not vectors:
        raise ValueError("Cannot bundle empty vector list")
    
    # Validate dimensions
    dim = vectors[0].shape[0]
    for i, vec in enumerate(vectors):
        if vec.shape[0] != dim:
            raise ValueError(
                f"Dimension mismatch: vector 0 has {dim} dimensions, "
                f"vector {i} has {vec.shape[0]} dimensions"
            )
    
    if use_rust and RUST_AVAILABLE:
        # Convert to int8 for Rust
        vectors_int8 = [v.astype(np.int8) for v in vectors]
        return rust_bundle(vectors_int8)
    else:
        # Pure Python fallback
        sums = np.sum(vectors, axis=0)
        return np.where(sums >= 0, 1, -1).astype(np.int8)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray, use_rust: bool = True) -> float:
    """
    Cosine similarity for bipolar vectors.
    
    Range: [-1, 1]
    - 1.0: Identical vectors
    - 0.0: Orthogonal (50% agreement)
    - -1.0: Opposite vectors
    
    Args:
        v1: First bipolar vector
        v2: Second bipolar vector
        use_rust: Use Rust implementation if available (default: True)
    
    Returns:
        Cosine similarity score
    
    Raises:
        ValueError: If dimensions don't match
    """
    if v1.shape != v2.shape:
        raise ValueError(f"Dimension mismatch: v1 has shape {v1.shape}, v2 has shape {v2.shape}")
    
    if use_rust and RUST_AVAILABLE:
        return float(rust_cosine_similarity(v1.astype(np.int8), v2.astype(np.int8)))
    else:
        # Pure Python fallback
        # Cast to int32 to avoid overflow in dot product
        v1_int32 = v1.astype(np.int32)
        v2_int32 = v2.astype(np.int32)
        dot = np.dot(v1_int32, v2_int32)
        # For bipolar vectors {-1, +1}, norm is sqrt(dimension)
        # So norm1 * norm2 = dimension
        norm = len(v1)
        return float(dot / norm)


def hamming_similarity(v1: np.ndarray, v2: np.ndarray, use_rust: bool = True) -> float:
    """
    Hamming similarity for bipolar vectors.
    
    Range: [0, 1]
    - 1.0: Identical vectors
    - 0.5: Orthogonal (50% agreement)
    - 0.0: Opposite vectors
    
    Relationship: hamming = (cosine + 1) / 2
    
    Args:
        v1: First bipolar vector
        v2: Second bipolar vector
        use_rust: Use Rust implementation if available (default: True)
    
    Returns:
        Hamming similarity score
    
    Raises:
        ValueError: If dimensions don't match
    """
    if v1.shape != v2.shape:
        raise ValueError(f"Dimension mismatch: v1 has shape {v1.shape}, v2 has shape {v2.shape}")
    
    if use_rust and RUST_AVAILABLE:
        return float(rust_hamming_similarity(v1.astype(np.int8), v2.astype(np.int8)))
    else:
        # Pure Python fallback
        dimensions_agree = np.sum(v1 == v2)
        return float(dimensions_agree / len(v1))


def generate_symbol(seed: int, key: str, dimension: int, use_rust: bool = True) -> np.ndarray:
    """
    Generate a deterministic bipolar symbol vector.
    
    The same (seed, key, dimension) combination will always produce the same vector.
    This is critical for maintaining deterministic encoding guarantees.
    
    Args:
        seed: Base seed for the random number generator
        key: String key to generate vector for (e.g., "color", "red")
        dimension: Dimension of the vector to generate
        use_rust: Use Rust implementation if available (default: True)
    
    Returns:
        Deterministic bipolar vector
    """
    if use_rust and RUST_AVAILABLE:
        return rust_generate_symbol(seed, key, dimension)
    else:
        # Pure Python fallback using numpy's RandomState
        import hashlib
        
        # Create deterministic seed from base seed and key
        combined = f"{seed}:{key}:{dimension}"
        hash_digest = hashlib.sha256(combined.encode()).hexdigest()
        derived_seed = int(hash_digest[:8], 16)
        
        # Generate bipolar vector
        rng = np.random.RandomState(derived_seed)
        return rng.choice([-1, 1], size=dimension).astype(np.int8)


def is_rust_available() -> bool:
    """
    Check if Rust engine is available.
    
    Returns:
        True if Rust engine is available, False otherwise
    """
    return RUST_AVAILABLE


def get_backend_info() -> dict:
    """
    Get information about the current backend.
    
    Returns:
        Dictionary with backend information
    """
    return {
        "rust_available": RUST_AVAILABLE,
        "backend": "rust" if RUST_AVAILABLE else "python",
        "operations": {
            "bind": "rust" if RUST_AVAILABLE else "python",
            "bundle": "rust" if RUST_AVAILABLE else "python",
            "cosine_similarity": "rust" if RUST_AVAILABLE else "python",
            "hamming_similarity": "rust" if RUST_AVAILABLE else "python",
            "generate_symbol": "rust" if RUST_AVAILABLE else "python",
        }
    }
