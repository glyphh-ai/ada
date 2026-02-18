"""
Custom Hypothesis strategies for property-based testing.

This module provides strategies for generating valid test data for the Glyphh SDK,
including encoder configurations, concepts, glyphs, and vectors.
"""

from hypothesis import strategies as st
from hypothesis.strategies import composite
import numpy as np
from typing import Dict, Any, List

from glyphh.core.types import Concept, EncoderConfig


@composite
def encoder_config_strategy(draw) -> EncoderConfig:
    """
    Generate valid encoder configurations for property testing.
    
    Returns:
        EncoderConfig with random but valid parameters
    """
    return EncoderConfig(
        dimension=draw(st.integers(min_value=1000, max_value=10000)),
        seed=draw(st.integers(min_value=0, max_value=10000)),
        num_layers=draw(st.integers(min_value=1, max_value=3)),
        segments_per_layer=draw(st.integers(min_value=1, max_value=4)),
        default_roles=draw(
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                    min_size=3,
                    max_size=15,
                ),
                min_size=1,
                max_size=5,
                unique=True,
            )
        ),
    )


@composite
def concept_strategy(draw) -> Concept:
    """
    Generate valid concepts for property testing.
    
    Returns:
        Concept with random but valid attributes
    """
    name = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=1,
            max_size=50,
        )
    )
    
    # Generate attributes dictionary
    num_attributes = draw(st.integers(min_value=1, max_value=5))
    attributes = {}
    for i in range(num_attributes):
        key = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                min_size=3,
                max_size=15,
            )
        )
        value = draw(
            st.one_of(
                st.text(min_size=1, max_size=30),
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False),
            )
        )
        attributes[key] = value
    
    # Generate relationships
    num_relationships = draw(st.integers(min_value=0, max_value=3))
    relationships = []
    for i in range(num_relationships):
        rel_type = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                min_size=3,
                max_size=15,
            )
        )
        target = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                min_size=1,
                max_size=30,
            )
        )
        relationships.append((rel_type, target))
    
    # Generate metadata
    metadata = {
        "source": draw(st.text(min_size=1, max_size=20)),
        "timestamp": draw(st.integers(min_value=0, max_value=2000000000)),
    }
    
    return Concept(
        name=name, attributes=attributes, relationships=relationships, metadata=metadata
    )


@composite
def bipolar_vector_strategy(draw, dimension: int) -> np.ndarray:
    """
    Generate valid bipolar vectors for property testing.
    
    Args:
        dimension: Vector dimension
        
    Returns:
        Numpy array with values in {-1, +1}
    """
    return np.array(
        [draw(st.sampled_from([-1, 1])) for _ in range(dimension)], dtype=np.int8
    )


@composite
def space_id_strategy(draw) -> str:
    """
    Generate valid space IDs for property testing.
    
    Returns:
        16-character hexadecimal string
    """
    return draw(
        st.text(
            alphabet="0123456789abcdef",
            min_size=16,
            max_size=16,
        )
    )


@composite
def weights_dict_strategy(draw) -> Dict[str, float]:
    """
    Generate valid weight dictionaries for property testing.
    
    Returns:
        Dictionary with weight values between 0.0 and 1.0
    """
    num_weights = draw(st.integers(min_value=1, max_value=5))
    weights = {}
    
    weight_keys = ["cortex", "layer", "segment", "role"]
    for i in range(min(num_weights, len(weight_keys))):
        weights[weight_keys[i]] = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return weights


@composite
def security_levels_strategy(draw) -> Dict[str, float]:
    """
    Generate valid security level dictionaries for property testing.
    
    Returns:
        Dictionary with security clearance values between 0.0 and 1.0
    """
    return {
        "cortex": draw(st.floats(min_value=0.0, max_value=1.0)),
        "layer": draw(st.floats(min_value=0.0, max_value=1.0)),
        "segment": draw(st.floats(min_value=0.0, max_value=1.0)),
        "role": draw(st.floats(min_value=0.0, max_value=1.0)),
    }


@composite
def version_string_strategy(draw) -> str:
    """
    Generate valid semantic version strings for property testing.
    
    Returns:
        Semantic version string in format X.Y.Z
    """
    major = draw(st.integers(min_value=0, max_value=10))
    minor = draw(st.integers(min_value=0, max_value=20))
    patch = draw(st.integers(min_value=0, max_value=100))
    
    return f"{major}.{minor}.{patch}"
