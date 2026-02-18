"""
Fixture loader utilities for tests.

This module provides helper functions to load test fixtures from JSON files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from glyphh.core.types import Concept
from glyphh.core.config import EncoderConfig


FIXTURES_DIR = Path(__file__).parent


def load_sample_concepts() -> List[Concept]:
    """
    Load sample concepts from fixtures.
    
    Returns:
        List of Concept objects
    """
    with open(FIXTURES_DIR / "sample_concepts.json", "r") as f:
        data = json.load(f)
    
    concepts = []
    for concept_data in data["concepts"]:
        concepts.append(
            Concept(
                name=concept_data["name"],
                attributes=concept_data["attributes"],
                relationships=[tuple(r) for r in concept_data["relationships"]],
                metadata=concept_data["metadata"],
            )
        )
    
    return concepts


def load_test_configs() -> Dict[str, EncoderConfig]:
    """
    Load test configurations from fixtures.
    
    Returns:
        Dictionary mapping config names to EncoderConfig objects
    """
    with open(FIXTURES_DIR / "test_configs.json", "r") as f:
        data = json.load(f)
    
    configs = {}
    for config_data in data["configs"]:
        name = config_data["name"]
        config_dict = config_data["config"]
        
        configs[name] = EncoderConfig(
            dimension=config_dict["dimension"],
            seed=config_dict["seed"],
            similarity_weight=config_dict.get("similarity_weight", 1.0),
            security_weight=config_dict.get("security_weight", 1.0),
        )
    
    return configs


def get_concept_by_name(name: str) -> Concept:
    """
    Get a specific concept by name.
    
    Args:
        name: Concept name
        
    Returns:
        Concept object
        
    Raises:
        ValueError: If concept not found
    """
    concepts = load_sample_concepts()
    for concept in concepts:
        if concept.name == name:
            return concept
    
    raise ValueError(f"Concept '{name}' not found in fixtures")


def get_config_by_name(name: str) -> EncoderConfig:
    """
    Get a specific configuration by name.
    
    Args:
        name: Config name
        
    Returns:
        EncoderConfig object
        
    Raises:
        ValueError: If config not found
    """
    configs = load_test_configs()
    if name not in configs:
        raise ValueError(f"Config '{name}' not found in fixtures")
    
    return configs[name]
