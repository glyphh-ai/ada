"""Shared fixtures for glyphh.state tests."""

import pytest
import numpy as np

from glyphh import Encoder, EncoderConfig, Concept
from glyphh.core.config import Layer, Segment, Role
from glyphh.state import ConversationState, PathwayLibrary
from glyphh.state.pathway import PathwayEncoder


DIMENSION = 1000   # Small dimension for fast tests


@pytest.fixture(scope="session")
def encoder():
    """Minimal Encoder — one BoW role for function names."""
    config = EncoderConfig(
        dimension=DIMENSION,
        seed=42,
        include_temporal=False,
        apply_weights_during_encoding=False,
        layers=[
            Layer(
                name="func",
                segments=[
                    Segment(
                        name="identity",
                        roles=[
                            Role(
                                name="name",
                                text_encoding="bag_of_words",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    return Encoder(config)


def _make_glyph(encoder, name: str):
    """Encode a function name as a Glyph."""
    return encoder.encode(Concept(name=name, attributes={"name": name}))


@pytest.fixture(scope="session")
def glyphs(encoder):
    """Pre-encoded Glyphs for common filesystem functions."""
    names = ["cd", "mv", "cp", "grep", "ls", "find", "touch", "mkdir", "cat", "sort"]
    return {n: _make_glyph(encoder, n) for n in names}


@pytest.fixture
def state():
    """Fresh ConversationState for each test."""
    return ConversationState(dimension=DIMENSION, seed=42, decay=0.75)


@pytest.fixture
def pathway_encoder():
    return PathwayEncoder(dimension=DIMENSION, seed=42, decay=0.75)


@pytest.fixture
def library():
    return PathwayLibrary(dimension=DIMENSION, seed=42, decay=0.75)
