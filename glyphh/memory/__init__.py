"""
glyphh.memory — Ada's lifelong memory system.

Three layers, like a brain:
  Atom       — 2048-dim neuron-level concept primitives
  Binding    — structured facts (role-filler frames from atoms)
  Thought    — flat text-encoded memories (legacy, used for simple recall)

Plus:
  Teacher    — decomposes language into atoms + bindings
  ThoughtStore — persistence for flat thoughts (cosine recall)
  FactStore  — persistence for structured facts (algebraic query + inference)
"""

from .atom import Atom, AtomForge
from .binding import Fact, FactStore
from .cognitive import CognitiveLoop, ReasoningChain
from .dream import DreamLoop, Insight, InsightKind
from .thought import Thought, ThoughtEncoder
from .store import ThoughtStore
from .teacher import Teacher

__all__ = [
    "Atom",
    "AtomForge",
    "CognitiveLoop",
    "DreamLoop",
    "Fact",
    "FactStore",
    "Insight",
    "InsightKind",
    "ReasoningChain",
    "Teacher",
    "Thought",
    "ThoughtEncoder",
    "ThoughtStore",
]
