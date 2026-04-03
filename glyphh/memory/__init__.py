"""
glyphh.memory — Ada's lifelong memory system.

New architecture (Thought Glyphs):
  PrimitiveSpace     — permanent exemplar Glyphs (loaded from standard JSONL)
  ThoughtGlyphEncoder — encodes text into 5-layer structured Glyphs
  ThoughtGlyphSpace  — absorb, store, recall thought glyphs
  GlyphCognitiveLoop — multi-hop reasoning with activation pathways
  GlyphDreamLoop     — dual-loop background reasoning (localized + deep)

Legacy (kept for backward compatibility — tests still reference these):
  Atom, AtomForge    — 2000-dim concept primitives
  Fact, FactStore    — structured facts (role-filler frames)
  CognitiveLoop      — atom/fact bind-chain reasoning
  DreamLoop          — single-loop background reasoning
  Teacher            — curriculum-based lesson loader
  Thought, ThoughtEncoder, ThoughtStore — flat text memories
"""

# New glyph-based system
from .primitives import PrimitiveSpace
from .thought_glyph import ThoughtGlyphEncoder
from .thought_space import ThoughtGlyphSpace, StoredThought, RecallResult
from .glyph_cognitive import GlyphCognitiveLoop, GlyphReasoningChain
from .glyph_dream import GlyphDreamLoop, Insight, InsightKind

# Legacy (backward compat)
from .atom import Atom, AtomForge
from .binding import Fact, FactStore
from .cognitive import CognitiveLoop, ReasoningChain
from .dream import DreamLoop
from .thought import Thought, ThoughtEncoder
from .store import ThoughtStore
from .teacher import Teacher

__all__ = [
    # New
    "PrimitiveSpace",
    "ThoughtGlyphEncoder",
    "ThoughtGlyphSpace",
    "StoredThought",
    "RecallResult",
    "GlyphCognitiveLoop",
    "GlyphReasoningChain",
    "GlyphDreamLoop",
    "Insight",
    "InsightKind",
    # Legacy
    "Atom",
    "AtomForge",
    "CognitiveLoop",
    "DreamLoop",
    "Fact",
    "FactStore",
    "ReasoningChain",
    "Teacher",
    "Thought",
    "ThoughtEncoder",
    "ThoughtStore",
]
