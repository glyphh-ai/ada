"""
glyphh.linguistics — Layered HDC language engine.

A 5-layer NLP stack built entirely on Glyphh HDC primitives.
No hardcoded domain vocabulary. No LLM. No external dependencies.

Layers:
  CharacterEncoder  — word → bipolar vector via positional char n-grams
  MorphologyEngine  — normalise plurals, conjugations, comparative forms
  POSTagger         — classify word POS via InductiveLayer prototype learning
  SyntaxParser      — extract verb/subject/object via HRR role binding
  HDCAttention      — context-aware feature weighting (Q·K^T·V in HD space)

Public API:
  LinguisticIntentParser — wraps all 5 layers; models seed it with examples

Quick start:
    from glyphh.linguistics import LinguisticIntentParser

    parser = LinguisticIntentParser()
    parser.seed_actions({"get": ["fetch", "find", "retrieve", "show"]})
    parser.seed_targets({"file": ["document", "folder", "directory"]})

    result = parser.extract_intent("fetch all documents from the folder")
    # → {"action": "get", "target": "file", "domain": "", "keywords": "..."}

    # Normalise a word (plural → singular, conjugated → infinitive, etc.)
    parser.normalize("sculptures")   # → "sculpture"
    parser.normalize("running")      # → "run"
"""

from glyphh.linguistics.attention import HDCAttention
from glyphh.linguistics.character import CharacterEncoder
from glyphh.linguistics.intent import LinguisticIntentParser
from glyphh.linguistics.morphology import MorphologyEngine
from glyphh.linguistics.pos import POSTagger
from glyphh.linguistics.syntax import ParseResult, SyntaxParser

__all__ = [
    "LinguisticIntentParser",
    "CharacterEncoder",
    "MorphologyEngine",
    "POSTagger",
    "SyntaxParser",
    "ParseResult",
    "HDCAttention",
]
