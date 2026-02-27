"""
glyphh.state — Conversation state management via HDC pathway encoding.

State is encoded as a rolling superposition of position-bound action vectors —
a neural pathway in HD space.  Frequently-used patterns are strengthened
through Hebbian reinforcement (fire → wire), making the model progressively
better at predicting next actions from conversation context.

Core classes:

  ConversationState  — tracks trajectory, predicts next action, confirms results
  Pathway            — a named action-sequence pattern with a strength score
  PathwayLibrary     — collection of patterns with Hebbian strengthening

Usage:

    from glyphh.state import ConversationState

    state = ConversationState(dimension=10000, seed=42, decay=0.75)

    # Optional: pre-seed known patterns as prior knowledge
    state.add_pathway("navigate_then_operate", [cd_glyph, mv_glyph])

    # Each turn: update with what was called, predict what comes next
    state.update(action_glyphs=[cd_glyph, mv_glyph])
    scores = state.predict_next(query_glyph, candidates)

    # After ground truth — Hebbian reinforcement
    state.confirm(confirmed_glyphs=[mv_glyph])

    # New conversation — reset encoder, library persists
    state.reset()
"""

from glyphh.state.pathway import Pathway, PathwayLibrary
from glyphh.state.tracker import ConversationState

__all__ = [
    "ConversationState",
    "Pathway",
    "PathwayLibrary",
]
