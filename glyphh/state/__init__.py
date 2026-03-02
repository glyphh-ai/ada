"""
glyphh.state — Conversation state management via HDC pathway encoding.

State is encoded as a rolling superposition of position-bound action vectors —
a neural pathway in HD space.  Frequently-used patterns are strengthened
through Hebbian reinforcement (fire → wire), making the model progressively
better at predicting next actions from conversation context.

Core classes:

  ConversationState  — tracks trajectory, predicts next action, confirms results
  DeductiveLayer     — detects implicit prerequisites via predictive coding mismatch
  InductiveLayer     — learns general patterns from episodes via centroid classification
  Pathway            — a named action-sequence pattern with a strength score
  PathwayLibrary     — collection of patterns with Hebbian strengthening
  Centroid           — a learned pattern centroid in the inductive layer
  Transition         — a learned prerequisite pattern in the deductive layer

Usage:

    from glyphh.state import ConversationState, DeductiveLayer

    state = ConversationState(dimension=10000, seed=42, decay=0.75)

    # Optional: pre-seed known patterns as prior knowledge
    state.add_pathway("setup_then_execute", [action_a_glyph, action_b_glyph])

    # Each turn: update with what was called, predict what comes next
    state.update(action_glyphs=[action_a_glyph, action_b_glyph])
    scores = state.predict_next(query_glyph, candidates)

    # After ground truth — Hebbian reinforcement
    state.confirm(confirmed_glyphs=[action_b_glyph])

    # Deductive reasoning — detect implicit prerequisites
    deductive = DeductiveLayer(dimension=10000, seed=89)
    deductive.add_transition("context_switch", ["move", "copy"], ["read", "search"], "navigate")
    deductive.observe(state="state_x", actions=["move"], targets=["state_y"])
    result = deductive.deduce(query="search the items", current_state="state_x")
    # → {"prerequisites": ["navigate"], "target": "state_y", ...}

    # Inductive reasoning — learn patterns from episodes
    inductive = InductiveLayer(dimension=10000, seed=97)
    inductive.learn({"query": "search budget report", "state": "state_x"}, "transition_needed")
    inductive.learn({"query": "sort the output", "state": "state_y"}, "transition_not_needed")
    result = inductive.predict({"query": "search for budget", "state": "state_x"})
    # → {"label": "transition_needed", "confidence": 0.15, ...}

    # New conversation — reset encoder, library persists
    state.reset()
"""

from glyphh.state.deductive import DeductiveLayer, Transition
from glyphh.state.inductive import Centroid, InductiveLayer
from glyphh.state.pathway import Pathway, PathwayLibrary
from glyphh.state.tracker import ConversationState

__all__ = [
    "Centroid",
    "ConversationState",
    "DeductiveLayer",
    "InductiveLayer",
    "Pathway",
    "PathwayLibrary",
    "Transition",
]
