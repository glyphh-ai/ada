"""
DecisionSpace — sim across decisions, answered as a fact tree with confidence.

Not a forecaster: a precedent engine. Every decision the space has seen
is stored as a glyph plus its facts and outcome. A query encodes the
situation at hand, measures similarity across all past decisions, and
answers with a FactTree whose nodes are the matched precedents — each
one cited back to the stored decision it came from — an outcome
distribution derived from those precedents, and a confidence that means
something specific: how dense and how consistent the precedent is.

When the precedent is too thin (max similarity below threshold), the
answer says so explicitly, citing the nearest case and why it does not
apply, instead of guessing.
"""

from .space import Decision, DecisionAnswer, DecisionSpace

__all__ = ["Decision", "DecisionAnswer", "DecisionSpace"]
