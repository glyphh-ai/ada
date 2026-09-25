"""
Anticipator — strand-backed working memory for a session.

Owns one session's strand (the ordered trajectory of turn glyphs) and a
StrandPredictor that persists across sessions. Every recorded turn does
three things at once: scores how expected this turn was given the
history (drift), teaches the predictor that histories like this lead
here (one-shot, no gradients), and extends the strand.

The predictor is shared state — reset() clears the session, never the
learning — so a long-lived Anticipator gets better at its deployment's
workflows with every session it watches.
"""

from __future__ import annotations

from typing import Hashable, List, Tuple

import numpy as np

from .predictor import StrandPredictor
from .strand import Strand


class Anticipator:
    """Per-session trajectory tracking over a cross-session predictor."""

    def __init__(self, dimension: int, decay: float = 0.7):
        self.dimension = dimension
        self.decay = decay
        self.predictor = StrandPredictor(dimension)
        self._strand = Strand()
        self._margins: List[float] = []

    def reset(self) -> None:
        """Start a new session. Learning is kept; trajectory is cleared."""
        self._strand = Strand()
        self._margins = []

    @property
    def turns(self) -> int:
        return len(self._strand)

    def observe_turn(self, codon: np.ndarray, label: Hashable) -> float:
        """
        Record one turn: its glyph and what it resolved to.

        Returns this turn's margin (how far it fell below the expected
        continuation; 0.0 for the first turn of a session).
        """
        margin = 0.0
        if len(self._strand):
            state = self._strand.state(self.decay)
            margin = self.predictor.margin(state, label)
            self._margins.append(margin)
            self.predictor.observe(state, label)
        self._strand.append(codon)
        return margin

    def anticipate(self, top_k: int = 3) -> List[Tuple[Hashable, float]]:
        """Most likely labels for the NEXT turn, given the session so far."""
        if not len(self._strand):
            return []
        return self.predictor.predict(self._strand.state(self.decay), top_k)

    def drift(self) -> float:
        """
        Mean margin across this session's turns, in [0, 2].

        Near 0: the session is doing what sessions like it always do.
        High: continuations keep ranking below expectations — the
        trajectory-anomaly signal (hijack, injection, workflow drift).
        """
        if not self._margins:
            return 0.0
        return float(np.mean(self._margins))
