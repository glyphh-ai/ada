"""
StrandPredictor — online next-step prediction and trajectory surprise.

A nearest-class-mean model over strand states: for every label ever seen
following a history, a Hebbian prototype accumulates the (real-valued)
strand states that preceded it. Learning is one observation at a time,
permanent, and gradient-free; the model can keep learning in production.

Two products fall out of one structure:

- predict():  the most likely next label given the current strand state
              (next-intent suggestion, prefetch, tool pre-resolution)
- surprise(): how unlike the learned trajectories this continuation is
              (session-hijack / injection / workflow-drift detection)
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np


class StrandPredictor:
    """Online nearest-class-mean predictor over strand states."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._acc: Dict[Hashable, np.ndarray] = {}
        self._counts: Dict[Hashable, int] = {}

    def observe(self, state: np.ndarray, next_label: Hashable) -> None:
        """One-shot Hebbian update: this history preceded this label."""
        if state.shape[0] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: predictor is {self.dimension}, "
                f"state is {state.shape[0]}"
            )
        if next_label not in self._acc:
            self._acc[next_label] = np.zeros(self.dimension, dtype=np.float64)
            self._counts[next_label] = 0
        self._acc[next_label] += state.astype(np.float64)
        self._counts[next_label] += 1

    @property
    def labels(self) -> List[Hashable]:
        return list(self._acc.keys())

    def _prototype(self, label: Hashable) -> np.ndarray:
        acc = self._acc[label]
        norm = np.linalg.norm(acc)
        if norm == 0:
            return acc
        return acc / norm

    def similarity(self, state: np.ndarray, label: Hashable) -> float:
        """Cosine similarity of a strand state to a label's prototype."""
        if label not in self._acc:
            return 0.0
        s = state.astype(np.float64)
        norm = np.linalg.norm(s)
        if norm == 0:
            return 0.0
        return float(np.dot(self._prototype(label), s / norm))

    def predict(
        self, state: np.ndarray, top_k: int = 1
    ) -> List[Tuple[Hashable, float]]:
        """Most likely next labels for this strand state, best first."""
        if not self._acc:
            return []
        scored = [
            (label, self.similarity(state, label)) for label in self._acc
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def margin(self, state: np.ndarray, actual_label: Hashable) -> float:
        """
        Gap between the best-predicted label and the actual one, in [0, 2].

        Near 0 means the session did what histories like this one usually
        do next; large means the actual continuation ranked well below
        the expected one. Averaged over a session, this is the drift
        score that separates hijacked trajectories from normal ones.
        """
        top = self.predict(state, top_k=1)
        if not top:
            return 0.0
        actual = (
            self.similarity(state, actual_label)
            if actual_label in self._acc else -1.0
        )
        return top[0][1] - actual

    def surprise(self, state: np.ndarray, next_label: Hashable) -> float:
        """
        How unlike learned trajectories this continuation is, in [0, 2].

        0 means the actual next label sits exactly where histories like
        this one always led; an unseen label scores the maximum. Averaged
        over a session's turns, this is a drift/hijack score.
        """
        if next_label not in self._acc:
            return 2.0
        return 1.0 - self.similarity(state, next_label)

    def session_surprise(
        self,
        states: List[np.ndarray],
        labels: List[Hashable],
        unseen_penalty: Optional[float] = None,
    ) -> float:
        """Mean surprise across a session's (state, next-label) steps."""
        if len(states) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(states)} states, {len(labels)} labels"
            )
        if not states:
            raise ValueError("Cannot score an empty session")
        total = 0.0
        for state, label in zip(states, labels):
            s = self.surprise(state, label)
            if unseen_penalty is not None and label not in self._acc:
                s = unseen_penalty
            total += s
        return total / len(states)
