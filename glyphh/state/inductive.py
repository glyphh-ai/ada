"""
InductiveLayer — HDC inductive reasoning for learning patterns from episodes.

A domain-agnostic few-shot classifier in HD space. Accumulates feature vectors
into labeled centroids via bundling, then predicts new episodes by finding the
closest centroid. This mirrors how the brain generalises:

  OBSERVE → GENERALISE → PREDICT
  (encode)   (bundle)     (cosine)

Where deductive reasoning is top-down (detect mismatch → infer prerequisite),
inductive reasoning is bottom-up (learn from examples → generalise pattern).

The layer maintains labeled centroids — each is a running bundled superposition
of all episode vectors with that label. Features are encoded as role-bound
bag-of-words, following the same pattern the SDK Encoder uses.

This is domain-agnostic — the same mechanism works for:
  - Auth: learn when login is needed from session + action patterns
  - Shopping: learn when cart-add is needed from browse + intent patterns
  - Navigation: learn when transition is needed from query + context patterns
  - Any domain where actions correlate with observable features

Load domain knowledge via packs:

    from glyphh.state import InductiveLayer

    # With domain packs — known patterns, no training data needed:
    inductive = InductiveLayer(dimension=10000, seed=97, packs=["my_domain"])

    # Predict immediately — packs provide the domain knowledge:
    result = inductive.predict(
        features={"query_tokens": "search budget analysis report"},
    )
    # -> {"label": "transition_needed", "confidence": 0.15, "scores": {...}}

    # Hebbian reinforcement
    inductive.confirm(was_correct=True, label="transition_needed")

    # Or learn additional patterns at runtime:
    inductive.learn(
        features={"query_tokens": "sort the output"},
        label="transition_not_needed",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol

# Data directory — bundled with the SDK package
_DATA_DIR = Path(__file__).parent / "data"


# ── Helpers ──

def _weighted_bundle(pairs: list[tuple[np.ndarray, float]], dimension: int) -> np.ndarray:
    """Weighted majority vote of bipolar vectors."""
    total = np.zeros(dimension, dtype=np.float32)
    for vec, weight in pairs:
        total += vec.astype(np.float32) * weight
    return np.where(total >= 0, 1, -1).astype(np.int8)


@dataclass
class Centroid:
    """A learned pattern centroid — bundled superposition of episode vectors.

    Each centroid represents a generalised pattern for one label. As more
    episodes are learned, the centroid stabilises toward the statistical
    centre of the label's distribution in HD space.

    strength tracks Hebbian reinforcement — correct predictions strengthen
    the centroid, incorrect ones weaken it.
    """

    label: str
    vector: np.ndarray          # Running bundled superposition
    count: int = 1              # Episodes accumulated
    strength: float = 1.0       # Hebbian weight [0.3, 3.0]

    def strengthen(self, amount: float = 0.1) -> None:
        """Hebbian reinforcement with diminishing returns."""
        self.strength = min(3.0, self.strength + amount * (1.0 / self.strength))

    def weaken(self, factor: float = 0.9) -> None:
        """Reduce strength but don't drop below 0.3."""
        self.strength = max(0.3, self.strength * factor)


class InductiveLayer:
    """HDC inductive reasoning — learn general patterns from observed episodes.

    Domain-agnostic few-shot classifier in HD space. Accumulates feature
    vectors into labeled centroids via bundling. Predicts by finding the
    closest centroid via cosine similarity, weighted by Hebbian strength.

    Features are encoded as role-bound bag-of-words:
      - Each feature key becomes a role vector (deterministic from seed)
      - Each feature value is split into words, each word → symbol vector
      - Words are bundled, then bound to the role
      - All role-value bindings are bundled into the episode vector

    This preserves structure (which words belong to which role) while
    allowing partial matching on shared vocabulary.
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: int = 97,
        min_confidence: float = 0.05,
        packs: list[str] | None = None,
    ):
        self._dim = dimension
        self._seed = seed
        self._min_confidence = min_confidence
        self._centroids: dict[str, Centroid] = {}

        # Load domain packs — known patterns, no training data needed
        if packs:
            self._load_packs(packs)

    def _load_packs(self, packs: list[str]) -> None:
        """Load domain knowledge packs and learn their patterns.

        Pack files live at glyphh/state/data/packs/{name}.json.
        Each pack contains labeled feature patterns that are learned
        into centroids, giving the layer domain knowledge without
        needing any training data.
        """
        packs_dir = _DATA_DIR / "packs"
        available = [p.stem for p in packs_dir.glob("*.json")] if packs_dir.exists() else []

        for pack_name in packs:
            pack_path = packs_dir / f"{pack_name}.json"
            if not pack_path.exists():
                raise FileNotFoundError(
                    f"Pack '{pack_name}' not found. Available packs: {available}"
                )
            with open(pack_path) as f:
                pack = json.load(f)

            for pattern in pack.get("patterns", []):
                label = pattern.get("label")
                features = pattern.get("features")
                if label and features:
                    self.learn(features, label)

    def _encode_features(self, features: dict[str, str]) -> np.ndarray:
        """Encode a feature dict as a role-bound BoW superposition.

        For each (key, value) pair:
          role_vec  = generate_symbol(seed, "role_{key}")
          value_vec = bundle([generate_symbol(seed, "val_{word}") for word in value.split()])
          binding   = bind(role_vec, value_vec)

        Result = bundle(all bindings)
        """
        bindings = []
        for key, value in sorted(features.items()):
            role_vec = generate_symbol(self._seed, f"role_{key}", self._dim)

            # Bag-of-words for multi-word values, symbol for single words
            words = str(value).lower().split()[:30]  # cap to prevent huge bundles
            if len(words) > 1:
                word_vecs = [
                    generate_symbol(self._seed, f"val_{w}", self._dim)
                    for w in words
                ]
                value_vec = bundle(word_vecs)
            elif words:
                value_vec = generate_symbol(self._seed, f"val_{words[0]}", self._dim)
            else:
                continue  # skip empty values

            bindings.append(bind(role_vec, value_vec))

        if not bindings:
            return generate_symbol(self._seed, "empty_episode", self._dim)

        return bundle(bindings) if len(bindings) > 1 else bindings[0]

    def learn(self, features: dict[str, str], label: str) -> None:
        """Accumulate an episode into the labeled centroid.

        New episodes are merged with the existing centroid using weighted
        bundling — old centroid weighted by count, new episode weighted by 1.
        This creates a running average in HD space that stabilises with
        more examples.

        Args:
            features: Feature dict (e.g. {"query": "search budget", "state": "state_x"}).
            label: The label for this episode (e.g. "transition_needed").
        """
        episode_vec = self._encode_features(features)

        if label not in self._centroids:
            self._centroids[label] = Centroid(
                label=label,
                vector=episode_vec,
                count=1,
            )
        else:
            centroid = self._centroids[label]
            # Online bundling: merge new episode into existing centroid
            centroid.vector = _weighted_bundle(
                [(centroid.vector, float(centroid.count)), (episode_vec, 1.0)],
                self._dim,
            )
            centroid.count += 1

    def predict(self, features: dict[str, str]) -> dict:
        """Find closest centroid for the given features.

        Args:
            features: Feature dict for the current episode.

        Returns:
            {
                "label": "transition_needed",       # Best label or None
                "confidence": 0.15,                  # Cosine x strength
                "scores": {"transition_needed": 0.15, "transition_not_needed": 0.08},
            }
        """
        if not self._centroids:
            return {"label": None, "confidence": 0.0, "scores": {}}

        episode_vec = self._encode_features(features)

        scores = {}
        for label, centroid in self._centroids.items():
            sim = cosine_similarity(episode_vec, centroid.vector)
            scores[label] = round(sim * centroid.strength, 4)

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        if best_score < self._min_confidence:
            return {"label": None, "confidence": 0.0, "scores": scores}

        return {
            "label": best_label,
            "confidence": round(best_score, 4),
            "scores": scores,
        }

    def confirm(self, was_correct: bool, label: str) -> None:
        """Hebbian reinforcement on the centroid that fired.

        Args:
            was_correct: Whether the prediction was correct.
            label: The label that was predicted.
        """
        if label in self._centroids:
            if was_correct:
                self._centroids[label].strengthen()
            else:
                self._centroids[label].weaken()

    @property
    def centroids(self) -> dict[str, Centroid]:
        """Access all learned centroids."""
        return dict(self._centroids)

    @property
    def labels(self) -> list[str]:
        """All learned labels."""
        return list(self._centroids.keys())

    def reset(self) -> None:
        """Reset strengths for a new conversation. Centroids persist.

        The learned patterns (centroids) are long-term memory —
        they survive across conversations. Only Hebbian strengths
        are reset to baseline.
        """
        for centroid in self._centroids.values():
            centroid.strength = 1.0
