"""
HDCAttention — Layer 5 of the Glyphh linguistics engine.

Context-aware feature selection in HD bipolar space. Given a query vector
and a set of (key, value) vector pairs, returns a weighted combination —
equivalent to single-head attention Q·K^T·V but operating entirely on
bipolar {-1, +1} vectors with no learned parameters.

Neural equivalence:
  Standard attention:   softmax(Q·K^T / √d) · V
  HDC attention:        sign(Σ  cosine(Q, Kᵢ) · Vᵢ)

The cosine(Q, Kᵢ) term plays the role of Q·Kᵢ^T (unnormalised dot product
in unit sphere, which bipolar vectors approximate). The weighted bundle
sign(Σ  wᵢ · Vᵢ) plays the role of the weighted sum.

No softmax, no temperature, no learned weights — the query vector IS the
"attention query" and the role/prototype vectors ARE the "keys".

Use cases:
  - Disambiguate word sense: "bank" near "river" → financial key gets low weight
  - Weight parse roles: in "send email to John", verb-role key gets high weight
    when the attention query is the action prototype
  - Domain-sensitive emphasis: payment-domain query boosts payment target key

attend() returns a single weighted vector (equivalent to the full attention
output — a blend of the value vectors weighted by relevance).

attend_roles() returns a dict of (role_name → weight) for interpretability
— which role is most relevant for a given query.
"""

from __future__ import annotations

import numpy as np

from glyphh.core.ops import cosine_similarity


class HDCAttention:
    """Single-head attention in HD bipolar space.

    No parameters, no training. Query vector determines which value vectors
    get highest weight in the output blend.

    Usage:
        attn = HDCAttention(dimension=10000)

        # Blend value vectors by relevance to query
        output = attn.attend(
            query=verb_prototype_vec,
            key_value_pairs=[
                (verb_role_key, verb_value_vec),
                (noun_role_key, noun_value_vec),
                (adj_role_key,  adj_value_vec),
            ],
        )

        # Or just get the weights (for interpretability / role selection)
        weights = attn.attend_roles(
            query=action_query_vec,
            roles={
                "action":  action_prototype_vec,
                "target":  target_prototype_vec,
                "domain":  domain_prototype_vec,
            },
        )
        # → {"action": 0.72, "target": 0.31, "domain": 0.18}
    """

    def __init__(self, dimension: int = 10000) -> None:
        self._dim = dimension

    def attend(
        self,
        query: np.ndarray,
        key_value_pairs: list[tuple[np.ndarray, np.ndarray]],
        relu: bool = True,
    ) -> np.ndarray:
        """Compute attention-weighted bundle of value vectors.

        Args:
            query: Query vector (bipolar int8, shape (dim,))
            key_value_pairs: List of (key_vec, value_vec) tuples.
                             cosine(query, key_i) is the weight for value_i.
            relu: If True, use ReLU on weights (only positive cosines count).
                  If False, negative cosines subtract from the bundle.

        Returns:
            Bipolar int8 vector of shape (dim,): weighted bundle of values.
            If all weights are zero/negative (with relu=True), returns the
            uniform bundle (all values equal weight).
        """
        if not key_value_pairs:
            raise ValueError("key_value_pairs must not be empty")

        weights = [
            float(cosine_similarity(query, key))
            for key, _ in key_value_pairs
        ]

        if relu:
            weights = [max(0.0, w) for w in weights]

        total_weight = sum(abs(w) for w in weights)
        if total_weight < 1e-9:
            # All orthogonal or all negative — uniform bundle
            vecs = [v for _, v in key_value_pairs]
            from glyphh.core.ops import bundle
            return bundle(vecs)

        # Weighted sum → binarise
        accumulated = np.zeros(self._dim, dtype=np.float32)
        for (_, value), weight in zip(key_value_pairs, weights):
            accumulated += value.astype(np.float32) * weight

        return np.where(accumulated >= 0, 1, -1).astype(np.int8)

    def attend_roles(
        self,
        query: np.ndarray,
        roles: dict[str, np.ndarray],
        relu: bool = True,
    ) -> dict[str, float]:
        """Compute attention weights for named role prototype vectors.

        Args:
            query: Query vector
            roles: Dict mapping role_name → prototype_vector
            relu: If True, clip negative cosines to 0.

        Returns:
            Dict mapping role_name → cosine weight (in [0, 1] with relu=True).
            Weights are normalised to sum to 1.0 if any are positive.
        """
        if not roles:
            return {}

        raw_weights = {
            name: float(cosine_similarity(query, proto))
            for name, proto in roles.items()
        }

        if relu:
            raw_weights = {k: max(0.0, v) for k, v in raw_weights.items()}

        total = sum(raw_weights.values())
        if total < 1e-9:
            # All orthogonal — uniform weights
            n = len(roles)
            return {k: round(1.0 / n, 4) for k in roles}

        return {k: round(v / total, 4) for k, v in raw_weights.items()}

    def top_role(
        self,
        query: np.ndarray,
        roles: dict[str, np.ndarray],
    ) -> tuple[str, float]:
        """Return the (role_name, weight) with the highest cosine similarity.

        Args:
            query: Query vector
            roles: Dict mapping role_name → prototype_vector

        Returns:
            (best_role_name, raw_cosine_score)
        """
        if not roles:
            raise ValueError("roles must not be empty")

        best_name = ""
        best_sim  = -2.0

        for name, proto in roles.items():
            sim = float(cosine_similarity(query, proto))
            if sim > best_sim:
                best_sim  = sim
                best_name = name

        return (best_name, round(best_sim, 4))
