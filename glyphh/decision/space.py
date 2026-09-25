"""
DecisionSpace: precedent storage, similarity search, fact-tree answers.

Encoding: a situation is a flat dict of attributes, each bound as
role(attr) * value and bundled; an optional event sequence is folded in
as a strand state bound under a trajectory role, so order-carrying
situations keep their order. Same substrate, both views.

Confidence semantics (all three numbers ride in the fact tree):
  support     mean positive similarity of the k precedents — how close
              the precedent is
  consensus   similarity-weighted share of the leading outcome — how
              much the precedent agrees
  confidence  consensus, gated by support: reported only when the
              precedent is sufficient (max similarity >= threshold)
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from glyphh.core.ops import bundle, generate_symbol
from glyphh.fact_tree.builder import Citation, FactTree
from glyphh.strand import Strand

SEED = 42


@dataclass
class Decision:
    """One recorded decision: what the situation was, what was decided."""

    decision_id: str
    situation: Dict[str, str]
    outcome: str
    facts: Dict[str, Any]
    timestamp: datetime
    glyph: np.ndarray


@dataclass
class DecisionAnswer:
    """A precedent-grounded answer to a decision query."""

    sufficient: bool
    confidence: float
    support: float
    outcomes: Dict[str, float]
    precedents: List[Tuple[str, float, str]]  # (decision_id, similarity, outcome)
    fact_tree: FactTree
    top_outcome: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "confidence": round(self.confidence, 4),
            "support": round(self.support, 4),
            "top_outcome": self.top_outcome,
            "outcomes": {k: round(v, 4) for k, v in self.outcomes.items()},
            "precedents": [
                {"decision_id": d, "similarity": round(s, 4), "outcome": o}
                for d, s, o in self.precedents
            ],
            "fact_tree": self.fact_tree.to_json(),
        }


class DecisionSpace:
    """Stores decisions as glyphs; answers queries with cited precedent."""

    def __init__(
        self,
        dimension: int = 4096,
        k: int = 10,
        min_precedent_sim: float = 0.15,
        decay: float = 0.7,
    ):
        self.dimension = dimension
        self.k = k
        self.min_precedent_sim = min_precedent_sim
        self.decay = decay
        self._symbols: Dict[str, np.ndarray] = {}
        self._decisions: List[Decision] = []
        self._matrix: Optional[np.ndarray] = None  # rebuilt lazily

    # ── encoding ────────────────────────────────────────────────────

    def _sym(self, key: str) -> np.ndarray:
        if key not in self._symbols:
            self._symbols[key] = generate_symbol(SEED, key, self.dimension)
        return self._symbols[key]

    def encode(
        self,
        situation: Dict[str, str],
        events: Optional[List[Dict[str, str]]] = None,
    ) -> np.ndarray:
        """Situation dict (+ optional ordered events) -> decision glyph."""
        parts = []
        for attr, value in sorted(situation.items()):
            parts.append(
                (self._sym(f"attr:{attr}") * self._sym(f"val:{value}"))
                .astype(np.int8)
            )
        if events:
            strand = Strand()
            for event in events:
                event_parts = [
                    (self._sym(f"eattr:{k}") * self._sym(f"eval:{v}"))
                    .astype(np.int8)
                    for k, v in sorted(event.items())
                ]
                strand.append(bundle(event_parts))
            parts.append(
                (self._sym("trajectory") * strand.state(self.decay))
                .astype(np.int8)
            )
        if not parts:
            raise ValueError("Cannot encode an empty situation")
        return bundle(parts)

    # ── recording ───────────────────────────────────────────────────

    def record(
        self,
        situation: Dict[str, str],
        outcome: str,
        facts: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, str]]] = None,
        decision_id: Optional[str] = None,
    ) -> str:
        """One-shot: store a decided case as precedent. No training run."""
        decision_id = decision_id or str(uuid.uuid4())[:12]
        self._decisions.append(Decision(
            decision_id=decision_id,
            situation=dict(situation),
            outcome=outcome,
            facts=dict(facts or {}),
            timestamp=datetime.now(),
            glyph=self.encode(situation, events),
        ))
        self._matrix = None
        return decision_id

    def __len__(self) -> int:
        return len(self._decisions)

    # ── querying ────────────────────────────────────────────────────

    def _citation(self, decision: Decision) -> Citation:
        return Citation(
            glyph_id=decision.decision_id,
            component="cortex",
            timestamp=decision.timestamp,
            version="v1",
            data_hash=hashlib.sha256(decision.glyph.tobytes()).hexdigest()[:16],
        )

    def query(
        self,
        situation: Dict[str, str],
        events: Optional[List[Dict[str, str]]] = None,
        k: Optional[int] = None,
    ) -> DecisionAnswer:
        """Sim across all past decisions -> fact tree + confidence."""
        k = k or self.k
        tree = FactTree()
        tree.add_fact(
            path=["query"],
            description="Decision under consideration",
            value=dict(situation),
            data_context={"events": len(events or []),
                          "precedent_pool": len(self._decisions)},
        )

        if not self._decisions:
            tree.add_fact(
                path=["insufficient_precedent"],
                description=("Insufficient precedent: no recorded "
                             "decisions to compare against"),
                value=None,
            )
            return DecisionAnswer(
                sufficient=False, confidence=0.0, support=0.0,
                outcomes={}, precedents=[], fact_tree=tree,
            )

        query_glyph = self.encode(situation, events).astype(np.float32)
        if self._matrix is None:
            self._matrix = np.stack(
                [d.glyph for d in self._decisions]).astype(np.float32)
        sims = (self._matrix @ query_glyph) / self.dimension
        order = np.argsort(sims)[::-1][:k]
        top = [(self._decisions[i], float(sims[i])) for i in order]

        for rank, (decision, sim) in enumerate(top, 1):
            tree.add_fact(
                path=["precedents"],
                description=(f"Precedent {rank}: {decision.decision_id} "
                             f"-> {decision.outcome}"),
                value=round(sim, 4),
                citations=[self._citation(decision)],
                data_sample={"situation": decision.situation,
                             "facts": decision.facts},
                math_explanation=("cosine(query, precedent) = "
                                  "dot / dimension for bipolar glyphs"),
            )

        best_sim = top[0][1]
        if best_sim < self.min_precedent_sim:
            tree.add_fact(
                path=["insufficient_precedent"],
                description=("Insufficient precedent: nearest case is below "
                             "the similarity threshold — no answer asserted"),
                value=round(best_sim, 4),
                citations=[self._citation(top[0][0])],
                data_context={"threshold": self.min_precedent_sim,
                              "nearest": top[0][0].decision_id},
            )
            return DecisionAnswer(
                sufficient=False, confidence=0.0,
                support=max(best_sim, 0.0), outcomes={},
                precedents=[(d.decision_id, s, d.outcome) for d, s in top],
                fact_tree=tree,
            )

        weights = np.array([max(s, 0.0) for _, s in top])
        weight_sum = float(weights.sum()) or 1.0
        outcomes: Dict[str, float] = {}
        for (decision, _), w in zip(top, weights):
            outcomes[decision.outcome] = (
                outcomes.get(decision.outcome, 0.0) + float(w))
        outcomes = {o: w / weight_sum for o, w in outcomes.items()}
        top_outcome = max(outcomes, key=outcomes.get)
        consensus = outcomes[top_outcome]
        support = float(weights.mean())
        confidence = consensus

        for outcome, prob in sorted(outcomes.items(), key=lambda x: -x[1]):
            tree.add_fact(
                path=["outcomes"],
                description=f"Outcome {outcome}",
                value=round(prob, 4),
                math_explanation=("similarity-weighted frequency over "
                                  f"{len(top)} precedents"),
            )
        tree.add_fact(
            path=["confidence"],
            description="Consensus among sufficient precedent",
            value=round(confidence, 4),
            math_explanation=("confidence = weighted share of leading "
                              "outcome; support = mean positive "
                              "similarity of precedents"),
            data_context={"support": round(support, 4),
                          "consensus": round(consensus, 4),
                          "k": len(top),
                          "min_precedent_sim": self.min_precedent_sim},
        )

        return DecisionAnswer(
            sufficient=True,
            confidence=confidence,
            support=support,
            outcomes=outcomes,
            precedents=[(d.decision_id, s, d.outcome) for d, s in top],
            fact_tree=tree,
            top_outcome=top_outcome,
        )
