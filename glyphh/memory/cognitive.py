"""
CognitiveLoop — Ada's reasoning engine.

The loop follows bind chains through facts, encoding each hop as a
pathway vector.  Completed chains become patterns in the PathwayLibrary.
Known patterns guide future reasoning — priming the next hop.  Confirmed
chains strengthen, rejected chains weaken, unused chains decay.

This is the bridge between knowledge (FactStore) and thought (PathwayLibrary).

Usage:
    loop = CognitiveLoop(forge, facts)

    # Reason about a question
    chains = loop.reason("billing", "responsible_for")

    # User confirms the answer
    loop.confirm(chains[0])

    # Next time — the pathway fires faster and more confident
    chains = loop.reason("database", "responsible_for")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from glyphh.core.ops import bind, cosine_similarity, generate_symbol
from .atom import AtomForge
from .binding import Fact, FactStore

from glyphh.state.pathway import Pathway, PathwayEncoder, PathwayLibrary

logger = logging.getLogger(__name__)


# ── ReasoningChain ────────────────────────────────────────────────────────

@dataclass
class ReasoningChain:
    """A single reasoning trace — the hops Ada took to reach an answer."""

    hops: list[tuple[str, str, str, float]]  # (subject, relation, object, confidence)
    answer: str | None = None
    confidence: float = 0.0
    pathway_vector: np.ndarray | None = None
    pathway_name: str | None = None
    pattern_boost: float = 0.0        # how much a known pattern helped
    contradicted: bool = False

    @property
    def depth(self) -> int:
        return len(self.hops)

    def __repr__(self) -> str:
        chain_str = " → ".join(f"{s} {r} {o}" for s, r, o, _ in self.hops)
        return f"Chain({chain_str} | answer={self.answer} conf={self.confidence:.3f})"


# ── CognitiveLoop ─────────────────────────────────────────────────────────

class CognitiveLoop:
    """Ada's reasoning engine — follows bind chains, learns patterns.

    The loop does three things:
      1. Reason — follow facts through HDC algebra
      2. Remember — encode successful chains as pathway patterns
      3. Reinforce — strengthen what works, weaken what doesn't

    Args:
        forge:       AtomForge for atom vectors and cleanup.
        facts:       FactStore for querying knowledge.
        dimension:   Must match forge.dimension (default 2048).
        seed:        Deterministic seed for pathway position symbols.
        decay:       Exponential decay for pathway working memory.
        max_depth:   Maximum reasoning hops.
        min_confidence: Minimum hop confidence to continue chain.
    """

    def __init__(
        self,
        forge: AtomForge,
        facts: FactStore,
        dimension: int = 2048,
        seed: int = 42,
        decay: float = 0.75,
        max_depth: int = 3,
        min_confidence: float = 0.03,
    ) -> None:
        self._forge = forge
        self._facts = facts
        self._dim = dimension
        self._seed = seed
        self._decay = decay
        self._max_depth = max_depth
        self._min_confidence = min_confidence

        self._library = PathwayLibrary(
            dimension=dimension, seed=seed, decay=decay
        )

        # Track last reasoning for confirm/reject
        self._last_chains: list[ReasoningChain] = []

    @property
    def library(self) -> PathwayLibrary:
        return self._library

    # ── Core reasoning ─────────────────────────────────────────────────────

    def reason(
        self,
        subject: str,
        relation: str | None = None,
        top_k: int = 5,
    ) -> list[ReasoningChain]:
        """Follow bind chains from subject, optionally filtered by relation.

        Returns all chains found, sorted by confidence (boosted by patterns).
        """
        # Decay all pathways slightly each time we reason
        self._library.decay_all(0.995)

        chains: list[ReasoningChain] = []

        if relation:
            # Targeted: find what subject <relation> via chains
            self._reason_targeted(subject, relation, chains)
        else:
            # Open: find everything connected to subject
            self._reason_open(subject, chains)

        # Auto-store: every completed chain becomes a weak pattern
        for chain in chains:
            if chain.depth >= 2 and chain.pathway_vector is not None:
                self._store_pattern(chain, strength=0.3)

        # Detect contradictions between chains
        self._detect_contradictions(chains)

        # Sort by confidence with pattern boost
        chains.sort(
            key=lambda c: c.confidence * (1.0 + c.pattern_boost),
            reverse=True,
        )

        self._last_chains = chains[:top_k]
        return self._last_chains

    def _reason_targeted(
        self,
        subject: str,
        relation: str,
        chains: list[ReasoningChain],
    ) -> None:
        """Find: subject <relation> → ? via multi-hop chains."""
        # Depth 1: direct facts
        direct = self._facts.query_object(subject, relation, top_k=5)
        for obj, score in direct:
            if score >= self._min_confidence:
                encoder = PathwayEncoder(self._dim, self._seed, self._decay)
                hop_vec = self._encode_hop(subject, relation, obj)
                encoder.update(hop_vec)

                chain = ReasoningChain(
                    hops=[(subject, relation, obj, score)],
                    answer=obj,
                    confidence=score,
                    pathway_vector=encoder.get_state(),
                )
                chain.pattern_boost = self._check_pattern(chain)
                chains.append(chain)

        if self._max_depth < 2:
            return

        # Depth 2+: follow connections from subject, then query from intermediates
        connections = self._facts._get_connections(subject)
        for intermediate, via_relation, conn_score in connections:
            if conn_score < self._min_confidence:
                continue

            # From intermediate, query for target relation
            indirect = self._facts.query_object(intermediate, relation, top_k=3)
            for obj, obj_score in indirect:
                combined = conn_score * obj_score
                if combined < self._min_confidence:
                    continue

                encoder = PathwayEncoder(self._dim, self._seed, self._decay)
                hop1_vec = self._encode_hop(subject, via_relation, intermediate)
                hop2_vec = self._encode_hop(intermediate, relation, obj)
                encoder.update(hop1_vec)
                encoder.update(hop2_vec)

                chain = ReasoningChain(
                    hops=[
                        (subject, via_relation, intermediate, conn_score),
                        (intermediate, relation, obj, obj_score),
                    ],
                    answer=obj,
                    confidence=combined,
                    pathway_vector=encoder.get_state(),
                )
                chain.pattern_boost = self._check_pattern(chain)
                chains.append(chain)

            # Depth 3: one more hop
            if self._max_depth >= 3:
                sub_connections = self._facts._get_connections(intermediate)
                for inter2, via_rel2, score2 in sub_connections[:3]:
                    if conn_score * score2 < self._min_confidence:
                        continue
                    deep = self._facts.query_object(inter2, relation, top_k=2)
                    for obj, obj_score in deep:
                        combined = conn_score * score2 * obj_score
                        if combined < self._min_confidence:
                            continue

                        encoder = PathwayEncoder(self._dim, self._seed, self._decay)
                        encoder.update(self._encode_hop(subject, via_relation, intermediate))
                        encoder.update(self._encode_hop(intermediate, via_rel2, inter2))
                        encoder.update(self._encode_hop(inter2, relation, obj))

                        chain = ReasoningChain(
                            hops=[
                                (subject, via_relation, intermediate, conn_score),
                                (intermediate, via_rel2, inter2, score2),
                                (inter2, relation, obj, obj_score),
                            ],
                            answer=obj,
                            confidence=combined,
                            pathway_vector=encoder.get_state(),
                        )
                        chain.pattern_boost = self._check_pattern(chain)
                        chains.append(chain)

    def _reason_open(
        self,
        subject: str,
        chains: list[ReasoningChain],
    ) -> None:
        """Find everything connected to subject."""
        connections = self._facts._get_connections(subject)
        for obj, relation, score in connections:
            if score >= self._min_confidence:
                encoder = PathwayEncoder(self._dim, self._seed, self._decay)
                hop_vec = self._encode_hop(subject, relation, obj)
                encoder.update(hop_vec)

                chain = ReasoningChain(
                    hops=[(subject, relation, obj, score)],
                    answer=obj,
                    confidence=score,
                    pathway_vector=encoder.get_state(),
                )
                chain.pattern_boost = self._check_pattern(chain)
                chains.append(chain)

    def _encode_hop(self, subject: str, relation: str, obj: str) -> np.ndarray:
        """Encode a single reasoning hop as an HDC vector."""
        s_vec = self._forge.atom(subject)
        r_vec = self._forge.atom(relation)
        o_vec = self._forge.atom(obj)
        # Bind all three — the hop is the structured association
        return bind(bind(s_vec, r_vec), o_vec)

    # ── Pattern matching ──────────────────────────────────────────────────

    def _check_pattern(self, chain: ReasoningChain) -> float:
        """Check if a chain matches a known pathway pattern. Returns boost."""
        if chain.pathway_vector is None or chain.depth < 2:
            return 0.0

        matches = self._library.match(chain.pathway_vector, top_k=1)
        if matches:
            pathway, score = matches[0]
            if score > 0.1:
                chain.pathway_name = pathway.name
                return score * pathway.strength
        return 0.0

    def _store_pattern(self, chain: ReasoningChain, strength: float = 0.5) -> str:
        """Store a reasoning chain as a new pathway pattern."""
        # Name from hop sequence hash
        hop_sig = "|".join(f"{s}_{r}_{o}" for s, r, o, _ in chain.hops)
        name = f"chain_{hashlib.md5(hop_sig.encode()).hexdigest()[:8]}"

        if chain.pathway_vector is not None and name not in self._library:
            # Store the pathway vector directly
            self._library._pathways[name] = Pathway(
                name=name,
                vector=chain.pathway_vector,
                strength=strength,
            )
            logger.debug("Stored pattern: %s (strength=%.2f)", name, strength)

        chain.pathway_name = name
        return name

    # ── Contradiction detection ───────────────────────────────────────────

    def _detect_contradictions(self, chains: list[ReasoningChain]) -> None:
        """Flag chains that lead to conflicting answers."""
        if len(chains) < 2:
            return

        # Group by answer atom
        by_answer: dict[str, list[ReasoningChain]] = {}
        for chain in chains:
            if chain.answer:
                by_answer.setdefault(chain.answer, []).append(chain)

        # If multiple different answers exist, check if any conflict
        answers = list(by_answer.keys())
        for i, a1 in enumerate(answers):
            for a2 in answers[i + 1:]:
                # Check if these answers are dissimilar (contradictory)
                if self._forge.has(a1) and self._forge.has(a2):
                    v1 = self._forge.atom(a1)
                    v2 = self._forge.atom(a2)
                    sim = self._forge.similarity(v1, v2)
                    if sim < 0.3:
                        # These are different enough to be contradictory
                        # Weaken the lower-confidence chains
                        all_chains = by_answer[a1] + by_answer[a2]
                        all_chains.sort(key=lambda c: c.confidence, reverse=True)
                        for chain in all_chains[1:]:
                            chain.contradicted = True

    # ── Reinforcement ─────────────────────────────────────────────────────

    def confirm(self, chain: ReasoningChain | None = None) -> None:
        """Confirm a reasoning chain was correct — Hebbian strengthening.

        If no chain is passed, confirms the best chain from the last reason() call.
        """
        if chain is None:
            chain = self._last_chains[0] if self._last_chains else None
        if chain is None:
            return

        # Strengthen existing pattern or boost a weak auto-stored one
        name = chain.pathway_name
        if name and name in self._library:
            p = self._library._pathways[name]
            if p.strength < 0.7:
                # Auto-stored weak pattern — promote it
                p.strength = 0.8
            self._library.strengthen(name, amount=0.15)
        else:
            name = self._store_pattern(chain, strength=0.8)

        # Reinforce the facts that were used
        for subj, rel, obj, _ in chain.hops:
            for fact in self._facts.facts:
                if fact.subject == subj and fact.relation == rel and fact.object == obj:
                    fact.reinforce(0.1)
                    break

        # Reinforce the atoms touched
        for subj, rel, obj, _ in chain.hops:
            self._forge.atom(subj)  # reinforce on access
            self._forge.atom(obj)

        logger.info("Confirmed chain: %s → %s", name, chain.answer)

    def reject(self, chain: ReasoningChain | None = None) -> None:
        """Reject a reasoning chain — weaken the pathway.

        Does NOT weaken individual facts — the facts might be correct,
        the chain might just be wrong.
        """
        if chain is None:
            chain = self._last_chains[0] if self._last_chains else None
        if chain is None:
            return

        name = chain.pathway_name
        if name and name in self._library:
            self._library.weaken(name, factor=0.7)
            logger.info("Rejected chain: %s (weakened)", name)
        elif chain.pathway_vector is not None:
            # Store as a weak pattern so we remember to avoid it
            name = self._store_pattern(chain, strength=0.2)
            logger.info("Rejected chain: %s (stored weak)", name)

    # ── Recall (replaces string matching) ─────────────────────────────────

    def recall(self, text: str) -> list[str]:
        """Recall facts relevant to a question, using reasoning chains.

        Returns formatted lines for injection into the LLM prompt.
        This replaces the string-matching hack.
        """
        # Extract content words — keep pronouns (my/your/i) since they
        # distinguish "what is my name" from "what is your name"
        stop = {"what", "does", "did", "do", "is", "am", "are", "the", "a", "an",
                "who", "how", "why", "when", "where", "can", "will", "would",
                "should", "tell", "me", "about", "for", "responsible"}
        # Keep pronouns (i/my/your) — they distinguish "my name" from "your name"
        pronouns = {"i", "my", "your", "you", "her", "his", "our", "we"}
        words = [w.lower().strip("?.,!") for w in text.split()
                 if w.lower().strip("?.,!") not in stop
                 and (len(w) > 1 or w.lower() in pronouns)]

        if not words:
            return []

        # Find atoms that match query words (including compound names)
        # Score atoms by how many query words they match — prefer specific hits
        atom_scores: dict[str, int] = {}
        for word in words:
            if self._forge.has(word):
                atom_scores[word] = atom_scores.get(word, 0) + 1
            for atom in self._forge.all_atoms():
                if atom.kind != "role" and word in atom.name:
                    atom_scores[atom.name] = atom_scores.get(atom.name, 0) + 1

        if not atom_scores:
            return []

        # Sort by match count — atoms matching more query words come first
        ranked = sorted(atom_scores.items(), key=lambda x: -x[1])

        # Reason from the best-matching atoms
        all_chains: list[ReasoningChain] = []
        for atom_name, _score in ranked[:4]:
            # Open reasoning — find everything connected
            chains = self.reason(atom_name, relation=None, top_k=5)
            all_chains.extend(chains)

            # Also try with common relations
            for rel in words:
                if rel != atom_name:
                    chains = self.reason(atom_name, relation=rel, top_k=3)
                    all_chains.extend(chains)

        # Deduplicate and format
        seen = set()
        lines = []
        for chain in all_chains:
            for subj, rel, obj, conf in chain.hops:
                key = (subj, rel, obj)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"- {subj} {rel} {obj}")

            # Show inferred conclusions
            if chain.depth > 1 and chain.answer:
                hop_str = " → ".join(
                    f"{s} {r} {o}" for s, r, o, _ in chain.hops
                )
                line = f"- inferred: {hop_str} (confidence: {chain.confidence:.2f})"
                if line not in lines:
                    lines.append(line)

        return lines

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the pathway library to disk."""
        self._library.prune(min_strength=0.15)
        self._library.save(Path(path))

    def load(self, path: str | Path) -> None:
        """Load the pathway library from disk."""
        self._library.load(Path(path))
