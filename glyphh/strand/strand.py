"""
Strand operations: ordered permute-bind chains over bipolar vectors.

A strand keeps its codon list as the source of truth (the genotype) and
derives three views from it:

- state():        an order-sensitive recurrent summary vector, anchored at
                  the most recent codon (age 0), with exponential decay —
                  the working-memory view used for prediction and kNN
- kmer_profile(): an order-insensitive bundle of locally-ordered k-windows
                  — the searchable lattice view (k-mers, as in genomics)
- duplex():       pairwise binding against a template strand, from which
                  either strand is exactly recoverable (transcribe)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from glyphh.core.ops import bundle


def permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Permutation operator: cyclic shift by k positions.

    Permutation is the positional encoding of a strand. It distributes
    over bind, preserves bipolarity, and rho^-k inverts rho^k exactly.
    """
    return np.roll(v, k)


class Strand:
    """
    An ordered chain of bipolar codons.

    Codons are stored exactly; vector views are derived on demand. All
    codons must share one dimension.
    """

    def __init__(self, codons: Optional[Sequence[np.ndarray]] = None):
        self.codons: List[np.ndarray] = []
        for c in codons or []:
            self.append(c)

    def append(self, codon: np.ndarray) -> "Strand":
        if self.codons and codon.shape != self.codons[0].shape:
            raise ValueError(
                f"Dimension mismatch: strand has {self.codons[0].shape}, "
                f"codon has {codon.shape}"
            )
        self.codons.append(codon.astype(np.int8))
        return self

    def __len__(self) -> int:
        return len(self.codons)

    def state(self, decay: float = 0.7) -> np.ndarray:
        """
        Order-sensitive summary vector of the whole strand.

        Anchored at the tail: the newest codon enters unpermuted (age 0),
        the one before it as rho^1 with weight decay^1, and so on. Two
        strands that share a recent suffix are therefore similar even if
        their pasts differ — the property kNN over histories needs.
        """
        return np.where(self.state_accumulator(decay) >= 0, 1, -1).astype(np.int8)

    def state_accumulator(self, decay: float = 0.7) -> np.ndarray:
        """Real-valued pre-sign state; exposed for injection-style updates."""
        if not self.codons:
            raise ValueError("Cannot take the state of an empty strand")
        dim = self.codons[0].shape[0]
        acc = np.zeros(dim, dtype=np.float64)
        for age, codon in enumerate(reversed(self.codons)):
            acc += (decay ** age) * permute(codon, age)
        return acc

    def complement(self) -> "Strand":
        """The exact complement strand: every codon negated."""
        return Strand([(-c).astype(np.int8) for c in self.codons])

    def kmer_profile(self, k: int = 2) -> np.ndarray:
        """
        Order-insensitive bundle of locally-ordered k-windows.

        Each window binds its k codons under rho^0..rho^(k-1), so order
        matters within a window but not across windows — the BLAST-style
        seed index that makes strands searchable despite permutation.
        """
        if len(self.codons) < k:
            raise ValueError(f"Need at least {k} codons for a {k}-mer profile")
        windows = []
        for i in range(len(self.codons) - k + 1):
            w = self.codons[i].astype(np.int8)
            for j in range(1, k):
                w = w * permute(self.codons[i + j], j)
            windows.append(w.astype(np.int8))
        return bundle(windows)


def strand_state(codons: Sequence[np.ndarray], decay: float = 0.7) -> np.ndarray:
    """Convenience: state vector of a codon sequence without a Strand object."""
    return Strand(codons).state(decay)


def duplex(template: Strand, coding: Strand) -> List[np.ndarray]:
    """
    Pairwise-bind two strands into a double strand.

    Bind is self-inverse in bipolar space, so given the duplex and either
    strand, the other is exactly recoverable — base pairing as algebra.
    """
    if len(template) != len(coding):
        raise ValueError(
            f"Strand length mismatch: template {len(template)}, coding {len(coding)}"
        )
    return [
        (t * c).astype(np.int8)
        for t, c in zip(template.codons, coding.codons)
    ]


def transcribe(pairs: Sequence[np.ndarray], template: Strand) -> Strand:
    """Recover the coding strand from a duplex given the template strand."""
    if len(pairs) != len(template):
        raise ValueError(
            f"Length mismatch: duplex {len(pairs)}, template {len(template)}"
        )
    return Strand([
        (p * t).astype(np.int8)
        for p, t in zip(pairs, template.codons)
    ])


def splice(a: Strand, b: Strand, at_a: int, at_b: int) -> Strand:
    """
    Crossover: the first at_a codons of a followed by b's codons from at_b.

    The recombination operator — cheap novel candidate sequences from two
    parents, every one of them a well-formed strand.
    """
    if not 0 <= at_a <= len(a):
        raise ValueError(f"at_a {at_a} out of range for strand of length {len(a)}")
    if not 0 <= at_b <= len(b):
        raise ValueError(f"at_b {at_b} out of range for strand of length {len(b)}")
    return Strand(a.codons[:at_a] + b.codons[at_b:])


def kmer_profile(codons: Sequence[np.ndarray], k: int = 2) -> np.ndarray:
    """Convenience: k-mer profile of a codon sequence without a Strand object."""
    return Strand(codons).kmer_profile(k)
