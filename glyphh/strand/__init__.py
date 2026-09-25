"""
Strand — DNA-shaped sequence encoding for the Glyphh HDC substrate.

Where the ThoughtGlyph lattice bundles structure (unordered, similarity-
preserving), a Strand chains it (ordered, exact). The two are duals:
the strand is the genotype — a durable, position-exact record built by
permute-and-bind; the lattice-style k-mer profile is the phenotype — a
fuzzy, searchable expression of the same sequence.

Four operations carry the biology, and each is real algebra in bipolar
space:

- ordered chains:   position via permutation (np.roll), so order is
                    structural rather than annotated
- exact complement: -v is a perfect inverse; bind is self-inverse
- duplex/transcribe: pairwise binding of a template strand and a coding
                    strand; either recovers the other exactly
- recombination:    splice/crossover of two strands yields novel,
                    well-formed candidate sequences
"""

from .strand import (
    Strand,
    permute,
    duplex,
    transcribe,
    splice,
    kmer_profile,
    strand_state,
)

__all__ = [
    "Strand",
    "permute",
    "duplex",
    "transcribe",
    "splice",
    "kmer_profile",
    "strand_state",
]
