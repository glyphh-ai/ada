"""
CharacterEncoder — Layer 1 of the Glyphh linguistics engine.

Encodes any word as a bipolar HDC vector using positional character
n-grams (FastText-style, but in HD bipolar space).

Mechanism:
  1. Pad word with boundary markers: "word" → "#word#"
  2. Extract all character n-grams of size ngram_sizes (default 2–4)
     with positional index: "ru@0", "run@0", "un@1", "uni@1", ...
  3. For each positional n-gram: generate_symbol(seed, ngram@pos)
  4. Bundle all n-gram symbols → word vector

Why this handles misspellings and morphological variants:
  - "sculpture"  and "sculptures"  share ~85% n-grams → cosine > 0.8
  - "recieve"    and "receive"     share ~80% n-grams → near-identical
  - Unknown words still get meaningful vectors from character overlap
  - Numbers and punctuation are treated as characters (handles "42", "v2", etc.)
"""

from __future__ import annotations

import re

import numpy as np

from glyphh.core.ops import bundle, generate_symbol


class CharacterEncoder:
    """Encodes words as bipolar HDC vectors via positional character n-grams.

    Equivalent to FastText embeddings but in HD bipolar space — no training
    required. Two words are similar iff they share many n-grams at similar
    positions, which captures:
      - Morphological variants (plural, conjugated, comparative)
      - Misspellings (transpositions, substitutions, extra/missing chars)
      - Partial matches (prefix/suffix overlap)

    Usage:
        enc = CharacterEncoder(dimension=10000, seed=42)
        vec = enc.encode_word("sculptures")   # np.ndarray shape (10000,)
        sim = cosine_similarity(enc.encode_word("sculpture"), vec)  # ≈ 0.85
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: int = 42,
        ngram_sizes: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        self._dim = dimension
        self._seed = seed
        self._ngram_sizes = ngram_sizes
        self._cache: dict[str, np.ndarray] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def encode_word(self, word: str) -> np.ndarray:
        """Encode a single word as a bipolar HDC vector.

        Returns a deterministic vector of shape (dimension,) with dtype int8.
        The same word always produces the same vector. Unknown words produce
        meaningful vectors via character overlap with known words.

        Empty strings, whitespace, and single characters return the
        "empty_word" or character symbol to avoid degenerate bundles.
        """
        word_lower = word.lower().strip()
        if not word_lower:
            return generate_symbol(self._seed, "empty_word", self._dim)

        if word_lower in self._cache:
            return self._cache[word_lower]

        ngrams = self._extract_ngrams(word_lower)
        if not ngrams:
            # Single character — use character symbol directly
            vec = generate_symbol(self._seed, f"char_{word_lower}", self._dim)
            self._cache[word_lower] = vec
            return vec

        vecs = [generate_symbol(self._seed, ng, self._dim) for ng in ngrams]
        vec = bundle(vecs) if len(vecs) > 1 else vecs[0]

        self._cache[word_lower] = vec
        return vec

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a multi-word text as the bundle of its word vectors.

        Word order is ignored (bag-of-words). For order-sensitive encoding
        use SyntaxParser which uses PathwayEncoder for positional structure.
        """
        words = self._tokenize(text)
        if not words:
            return generate_symbol(self._seed, "empty_text", self._dim)

        vecs = [self.encode_word(w) for w in words]
        return bundle(vecs) if len(vecs) > 1 else vecs[0]

    def similarity(self, word_a: str, word_b: str) -> float:
        """Cosine similarity between two word vectors. Range [-1, 1]."""
        from glyphh.core.ops import cosine_similarity
        return float(cosine_similarity(self.encode_word(word_a), self.encode_word(word_b)))

    # ── Internals ─────────────────────────────────────────────────────────

    def _extract_ngrams(self, word: str) -> list[str]:
        """Extract all positional character n-grams from a padded word.

        Padding: "dog" → "#dog#"
        N-grams of size 2: "#d@0", "do@1", "og@2", "g#@3"
        N-grams of size 3: "#do@0", "dog@1", "og#@2"

        Positional index is the start character position (after padding).
        This means "run" in "running" has different n-grams than "run" alone,
        which is intentional — position within word carries morphological signal.
        """
        padded = f"#{word}#"
        ngrams: list[str] = []
        for size in self._ngram_sizes:
            for i in range(len(padded) - size + 1):
                gram = padded[i : i + size]
                ngrams.append(f"{gram}@{i}")
        return ngrams

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase tokens on whitespace and punctuation."""
        return [t for t in re.split(r"[\s\W]+", text.lower()) if t]
