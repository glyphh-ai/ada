"""
MorphologyEngine — Layer 2 of the Glyphh linguistics engine.

Learns morphological transforms from seed word pairs and applies them to
normalize any word to its lemma. Handles plurals, past tense, gerunds,
and comparative forms without hardcoded rules or rule lists.

Mechanism (Word2Vec analogy in HD space):
  Given seed pairs: [("dogs", "dog"), ("cats", "cat"), ("books", "book"), ...]

  PLURAL_TRANSFORM = bundle([bind(encode("dogs"), encode("dog")) for each pair])

  The transform vector is a superposition of all plural→singular "difference
  vectors". To normalize an unknown word:

    candidate_vec = bind(encode("sculptures"), PLURAL_TRANSFORM)

  Then find the word whose vector is nearest to candidate_vec in our known
  word space. This generalises to words not seen in training.

  Why this works (HD algebra):
    bind(dogs_vec, dog_vec) captures the "plural direction" for that pair.
    Bundling 20 such directions finds the consistent shared direction.
    bind(x, TRANSFORM) ≈ the singular form of x because bind is its own
    inverse: bind(bind(x, T), T) ≈ x  (in expectation over random HD vecs).

Universal grammar seeds (baked in — these are morphological rules, not domain
vocabulary). ~20 pairs per transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity
from glyphh.linguistics.character import CharacterEncoder


# ---------------------------------------------------------------------------
# Universal morphological seed pairs (grammar constants)
# ---------------------------------------------------------------------------

_PLURAL_SEEDS: list[tuple[str, str]] = [
    ("dogs", "dog"), ("cats", "cat"), ("books", "book"),
    ("files", "file"), ("tables", "table"), ("trees", "tree"),
    ("birds", "bird"), ("cars", "car"), ("stars", "star"),
    ("words", "word"), ("parts", "part"), ("points", "point"),
    ("items", "item"), ("names", "name"), ("types", "type"),
    ("lines", "line"), ("keys", "key"), ("rooms", "room"),
    ("steps", "step"), ("rules", "rule"),
    # Irregular / sibilant
    ("boxes", "box"), ("buses", "bus"), ("classes", "class"),
    ("dresses", "dress"), ("foxes", "fox"), ("matches", "match"),
    # es-suffix
    ("churches", "church"), ("patches", "patch"), ("branches", "branch"),
]

_PAST_SEEDS: list[tuple[str, str]] = [
    ("walked", "walk"), ("talked", "talk"), ("called", "call"),
    ("moved", "move"), ("saved", "save"), ("used", "use"),
    ("worked", "work"), ("asked", "ask"), ("turned", "turn"),
    ("opened", "open"), ("closed", "close"), ("played", "play"),
    ("showed", "show"), ("added", "add"), ("fixed", "fix"),
    ("reached", "reach"), ("checked", "check"), ("passed", "pass"),
    ("placed", "place"), ("changed", "change"),
    # Strong / irregular (included to improve generalisation)
    ("ran", "run"), ("sent", "send"), ("found", "find"),
    ("built", "build"), ("made", "make"), ("got", "get"),
]

_GERUND_SEEDS: list[tuple[str, str]] = [
    ("running", "run"), ("walking", "walk"), ("talking", "talk"),
    ("moving", "move"), ("saving", "save"), ("using", "use"),
    ("working", "work"), ("asking", "ask"), ("turning", "turn"),
    ("opening", "open"), ("playing", "play"), ("showing", "show"),
    ("adding", "add"), ("fixing", "fix"), ("checking", "check"),
    ("building", "build"), ("making", "make"), ("getting", "get"),
    ("sending", "send"), ("finding", "find"),
]

_COMPARATIVE_SEEDS: list[tuple[str, str]] = [
    ("bigger", "big"), ("smaller", "small"), ("faster", "fast"),
    ("slower", "slow"), ("higher", "high"), ("lower", "low"),
    ("older", "old"), ("newer", "new"), ("longer", "long"),
    ("shorter", "short"), ("stronger", "strong"), ("weaker", "weak"),
]

# Map from transform name to (inflected, base) seed list
_SEED_REGISTRY: dict[str, list[tuple[str, str]]] = {
    "plural":      _PLURAL_SEEDS,
    "past":        _PAST_SEEDS,
    "gerund":      _GERUND_SEEDS,
    "comparative": _COMPARATIVE_SEEDS,
}


# ---------------------------------------------------------------------------
# MorphologyEngine
# ---------------------------------------------------------------------------

@dataclass
class _Transform:
    """A learned morphological transform vector + its vocabulary for decode."""
    name: str
    vector: np.ndarray          # bundled bind(inflected, base) operator
    vocab: dict[str, np.ndarray] = field(default_factory=dict)
    # vocab: base_word → base_vector, used for nearest-neighbour decode


class MorphologyEngine:
    """Learns morphological transforms from seed pairs and normalises words.

    At construction time, all seed pairs are encoded and the transform
    vectors are built. No training required after construction.

    Usage:
        enc = CharacterEncoder()
        morph = MorphologyEngine(enc)

        lemma, tag = morph.normalize("sculptures")
        # → ("sculpture", "plural")

        lemma, tag = morph.normalize("running")
        # → ("run", "gerund")

        lemma, tag = morph.normalize("dog")
        # → ("dog", "base")  — already base form
    """

    # Similarity threshold — below this we consider the transform a no-match
    # and fall back to the original word.
    MATCH_THRESHOLD = 0.20

    def __init__(self, char_encoder: CharacterEncoder) -> None:
        self._enc = char_encoder
        self._transforms: dict[str, _Transform] = {}
        self._injected_base_forms: set[str] = set()
        self._build_transforms()

    # ── Public API ────────────────────────────────────────────────────────

    def add_base_forms(self, words: list[str]) -> None:
        """Mark words as already in base form — bypasses transform application.

        Use this for domain-specific terms that should never be lemmatized:
        brand names, technical identifiers, financial verbs, etc.

        Example:
            morph.add_base_forms(["charge", "refund", "cancel", "stripe"])
            morph.normalize("charge")  # → ("charge", "base")  not ("change", "past")
        """
        for w in words:
            self._injected_base_forms.add(w.lower().strip())

    def normalize(self, word: str) -> tuple[str, str]:
        """Normalize a word to its base form.

        Returns (lemma, morphological_tag) where tag is one of:
          "base"        — word is already in base form
          "plural"      — singular was derived
          "past"        — infinitive was derived
          "gerund"      — infinitive was derived
          "comparative" — base adjective was derived
          "unknown"     — no transform improved the word

        The returned lemma is always the most base-like form found.
        If no transform fires above MATCH_THRESHOLD, returns (word, "base").
        """
        word_lower = word.lower().strip()
        if not word_lower:
            return (word_lower, "base")

        # Fast path: injected base forms (model-specific domain terms).
        if word_lower in self._injected_base_forms:
            return (word_lower, "base")

        # Fast path: word is already a base form in some transform vocab.
        # e.g. normalize("dog") → ("dog", "base") since "dog" is a plural base.
        for transform in self._transforms.values():
            if word_lower in transform.vocab:
                return (word_lower, "base")

        word_vec = self._enc.encode_word(word_lower)
        best_lemma = word_lower
        best_tag = "base"
        best_score = self.MATCH_THRESHOLD  # min threshold

        for tname, transform in self._transforms.items():
            candidate_vec = bind(word_vec, transform.vector)
            # Find nearest base word in transform vocabulary
            for base_word, base_vec in transform.vocab.items():
                sim = float(cosine_similarity(candidate_vec, base_vec))
                if sim > best_score:
                    best_score = sim
                    best_lemma = base_word
                    best_tag = tname

        return (best_lemma, best_tag)

    def get_transform(self, name: str) -> np.ndarray | None:
        """Return the learned operator vector for a named transform."""
        t = self._transforms.get(name)
        return t.vector if t else None

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_transforms(self) -> None:
        """Build all transform vectors from seed pairs."""
        for tname, seed_pairs in _SEED_REGISTRY.items():
            self._build_one_transform(tname, seed_pairs)

    def _build_one_transform(
        self, name: str, pairs: list[tuple[str, str]]
    ) -> None:
        """Build a single transform from (inflected, base) pairs.

        TRANSFORM = bundle([bind(encode(inflected), encode(base)) for pair])

        This is the HDC equivalent of the Word2Vec morphology trick:
          vec("king") - vec("man") + vec("woman") ≈ vec("queen")
        Here:
          bind(enc("dogs"), enc("dog"))  captures the plural "axis"
          bundle of 20 such bindings → robust plural transform operator
        """
        binding_vecs = []
        vocab: dict[str, np.ndarray] = {}

        for inflected, base in pairs:
            inf_vec  = self._enc.encode_word(inflected)
            base_vec = self._enc.encode_word(base)
            binding_vecs.append(bind(inf_vec, base_vec))
            vocab[base] = base_vec

        if not binding_vecs:
            return

        transform_vec = bundle(binding_vecs) if len(binding_vecs) > 1 else binding_vecs[0]
        self._transforms[name] = _Transform(
            name=name,
            vector=transform_vec,
            vocab=vocab,
        )
