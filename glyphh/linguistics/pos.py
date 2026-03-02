"""
POSTagger — Layer 3 of the Glyphh linguistics engine.

Classifies each word as VERB, NOUN, ADJ, ADV, DET, PREP, MODAL, NUM, PUNCT
using InductiveLayer prototype learning.

Approach:
  - Closed-class words (~150 total) are seeded directly with exact labels.
    These are grammar constants: determiners, modals, prepositions, conjunctions.
    They don't change across languages/domains and can be treated as axioms.

  - Open-class words (nouns, verbs, adjectives, adverbs) are seeded from ~50
    representative examples per class. The InductiveLayer builds a centroid
    (prototype) for each class. Unknown words are classified by cosine
    similarity to these prototypes — so "sculpture" → NOUN not because it's
    in the table, but because its character n-gram vector is near the NOUN
    centroid.

  - Online update: new labeled examples can be added via learn(), which
    updates the prototype via Hebbian bundling.

Labels:
  VERB    — action/state words: run, send, create, was, is, be
  NOUN    — entity/concept words: dog, file, user, message, city
  ADJ     — modifier words: big, fast, blue, important, statistical
  ADV     — manner/degree words: quickly, very, always, never, just
  DET     — determiners: the, a, an, this, that, these, those, my, your
  PREP    — prepositions: in, on, at, to, for, with, from, by, about
  MODAL   — modal verbs: can, could, would, should, will, shall, may, might
  CONJ    — conjunctions: and, or, but, nor, yet, so, although, because
  NUM     — numbers: 1, 42, 3.14, first, second, one, two
  PUNCT   — punctuation tokens
"""

from __future__ import annotations

import re

import numpy as np

from glyphh.core.ops import cosine_similarity
from glyphh.linguistics.character import CharacterEncoder


# ---------------------------------------------------------------------------
# Closed-class word sets (grammar axioms — ~150 words total)
# ---------------------------------------------------------------------------

_CLOSED_CLASS: dict[str, set[str]] = {
    "DET": {
        "the", "a", "an", "this", "that", "these", "those",
        "my", "your", "his", "her", "its", "our", "their",
        "some", "any", "all", "both", "each", "every",
        "no", "neither", "either", "much", "many", "few",
        "more", "most", "less", "least", "several",
    },
    "MODAL": {
        "can", "could", "will", "would", "shall", "should",
        "may", "might", "must", "ought", "need", "dare",
    },
    "PREP": {
        "in", "on", "at", "to", "for", "with", "from", "by",
        "about", "above", "across", "after", "against", "along",
        "among", "around", "before", "behind", "below", "beneath",
        "beside", "between", "beyond", "during", "except", "into",
        "near", "of", "off", "onto", "outside", "over", "past",
        "since", "through", "throughout", "under", "until", "up",
        "upon", "via", "within", "without",
    },
    "CONJ": {
        "and", "or", "but", "nor", "yet", "so", "for",
        "although", "because", "since", "unless", "until",
        "while", "whereas", "whether", "if", "though",
    },
}

# ---------------------------------------------------------------------------
# Open-class seed words (prototype seeds — not exhaustive lookup)
# ---------------------------------------------------------------------------

_OPEN_CLASS_SEEDS: dict[str, list[str]] = {
    "VERB": [
        "run", "send", "create", "get", "find", "update", "delete",
        "add", "remove", "fetch", "check", "build", "make", "start",
        "stop", "open", "close", "read", "write", "save", "load",
        "move", "copy", "search", "list", "calculate", "compute",
        "show", "display", "set", "reset", "parse", "generate",
        "convert", "format", "filter", "sort", "count", "merge",
        "split", "encode", "decode", "upload", "download", "track",
        "call", "return", "print", "log", "validate", "verify",
    ],
    "NOUN": [
        "file", "user", "message", "email", "task", "order", "item",
        "account", "server", "data", "record", "table", "index",
        "key", "value", "name", "type", "size", "path", "link",
        "image", "video", "text", "word", "number", "date", "time",
        "city", "country", "language", "system", "service", "tool",
        "method", "function", "class", "object", "array", "string",
        "list", "map", "set", "request", "response", "error", "dog",
        "cat", "book", "car", "tree", "house", "person", "team",
        # Extended vocabulary for broader generalisation
        "document", "folder", "directory", "attachment", "report",
        "notification", "channel", "contact", "member", "ticket",
        "invoice", "payment", "subscription", "project", "issue",
        "comment", "label", "category", "database", "collection",
        "resource", "endpoint", "parameter", "result", "event",
        "session", "token", "permission", "role", "group", "queue",
        "repository", "branch", "commit", "deployment", "container",
        "volume", "network", "policy", "metric", "alert", "dashboard",
    ],
    "ADJ": [
        "big", "small", "fast", "slow", "high", "low", "long", "short",
        "old", "new", "good", "bad", "true", "false", "active", "inactive",
        "public", "private", "secure", "open", "closed", "valid", "invalid",
        "primary", "secondary", "main", "default", "custom", "global",
        "local", "remote", "direct", "indirect", "full", "empty", "free",
        "busy", "ready", "done", "pending", "failed", "successful",
        "important", "critical", "optional", "required", "unique",
    ],
    "ADV": [
        "quickly", "slowly", "always", "never", "often", "rarely",
        "here", "there", "now", "then", "today", "yesterday", "soon",
        "already", "still", "again", "also", "only", "just", "very",
        "quite", "almost", "exactly", "clearly", "easily", "directly",
        "automatically", "manually", "simultaneously", "currently",
    ],
    "NUM": [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "hundred", "thousand", "million",
        "first", "second", "third", "fourth", "fifth",
        "zero", "half", "double", "triple",
    ],
}


# ---------------------------------------------------------------------------
# POSTagger
# ---------------------------------------------------------------------------

class POSTagger:
    """Assigns POS tags to words using InductiveLayer prototype classification.

    Closed-class words (DET, MODAL, PREP, CONJ) are always classified
    by direct lookup — they are grammar constants.

    Open-class words (VERB, NOUN, ADJ, ADV, NUM) are classified by
    cosine similarity between the word's CharacterEncoder vector and
    each POS prototype centroid. This generalises to unseen words.

    Usage:
        enc = CharacterEncoder()
        tagger = POSTagger(enc)
        results = tagger.tag("send the email quickly")
        # → [("send", "VERB", 0.85), ("the", "DET", 1.0),
        #    ("email", "NOUN", 0.71), ("quickly", "ADV", 0.68)]
    """

    def __init__(
        self,
        char_encoder: CharacterEncoder,
        dimension: int = 10000,
        min_confidence: float = 0.05,
    ) -> None:
        self._enc = char_encoder
        self._dim = dimension
        self._min_confidence = min_confidence

        # Build open-class centroids from seed words
        self._centroids: dict[str, np.ndarray] = {}
        self._build_centroids()

    # ── Public API ────────────────────────────────────────────────────────

    def tag(self, text: str) -> list[tuple[str, str, float]]:
        """Tag all tokens in a text string.

        Returns list of (token, pos_label, confidence).
        Confidence is 1.0 for closed-class (exact match), cosine score
        for open-class.
        """
        tokens = _tokenize(text)
        return [self._tag_token(tok) for tok in tokens]

    def tag_word(self, word: str) -> tuple[str, float]:
        """Tag a single word. Returns (pos_label, confidence)."""
        _, pos, conf = self._tag_token(word)
        return (pos, conf)

    def learn(self, word: str, pos: str) -> None:
        """Online update: add a labeled example to the POS centroid.

        This performs Hebbian bundling: new word vector is bundled into
        the existing centroid, pulling the prototype slightly toward
        the new example.
        """
        from glyphh.core.ops import bundle
        pos_upper = pos.upper()
        word_vec = self._enc.encode_word(word.lower())
        if pos_upper in self._centroids:
            self._centroids[pos_upper] = bundle([
                self._centroids[pos_upper], word_vec
            ])
        else:
            self._centroids[pos_upper] = word_vec

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_centroids(self) -> None:
        """Build prototype centroids from seed words using bundle()."""
        from glyphh.core.ops import bundle
        for pos_label, seed_words in _OPEN_CLASS_SEEDS.items():
            vecs = [self._enc.encode_word(w) for w in seed_words]
            self._centroids[pos_label] = bundle(vecs)

    def _tag_token(self, token: str) -> tuple[str, str, float]:
        """Tag a single token. Closed-class first, then prototype match."""
        lower = token.lower()

        # Punctuation
        if re.fullmatch(r"[^\w\s]+", token):
            return (token, "PUNCT", 1.0)

        # Numbers
        if re.fullmatch(r"\d+\.?\d*", token):
            return (token, "NUM", 1.0)

        # Closed-class lookup (O(1), grammar axioms)
        for pos_label, word_set in _CLOSED_CLASS.items():
            if lower in word_set:
                return (token, pos_label, 1.0)

        # Open-class: nearest prototype centroid
        if not self._centroids:
            return (token, "NOUN", 0.0)

        word_vec = self._enc.encode_word(lower)
        best_pos = "NOUN"
        best_sim = -1.0

        for pos_label, centroid in self._centroids.items():
            sim = float(cosine_similarity(word_vec, centroid))
            if sim > best_sim:
                best_sim = sim
                best_pos = pos_label

        confidence = max(0.0, best_sim)
        if confidence < self._min_confidence:
            # Below threshold — default to NOUN (most common open-class)
            return (token, "NOUN", 0.0)

        return (token, best_pos, round(confidence, 4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split text into tokens preserving punctuation as separate tokens."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens
