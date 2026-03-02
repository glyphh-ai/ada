"""
LinguisticIntentParser — Public API for the Glyphh linguistics engine.

Wraps all 5 layers (CharacterEncoder → MorphologyEngine → POSTagger →
SyntaxParser → HDCAttention) into a single interface.

Models use this as a replacement for flat hardcoded lookup tables. Instead
of `_VERB_MAP = {"get": "get", "fetch": "get", ...}`, a model provides
seed examples and the engine generalises:

    parser = LinguisticIntentParser()
    parser.seed_actions({
        "get":    ["get", "fetch", "retrieve", "find", "show", "check"],
        "create": ["create", "make", "build", "generate", "add"],
        "send":   ["send", "email", "mail", "post", "dispatch"],
    })
    parser.seed_targets({
        "file":    ["file", "document", "directory", "folder"],
        "message": ["message", "email", "notification", "post"],
        "user":    ["user", "person", "contact", "account"],
    })

    result = parser.extract_intent("fetch all documents from the folder")
    # → {"action": "get", "target": "file", "domain": "", "keywords": "..."}

    # Handles morphological variants automatically:
    result = parser.extract_intent("searching for sculptures in the museum")
    # → {"action": "get", "target": "artwork", "domain": "", ...}
    #   (because "sculptures" → "sculpture" via MorphologyEngine)

Pipeline:
  1. Tokenize query
  2. CharacterEncoder.encode_word() per token → word vectors
  3. MorphologyEngine.normalize() → lemma + morphological tag
  4. POSTagger.tag() → (word, POS) sequence
  5. SyntaxParser.parse() → {verb, object, modifiers, sentence_vec}
  6. HDCAttention.attend_roles() → weight each seeded action/target prototype
  7. Find best matching action/target by prototype similarity
  8. Extract keywords from content tokens
  → {"action", "target", "domain", "keywords"}
"""

from __future__ import annotations

import re

import numpy as np

from glyphh.core.ops import bundle, cosine_similarity
from glyphh.linguistics.attention import HDCAttention
from glyphh.linguistics.character import CharacterEncoder
from glyphh.linguistics.morphology import MorphologyEngine
from glyphh.linguistics.pos import POSTagger
from glyphh.linguistics.syntax import ParseResult, SyntaxParser


# ---------------------------------------------------------------------------
# LinguisticIntentParser
# ---------------------------------------------------------------------------

class LinguisticIntentParser:
    """End-to-end NL intent extraction using the 5-layer linguistics engine.

    No hardcoded vocabulary — models seed action/target/domain examples
    and the engine generalises via vector-space geometry. Handles plurals,
    conjugations, misspellings, and unseen words automatically.

    Usage:
        from glyphh.linguistics import LinguisticIntentParser

        parser = LinguisticIntentParser()
        parser.seed_actions({"get": ["fetch", "retrieve", "find", "show"]})
        parser.seed_targets({"file": ["document", "folder", "directory"]})
        parser.seed_domains({"storage": ["gdrive", "dropbox", "s3", "bucket"]})

        result = parser.extract_intent("find documents in my drive")
        # → {"action": "get", "target": "file", "domain": "storage",
        #    "keywords": "find documents drive"}
    """

    def __init__(self, dimension: int = 10000, seed: int = 42) -> None:
        self._dim  = dimension
        self._seed = seed

        # Layer stack
        self._char_enc  = CharacterEncoder(dimension, seed)
        self._morph     = MorphologyEngine(self._char_enc)
        self._pos       = POSTagger(self._char_enc, dimension)
        self._parser    = SyntaxParser(self._pos, self._char_enc, dimension, seed)
        self._attention = HDCAttention(dimension)

        # Action/target/domain prototypes: label → centroid vector
        self._action_protos:  dict[str, np.ndarray] = {}
        self._target_protos:  dict[str, np.ndarray] = {}
        self._domain_protos:  dict[str, np.ndarray] = {}
        self._domain_signals: dict[str, list[str]]  = {}

    # ── Seeding API ───────────────────────────────────────────────────────

    def seed_pos_hints(self, pos_hints: dict[str, list[str]]) -> None:
        """Inject domain-specific POS knowledge into the tagger.

        Each model can teach the engine its domain vocabulary so that
        domain-specific verbs (email, charge, notify, …) and nouns
        (stripe, jira, gdrive, …) are correctly classified.

        Args:
            pos_hints: {"VERB": ["email", "charge", "notify", ...],
                        "NOUN": ["stripe", "jira", "gdrive", ...]}

        This performs online Hebbian update on the POS centroid vectors —
        the injected words pull the prototype toward the domain vocabulary.

        Example:
            parser.seed_pos_hints({
                "VERB": ["email", "charge", "refund", "notify", "draft"],
                "NOUN": ["stripe", "jira", "slack", "gdrive"],
            })
        """
        for pos_label, words in pos_hints.items():
            for word in words:
                self._pos.learn(word.lower(), pos_label)

    def seed_base_forms(self, words: list[str]) -> None:
        """Mark words as morphological base forms — prevents false lemmatisation.

        The MorphologyEngine can misidentify domain-specific words as inflected
        forms of other words (e.g. "charge" → "change" via past-tense transform,
        "calendar" → "call" due to character similarity).  Injecting them as
        base forms short-circuits the transform pipeline.

        Args:
            words: Domain terms that should never be morphologically normalised.
                   E.g. ["charge", "refund", "cancel", "stripe", "jira"].

        Example:
            parser.seed_base_forms([
                "charge", "refund", "cancel", "subscribe",
                "stripe", "jira", "slack", "calendar",
            ])
        """
        self._morph.add_base_forms(words)

    def seed_actions(self, action_examples: dict[str, list[str]]) -> None:
        """Seed canonical action labels with example words.

        Args:
            action_examples: {"canonical_label": ["example1", "example2", ...]}
                E.g. {"get": ["fetch", "retrieve", "find", "show", "check"]}

        The centroid for each label is the bundle of encoded example words.
        At query time, the verb from SyntaxParser is compared to each centroid
        to find the best-matching canonical action.
        """
        for label, examples in action_examples.items():
            vecs = [self._char_enc.encode_word(w.lower()) for w in examples if w]
            if vecs:
                self._action_protos[label] = bundle(vecs) if len(vecs) > 1 else vecs[0]

    def seed_targets(self, target_examples: dict[str, list[str]]) -> None:
        """Seed canonical target labels with example words.

        Args:
            target_examples: {"canonical_label": ["example1", "example2", ...]}
                E.g. {"file": ["document", "folder", "directory", "attachment"]}
        """
        for label, examples in target_examples.items():
            vecs = [self._char_enc.encode_word(w.lower()) for w in examples if w]
            if vecs:
                self._target_protos[label] = bundle(vecs) if len(vecs) > 1 else vecs[0]

    def seed_domains(self, domain_signals: dict[str, list[str]]) -> None:
        """Seed domain labels with signal words/phrases.

        Args:
            domain_signals: {"domain_label": ["signal1", "signal2", ...]}
                E.g. {"payments": ["stripe", "invoice", "billing", "refund"]}

        Domain detection uses keyword scan (substring match) rather than
        vector similarity — domain signals tend to be proper nouns / brand
        names that are hard to generalise from.
        """
        self._domain_signals = {
            k: [s.lower() for s in signals]
            for k, signals in domain_signals.items()
        }
        for label, signals in domain_signals.items():
            vecs = [self._char_enc.encode_word(w.lower()) for w in signals if w]
            if vecs:
                self._domain_protos[label] = bundle(vecs) if len(vecs) > 1 else vecs[0]

    # ── Main API ──────────────────────────────────────────────────────────

    def extract_intent(self, query: str) -> dict:
        """Extract structured intent from a natural language query.

        Returns:
            {
                "action":   canonical action label (str) or "none"
                "target":   canonical target label (str) or "none"
                "domain":   domain label (str) or ""
                "keywords": space-separated content tokens (str)
            }
        """
        if not query or not query.strip():
            return {"action": "none", "target": "none", "domain": "", "keywords": ""}

        parse = self._parser.parse(query)
        action  = self._resolve_action(parse, query)
        target  = self._resolve_target(parse, query)
        domain  = self._detect_domain(query)
        keywords = self._extract_keywords(parse)

        return {
            "action":   action,
            "target":   target,
            "domain":   domain,
            "keywords": keywords,
        }

    def normalize(self, word: str) -> str:
        """Normalize a word to its base lemma form.

        Uses MorphologyEngine to strip plural, conjugation, comparative.
        """
        lemma, _tag = self._morph.normalize(word)
        return lemma

    def pos_tag(self, query: str) -> list[tuple[str, str]]:
        """POS-tag each word in the query.

        Returns list of (word, pos_label) tuples.
        """
        return [(w, pos) for w, pos, _ in self._pos.tag(query)]

    def parse(self, query: str) -> ParseResult:
        """Full parse: returns structured ParseResult with all role slots."""
        return self._parser.parse(query)

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_action(self, parse: ParseResult, query: str) -> str:
        """Find the best canonical action label for the parsed verb.

        Tries:
          1. Parsed verb (from SyntaxParser) after morphological normalisation
          2. First VERB token found in query tokens (fallback)
          3. All query tokens (wide net for function_name-style queries)

        Then compares to seeded action prototypes via cosine similarity.
        """
        if not self._action_protos:
            return "none"

        # Collect candidate verbs in priority order
        candidates: list[str] = []
        if parse.verb:
            lemma, _ = self._morph.normalize(parse.verb)
            candidates.append(lemma)
            candidates.append(parse.verb)

        # Also check all VERB-tagged tokens
        for word, pos, _ in parse.tokens:
            if pos == "VERB":
                lemma, _ = self._morph.normalize(word)
                candidates.append(lemma)
                candidates.append(word)

        if not candidates:
            return "none"

        best_label = "none"
        best_score = 0.0

        for cand in candidates:
            cand_vec = self._char_enc.encode_word(cand)
            for label, proto in self._action_protos.items():
                sim = float(cosine_similarity(cand_vec, proto))
                if sim > best_score:
                    best_score = sim
                    best_label = label

        return best_label if best_score > 0.0 else "none"

    def _resolve_target(self, parse: ParseResult, query: str) -> str:
        """Find the best canonical target label for the parsed object.

        Tries object slot first (highest priority), then all content tokens.
        Includes NOUN, NUM, ADJ, ADV candidates — POSTagger may misclassify
        open-class nouns as NUM/ADJ due to character similarity; trying all
        content tokens is more robust than relying on POS labels alone.
        Normalises via MorphologyEngine before comparison.
        """
        if not self._target_protos:
            return "none"

        # Closed-class POS tags that are never target candidates
        _SKIP_POS = {"PUNCT", "DET", "PREP", "CONJ", "MODAL", "VERB"}

        # Collect candidates: object slot first (highest priority)
        candidates: list[str] = []
        if parse.obj:
            lemma, _ = self._morph.normalize(parse.obj)
            candidates.append(lemma)
            candidates.append(parse.obj)

        # All content tokens (NOUN, ADJ, ADV, NUM) — cast wide net since
        # POSTagger may misclassify domain nouns as NUM or ADJ.
        for word, pos, _ in parse.tokens:
            if pos not in _SKIP_POS:
                lemma, _ = self._morph.normalize(word)
                candidates.append(lemma)
                candidates.append(word)

        if not candidates:
            return "none"

        best_label = "none"
        best_score = 0.0

        for cand in candidates:
            cand_vec = self._char_enc.encode_word(cand)
            for label, proto in self._target_protos.items():
                sim = float(cosine_similarity(cand_vec, proto))
                if sim > best_score:
                    best_score = sim
                    best_label = label

        return best_label if best_score > 0.0 else "none"

    def _detect_domain(self, query: str) -> str:
        """Detect domain from keyword signals in query text.

        Domain detection uses substring scan (not vector similarity) because
        domain signals are often proper nouns / brand names that benefit more
        from exact matching than from character-level generalisation.
        """
        lower = query.lower()
        for domain, signals in self._domain_signals.items():
            for sig in signals:
                if sig in lower:
                    return domain
        return ""

    def _extract_keywords(self, parse: ParseResult) -> str:
        """Extract content keywords from the parsed token list.

        Returns space-separated lemmatised content words, excluding
        closed-class words (DET, PREP, CONJ, MODAL, PUNCT).
        """
        content_pos = {"VERB", "NOUN", "ADJ", "ADV", "NUM"}
        words: list[str] = []
        seen: set[str] = set()

        for word, pos, _ in parse.tokens:
            if pos in content_pos:
                lemma, _ = self._morph.normalize(word)
                if lemma not in seen:
                    seen.add(lemma)
                    words.append(lemma)

        return " ".join(words) if words else ""
