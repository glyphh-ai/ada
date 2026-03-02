"""
SyntaxParser — Layer 4 of the Glyphh linguistics engine.

Extracts sentence structure using HRR (Holographic Reduced Representation)
role binding. Produces a sentence vector that encodes VERB, SUBJECT, OBJECT,
and MODIFIER roles — each slot is decodable by unbinding (bind is its own
inverse in HD space).

Mechanism:
  - Pre-generate role symbols (deterministic from seed):
      VERB_ROLE = generate_symbol(seed, "role_verb")
      OBJ_ROLE  = generate_symbol(seed, "role_object")
      SUBJ_ROLE = generate_symbol(seed, "role_subject")
      MOD_ROLE  = generate_symbol(seed, "role_modifier")

  - After POS tagging, assign each word to a slot based on POS + position:
      VERB tokens → verb slot
      First NOUN after verb → object slot
      First NOUN before verb → subject slot
      ADJ tokens → modifier slot (bound to their noun)

  - Encode sentence as HRR superposition:
      sentence_vec = bundle([
          bind(VERB_ROLE, verb_vec),
          bind(OBJ_ROLE,  obj_vec),
          bind(SUBJ_ROLE, subj_vec),
          bind(MOD_ROLE,  mod_vec),
      ])

  - Decode: verb ≈ bind(sentence_vec, VERB_ROLE)
    (approximate — works well in HD dimensions)

  - PathwayEncoder handles full positional sequence with decay for
    cases where order matters (multi-step commands, chained actions).

ParseResult fields:
  verb       — primary action word (str)
  subject    — who/what initiates (str)
  obj        — what is acted upon (str)
  modifiers  — list of (adjective, noun) pairs
  sentence_vec — full HRR-encoded sentence vector
  pathway_vec  — PathwayEncoder positional sequence vector
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol
from glyphh.linguistics.character import CharacterEncoder
from glyphh.linguistics.pos import POSTagger


# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Result of parsing a sentence into structural roles."""

    verb: str = ""            # Primary action word (lemma)
    subject: str = ""         # Subject entity
    obj: str = ""             # Object entity (what is being acted on)
    modifiers: list[tuple[str, str]] = field(default_factory=list)
    # modifiers: list of (adjective, head_noun) pairs

    sentence_vec: np.ndarray | None = None   # HRR superposition
    pathway_vec:  np.ndarray | None = None   # PathwayEncoder sequence

    # Raw POS-tagged tokens for downstream use
    tokens: list[tuple[str, str, float]] = field(default_factory=list)
    # tokens: [(word, pos, confidence), ...]


# ---------------------------------------------------------------------------
# SyntaxParser
# ---------------------------------------------------------------------------

_ROLE_NAMES = ("verb", "subject", "object", "modifier")


class SyntaxParser:
    """Parses NL text into structured ParseResult using HRR role binding.

    Uses POSTagger (Layer 3) for word classification, then assigns words
    to VERB/SUBJECT/OBJECT/MODIFIER roles based on POS + positional heuristics.

    Usage:
        enc = CharacterEncoder()
        tagger = POSTagger(enc)
        parser = SyntaxParser(tagger, enc)

        result = parser.parse("send the monthly report to Alice")
        result.verb      # "send"
        result.obj       # "report"
        result.modifiers # [("monthly", "report")]
        result.sentence_vec  # shape (10000,) HRR vector
    """

    def __init__(
        self,
        pos_tagger: POSTagger,
        char_encoder: CharacterEncoder,
        dimension: int = 10000,
        seed: int = 42,
        pathway_decay: float = 0.75,
    ) -> None:
        self._tagger = pos_tagger
        self._enc    = char_encoder
        self._dim    = dimension
        self._seed   = seed
        self._decay  = pathway_decay

        # Pre-generate role symbols (deterministic)
        self._role_vecs = {
            role: generate_symbol(seed, f"role_{role}", dimension)
            for role in _ROLE_NAMES
        }

    # ── Public API ────────────────────────────────────────────────────────

    def parse(self, text: str) -> ParseResult:
        """Parse text into a structured ParseResult.

        POS-tags each word, assigns roles, encodes as HRR sentence vector
        and PathwayEncoder sequence vector.
        """
        if not text or not text.strip():
            return ParseResult()

        tokens = self._tagger.tag(text)
        result = self._assign_roles(tokens)
        result.tokens = tokens
        result.sentence_vec = self._encode_hrr(result)
        result.pathway_vec  = self._encode_pathway(tokens)
        return result

    def decode_role(self, sentence_vec: np.ndarray, role: str) -> np.ndarray:
        """Decode a role slot from a sentence vector.

        Returns a candidate vector — find the nearest word in CharacterEncoder
        space to identify what word fills this role.

        Args:
            sentence_vec: HRR sentence vector from ParseResult.sentence_vec
            role: one of "verb", "subject", "object", "modifier"
        """
        if role not in self._role_vecs:
            raise ValueError(f"Unknown role '{role}'. Valid: {list(self._role_vecs)}")
        return bind(sentence_vec, self._role_vecs[role])

    # ── Role assignment ───────────────────────────────────────────────────

    def _assign_roles(
        self, tokens: list[tuple[str, str, float]]
    ) -> ParseResult:
        """Assign VERB/SUBJECT/OBJECT/MODIFIER roles from POS-tagged tokens.

        Simple left-to-right heuristic:
          1. First VERB → main verb
          2. NOUN before first VERB → subject
          3. First NOUN after VERB → direct object
          4. ADJ immediately before a NOUN → (adjective, noun) modifier pair

        This captures the dominant English SVO sentence pattern well for
        short imperative/interrogative NL queries.
        """
        result = ParseResult()

        verb_idx = -1
        last_adj  = ""

        # Pass 1: find verb
        for i, (word, pos, _conf) in enumerate(tokens):
            if pos == "VERB" and not result.verb:
                result.verb = word
                verb_idx = i

        # Pass 2: subject, object, modifiers
        for i, (word, pos, _conf) in enumerate(tokens):
            if pos == "ADJ":
                last_adj = word
                continue

            if pos == "NOUN":
                if verb_idx < 0 or i < verb_idx:
                    # Noun before verb → subject
                    if not result.subject:
                        result.subject = word
                else:
                    # Noun after verb → object
                    if not result.obj:
                        result.obj = word

                if last_adj:
                    result.modifiers.append((last_adj, word))

            # Reset adj tracker when we hit non-adj non-noun
            if pos not in ("ADJ", "NOUN", "DET"):
                last_adj = ""

        return result

    # ── HRR encoding ──────────────────────────────────────────────────────

    def _encode_hrr(self, result: ParseResult) -> np.ndarray:
        """Encode ParseResult as an HRR sentence vector.

        sentence = bundle([
            bind(VERB_ROLE,  verb_vec),
            bind(OBJ_ROLE,   obj_vec),
            bind(SUBJ_ROLE,  subj_vec),
            bind(MOD_ROLE,   mod_bundle),
        ])

        Only slots with content are included in the bundle.
        """
        bindings: list[np.ndarray] = []

        if result.verb:
            bindings.append(bind(
                self._role_vecs["verb"],
                self._enc.encode_word(result.verb),
            ))

        if result.obj:
            bindings.append(bind(
                self._role_vecs["object"],
                self._enc.encode_word(result.obj),
            ))

        if result.subject:
            bindings.append(bind(
                self._role_vecs["subject"],
                self._enc.encode_word(result.subject),
            ))

        if result.modifiers:
            mod_vecs = [
                self._enc.encode_word(adj)
                for adj, _noun in result.modifiers
            ]
            mod_bundle = bundle(mod_vecs) if len(mod_vecs) > 1 else mod_vecs[0]
            bindings.append(bind(self._role_vecs["modifier"], mod_bundle))

        if not bindings:
            return generate_symbol(self._seed, "empty_sentence", self._dim)

        return bundle(bindings) if len(bindings) > 1 else bindings[0]

    # ── PathwayEncoder sequence encoding ──────────────────────────────────

    def _encode_pathway(
        self, tokens: list[tuple[str, str, float]]
    ) -> np.ndarray:
        """Encode the full token sequence using positional binding + decay.

        Each token at position i is bound to its position symbol, then
        all are combined with exponential decay (recent tokens dominate).

        position_vec(i) = generate_symbol(seed, f"pos_{i}")
        step(i)         = bind(position_vec(i), encode_word(token))
        pathway         = weighted_bundle([(step(i), decay^(n-i-1)) for i])
        """
        content_tokens = [
            (w, pos) for w, pos, _ in tokens
            if pos not in ("PUNCT", "DET", "PREP", "CONJ")
        ]

        if not content_tokens:
            return generate_symbol(self._seed, "empty_pathway", self._dim)

        n = len(content_tokens)
        pairs: list[tuple[np.ndarray, float]] = []

        for i, (word, _pos) in enumerate(content_tokens):
            pos_sym  = generate_symbol(self._seed, f"pos_{i}", self._dim)
            word_vec = self._enc.encode_word(word)
            step_vec = bind(pos_sym, word_vec)
            weight   = self._decay ** (n - i - 1)   # recent steps higher weight
            pairs.append((step_vec, weight))

        # Weighted bundle (re-implement here to avoid circular import)
        total = np.zeros(self._dim, dtype=np.float32)
        for vec, w in pairs:
            total += vec.astype(np.float32) * w
        return np.where(total >= 0, 1, -1).astype(np.int8)
