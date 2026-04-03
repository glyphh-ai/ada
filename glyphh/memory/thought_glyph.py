"""
ThoughtGlyph — Ada's cognitive encoding pipeline.

Every thought Ada absorbs becomes a full Glyph with 5 layers:
  perspective (WHO) → semantic (WHAT) → relational (HOW) →
  temporal (WHEN) → direction (SOURCE)

Uses the same SDK Encoder pipeline as pipedream/toolrouter.
The encoder takes text + speaker, matches words against primitive
roles, and fills the appropriate layers/segments/roles. The SDK
does the rest: bind role→value, bundle into segments, layers,
and global cortex.

Usage:
    from glyphh.memory.thought_glyph import ThoughtGlyphEncoder

    encoder = ThoughtGlyphEncoder()
    glyph = encoder.encode_thought("my name is chris", speaker="incoming")
    # → Glyph with perspective/self, semantic/identity,
    #   relational/equals, direction/incoming all activated
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any

from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.types import Concept, Glyph
from glyphh.encoder.base import Encoder

logger = logging.getLogger(__name__)

# ── Filler words — stripped before encoding ──────────────────────────────
#
# ONLY true articles. Everything else is either a primitive keyword
# (structural signal) or a content word (semantic signal). Previous
# filler set stripped 20+ primitive keywords: "not", "and", "or",
# "but", "in", "at", "to", "for", "do", "it", "so", "if", "just",
# etc. — killing structural differentiation.

_FILLER = frozenset({"a", "an", "the"})

# ── Contraction expansion ─────────────────────────────────────────────────
#
# Contractions hide primitive keywords: "don't" = "do" + "not" (action +
# negation), "can't" = "can" + "not" (ability + negation). Expand them
# so both structural signals fire.

_CONTRACTIONS: dict[str, list[str]] = {
    "don't":   ["do", "not"],
    "doesn't": ["does", "not"],
    "didn't":  ["did", "not"],
    "can't":   ["can", "not"],
    "won't":   ["will", "not"],
    "couldn't": ["could", "not"],
    "shouldn't": ["should", "not"],
    "wouldn't": ["would", "not"],
    "isn't":   ["is", "not"],
    "aren't":  ["are", "not"],
    "wasn't":  ["was", "not"],
    "weren't": ["were", "not"],
    "haven't": ["have", "not"],
    "hasn't":  ["has", "not"],
    "hadn't":  ["had", "not"],
    "i'm":     ["i", "am"],
    "i've":    ["i", "have"],
    "i'll":    ["i", "will"],
    "i'd":     ["i", "would"],
    "you're":  ["you", "are"],
    "you've":  ["you", "have"],
    "you'll":  ["you", "will"],
    "he's":    ["he", "is"],
    "she's":   ["she", "is"],
    "it's":    ["it", "is"],
    "we're":   ["we", "are"],
    "we've":   ["we", "have"],
    "we'll":   ["we", "will"],
    "they're": ["they", "are"],
    "they've": ["they", "have"],
    "they'll": ["they", "will"],
    "that's":  ["that", "is"],
    "there's": ["there", "is"],
    "what's":  ["what", "is"],
    "who's":   ["who", "is"],
    "where's": ["where", "is"],
    "let's":   ["let", "us"],
}

# ── Primitive role → (layer, segment) mapping ───────────────────────────
#
# This maps role names from exemplar JSONL to the glyph structure.
# When a word matches a primitive role, it activates the corresponding
# layer and segment. The attribute key is "{layer}_{segment}" — globally
# unique as required by EncoderConfig.

ROLE_TO_LAYER_SEGMENT: dict[str, tuple[str, str]] = {
    # ── Perspective — WHO ────────────────────────────────────
    "self":       ("perspective", "self"),
    "other":      ("perspective", "other"),
    "third":      ("perspective", "third"),
    "group":      ("perspective", "group"),
    "others":     ("perspective", "group"),

    # ── Semantic — WHAT ──────────────────────────────────────
    "identity":       ("semantic", "identity"),
    "quality":        ("semantic", "quality"),
    "quantity":       ("semantic", "quantity"),
    "ordinal":        ("semantic", "quantity"),
    "emotion":        ("semantic", "emotion"),
    "preference":     ("semantic", "emotion"),
    "affirmation":    ("semantic", "category"),
    "negation":       ("semantic", "category"),
    "absence":        ("semantic", "category"),
    "presence":       ("semantic", "category"),
    "totality":       ("semantic", "category"),
    "classification": ("semantic", "category"),
    "size":           ("semantic", "quality"),
    "speed":          ("semantic", "quality"),
    "temperature":    ("semantic", "quality"),
    "age":            ("semantic", "quality"),
    "difficulty":     ("semantic", "quality"),
    "language":       ("semantic", "category"),
    "abstract":       ("semantic", "category"),
    "topic":          ("semantic", "category"),

    # ── Relational — HOW ─────────────────────────────────────
    "equals":         ("relational", "equals"),
    "similarity":     ("relational", "comparison"),
    "difference":     ("relational", "comparison"),
    "comparison":     ("relational", "comparison"),
    "possession":     ("relational", "possession"),
    "composition":    ("relational", "possession"),
    "cause":          ("relational", "causation"),
    "effect":         ("relational", "causation"),
    "causation":      ("relational", "causation"),
    "condition":      ("relational", "causation"),
    "consequence":    ("relational", "causation"),
    "action":         ("relational", "action"),
    "ability":        ("relational", "action"),
    "desire":         ("relational", "action"),
    "obligation":     ("relational", "action"),
    "attempt":        ("relational", "action"),
    "begin":          ("relational", "action"),
    "end":            ("relational", "action"),
    "transfer":       ("relational", "action"),
    "knowledge":      ("relational", "action"),
    "communication":  ("relational", "action"),
    "perception":     ("relational", "action"),
    "purpose":        ("relational", "action"),
    "opposition":     ("relational", "comparison"),
    "accompaniment":  ("relational", "possession"),
    "location":       ("relational", "action"),
    "spatial":        ("relational", "action"),

    # ── Temporal — WHEN ──────────────────────────────────────
    "past":       ("temporal", "past"),
    "present":    ("temporal", "present"),
    "future":     ("temporal", "future"),
    "duration":   ("temporal", "present"),

    # ── Direction — SOURCE ───────────────────────────────────
    "direction":  ("direction", "incoming"),
    "incoming":   ("direction", "incoming"),
    "outgoing":   ("direction", "outgoing"),

    # ── Structural (activate but don't route to a segment) ───
    "question":       ("semantic", "category"),
    "intensifier":    ("semantic", "quality"),
    "modifier":       ("semantic", "quality"),
}

# All possible attribute keys (layer_segment combinations)
ALL_ATTR_KEYS = set()
for _layer, _segment in ROLE_TO_LAYER_SEGMENT.values():
    ALL_ATTR_KEYS.add(f"{_layer}_{_segment}")


# ── Encoder Config ───────────────────────────────────────────────────────
#
# 5 layers, each with segments containing a single bag_of_words role.
# Role names are qualified: "{layer}_{segment}" for global uniqueness.
# Dimension: 2048.

def _make_role(layer: str, segment: str, weight: float = 1.0) -> Role:
    """Create a bag-of-words role with qualified name."""
    return Role(
        name=f"{layer}_{segment}",
        similarity_weight=weight,
        text_encoding="bag_of_words",
    )


THOUGHT_ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="perspective",
            similarity_weight=0.25,
            segments=[
                Segment(name="self",  roles=[_make_role("perspective", "self")]),
                Segment(name="other", roles=[_make_role("perspective", "other")]),
                Segment(name="third", roles=[_make_role("perspective", "third")]),
                Segment(name="group", roles=[_make_role("perspective", "group")]),
            ],
        ),
        Layer(
            name="semantic",
            similarity_weight=0.30,
            segments=[
                Segment(name="identity", roles=[_make_role("semantic", "identity")]),
                Segment(name="quality",  roles=[_make_role("semantic", "quality")]),
                Segment(name="quantity", roles=[_make_role("semantic", "quantity")]),
                Segment(name="emotion",  roles=[_make_role("semantic", "emotion")]),
                Segment(name="category", roles=[_make_role("semantic", "category")]),
            ],
        ),
        Layer(
            name="relational",
            similarity_weight=0.25,
            segments=[
                Segment(name="equals",     roles=[_make_role("relational", "equals")]),
                Segment(name="possession", roles=[_make_role("relational", "possession")]),
                Segment(name="causation",  roles=[_make_role("relational", "causation")]),
                Segment(name="comparison", roles=[_make_role("relational", "comparison")]),
                Segment(name="action",     roles=[_make_role("relational", "action")]),
            ],
        ),
        Layer(
            name="temporal",
            similarity_weight=0.10,
            segments=[
                Segment(name="past",    roles=[_make_role("temporal", "past")]),
                Segment(name="present", roles=[_make_role("temporal", "present")]),
                Segment(name="future",  roles=[_make_role("temporal", "future")]),
            ],
        ),
        Layer(
            name="direction",
            similarity_weight=0.10,
            segments=[
                Segment(name="incoming", roles=[_make_role("direction", "incoming")]),
                Segment(name="outgoing", roles=[_make_role("direction", "outgoing")]),
            ],
        ),
    ],
)


# ── ThoughtGlyphEncoder ──────────────────────────────────────────────────

class ThoughtGlyphEncoder:
    """Encodes natural language into structured Thought Glyphs.

    The encoder takes raw text and a speaker tag, matches each word
    against primitive roles via PrimitiveSpace, and routes content words
    into the activated layers/segments. The SDK Encoder handles the HDC
    algebra (bind, bundle, cortex hierarchy).

    Matching uses the exemplar architecture:
      - Known words: exact multi-role lookup (fast, all roles activate)
      - Unknown words: HDC cosine match against role centroids
      - A word can activate MULTIPLE roles simultaneously
      - "who" → question + identity (both fire)

    Args:
        primitives: PrimitiveSpace instance. If None, creates an empty one.
        config: Optional override for the encoder config.
    """

    def __init__(
        self,
        primitives: "PrimitiveSpace | None" = None,
        config: EncoderConfig | None = None,
    ) -> None:
        self._config = config or THOUGHT_ENCODER_CONFIG
        self._encoder = Encoder(self._config)
        # Import here to avoid circular import at module level
        from glyphh.memory.primitives import PrimitiveSpace
        self._primitives: PrimitiveSpace = primitives or PrimitiveSpace()

    @property
    def space_id(self) -> str:
        return self._encoder.space_id

    @property
    def dimension(self) -> int:
        return self._encoder.dimension

    def set_primitives(self, primitives: "PrimitiveSpace") -> None:
        """Update the primitive space."""
        self._primitives = primitives

    def encode_thought(
        self,
        text: str,
        speaker: str = "incoming",
        metadata: dict[str, Any] | None = None,
    ) -> Glyph:
        """Encode a thought into a full Glyph.

        1. Tokenize and strip filler words
        2. Match each word against PrimitiveSpace → get ALL activated roles
           (known words: exact multi-role, unknown: HDC cosine match)
        3. Activated roles determine which layers/segments fire
        4. Primitives contribute ROLE NAMES, content words fill activated segments
        5. Direction segment set by speaker param
        6. SDK Encoder builds the glyph hierarchy

        Args:
            text: Raw text to encode.
            speaker: "incoming" (user) or "outgoing" (Ada).
            metadata: Optional metadata to attach to the glyph.

        Returns:
            A full Glyph with hierarchical layer/segment/role structure.
        """
        words = self._tokenize(text)
        if not words:
            words = [text.lower().strip()]

        # Classify words: primitive (structural) or content (semantic).
        # Multi-role: "who" → question + identity. All fire their segments.
        #
        # Segment encoding: role name + content words (for glyph hierarchy).
        # Content words stored separately in metadata for direct matching
        # in recall — avoids role name dilution in cosine similarity.
        segment_roles: dict[str, str] = {}  # attr_key → single role name
        content_words: list[str] = []

        for word in words:
            matched_roles = self._primitives.match_roles(word)
            if matched_roles:
                for role, _score in matched_roles:
                    if role in ROLE_TO_LAYER_SEGMENT:
                        layer, segment = ROLE_TO_LAYER_SEGMENT[role]
                        attr_key = f"{layer}_{segment}"
                        if attr_key not in segment_roles:
                            segment_roles[attr_key] = role
            else:
                content_words.append(word)

        # Each segment: role name + content words
        attributes: dict[str, str] = {}
        for attr_key, role_name in segment_roles.items():
            all_words = [role_name] + content_words
            attributes[attr_key] = " ".join(all_words)

        # If no primitives matched, put content into default segment
        if not segment_roles:
            content_str = " ".join(content_words) if content_words else " ".join(words)
            attributes["semantic_category"] = content_str

        # Direction — with content words
        dir_key = f"direction_{speaker}"
        dir_value = " ".join(content_words) if content_words else " ".join(words)
        if dir_key not in attributes:
            attributes[dir_key] = dir_value

        # Build Concept and encode.
        # Track activated attrs in metadata so recall can filter
        # non-activated segments (SDK creates vectors for all configured
        # segments, but only activated ones carry signal).
        stable_id = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        meta = metadata.copy() if metadata else {}
        meta["_activated_attrs"] = sorted(attributes.keys())
        meta["_content_words"] = content_words

        concept = Concept(
            name=f"thought_{stable_id:08d}",
            attributes=attributes,
            metadata=meta,
        )

        glyph = self._encoder.encode(concept)
        return glyph

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text: expand contractions, strip filler (articles only)."""
        words = []
        for w in text.lower().split():
            w = w.strip("?.,!;:'\"()-")
            if not w or w in _FILLER:
                continue
            # Expand contractions: "don't" → ["do", "not"]
            if w in _CONTRACTIONS:
                words.extend(_CONTRACTIONS[w])
            else:
                words.append(w)
        return words
