"""
PrimitiveSpace — Ada's innate wiring.

Primitives are permanent exemplar Glyphs — the building blocks a child
learns that never change. They define the coordinate axes of the thought
manifold. Loaded from standard exemplar JSONL at boot, held in-memory
forever.

Same format as every glyphh model's exemplars.jsonl:
  {"record_type": "primitive", "role": "self", "layer": "perspective",
   "segment": "self", "question": "i am talking about myself",
   "keywords": ["i", "me", "my", "myself", "mine"], ...}

Two matching modes:
  1. Exact lookup: word → list[role] from keywords (fast, precise)
  2. Exemplar Glyph similarity: encode phrase → FIND SIMILAR against
     exemplar Glyphs → matched role (handles unknown words, phrases)

The exemplar architecture IS the glyphh model pattern:
  - Each role has an exemplar Glyph (encoded from the question field)
  - The exemplar's metadata carries role, layer, segment
  - Matching = full hierarchy similarity against exemplar Glyphs
  - This is the same path as GQL FIND SIMILAR in any model

Usage:
    prims = PrimitiveSpace()
    prims.load()   # loads JSONL + builds exemplar Glyphs

    # Exact match (known word)
    roles = prims.match_roles("myself")   # → [("self", 1.0)]

    # Exemplar Glyph match (unknown phrase)
    roles = prims.match_roles("whomever") # → [("question", 0.35), ...]

    # Pass to encoder
    encoder = ThoughtGlyphEncoder(primitives=prims)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Glyph
from glyphh.linguistics.character import CharacterEncoder
from glyphh.memory.thought_glyph import ROLE_TO_LAYER_SEGMENT

if TYPE_CHECKING:
    from glyphh.memory.thought_glyph import ThoughtGlyphEncoder

logger = logging.getLogger(__name__)

_LESSONS_DIR = Path(__file__).parent / "lessons"

# Cosine threshold for exemplar Glyph matching.
_MATCH_THRESHOLD = 0.30

# CharacterEncoder for unknown word fallback (morphological matching).
_CHAR_DIM = 2000


class PrimitiveExemplar:
    """A single primitive exemplar — a role with its Glyph and metadata."""
    __slots__ = ("role", "layer", "segment", "glyph", "keywords", "question", "description")

    def __init__(
        self,
        role: str,
        layer: str,
        segment: str,
        glyph: Glyph | None,
        keywords: list[str],
        question: str,
        description: str = "",
    ) -> None:
        self.role = role
        self.layer = layer
        self.segment = segment
        self.glyph = glyph
        self.keywords = keywords
        self.question = question
        self.description = description


class PrimitiveSpace:
    """Ada's innate concept vocabulary — permanent, immutable.

    Loads primitive exemplars from standard JSONL format (same as every
    glyphh model), then builds exemplar Glyphs using ThoughtGlyphEncoder.

    Bootstrap sequence:
      1. Load JSONL → extract word→role mappings from keywords
      2. Create ThoughtGlyphEncoder with flat word→role dict
      3. Encode each exemplar's question → exemplar Glyph
      4. Store exemplar Glyphs with metadata (role, layer, segment)

    Matching:
      - Known words: exact multi-role lookup from keywords (fast path)
      - Unknown words: CharacterEncoder centroid fallback (morphological)
      - Phrase matching: encode phrase → similarity against exemplar Glyphs

    Attributes:
        word_to_roles: Maps lowercase word → list of role names.
        role_to_words: Maps role name → set of words.
        exemplars: List of PrimitiveExemplar with Glyphs.
    """

    # Files that define Ada's innate primitive wiring.
    PRIMITIVE_FILES = (
        "00_primitives.jsonl",
    )

    def __init__(self) -> None:
        self._word_to_roles: dict[str, list[str]] = {}
        self._role_to_words: dict[str, set[str]] = {}
        self._exemplars: list[PrimitiveExemplar] = []
        self._role_to_exemplar: dict[str, PrimitiveExemplar] = {}
        self._facts: list[tuple[str, str, str]] = []
        self._loaded = False
        self._version: int = 0  # incremented on add_compound

        # CharacterEncoder for unknown word fallback
        self._char_encoder = CharacterEncoder(dimension=_CHAR_DIM, seed=42)
        self._role_centroids: dict[str, np.ndarray] = {}

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def word_to_role(self) -> dict[str, str]:
        """Word → first role mapping (backward compat)."""
        return {w: roles[0] for w, roles in self._word_to_roles.items() if roles}

    @property
    def word_to_roles(self) -> dict[str, list[str]]:
        """Word → all roles mapping."""
        return self._word_to_roles

    @property
    def role_to_words(self) -> dict[str, set[str]]:
        return self._role_to_words

    @property
    def exemplars(self) -> list[PrimitiveExemplar]:
        """All primitive exemplars with Glyphs."""
        return self._exemplars

    @property
    def role_centroids(self) -> dict[str, np.ndarray]:
        """Role → CharacterEncoder centroid (for unknown word fallback)."""
        return self._role_centroids

    @property
    def facts(self) -> list[tuple[str, str, str]]:
        return self._facts

    @property
    def count(self) -> int:
        return len(self._word_to_roles)

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ── Matching ──────────────────────────────────────────────────────────

    def match(self, word: str) -> str | None:
        """Look up a word's first primitive role. Returns None if not a primitive."""
        roles = self._word_to_roles.get(word.lower().strip())
        return roles[0] if roles else None

    def match_all(self, word: str) -> list[str]:
        """Look up ALL roles for a word (exact match). Returns [] if not a primitive."""
        return list(self._word_to_roles.get(word.lower().strip(), []))

    def match_roles(
        self,
        word: str,
        threshold: float = _MATCH_THRESHOLD,
    ) -> list[tuple[str, float]]:
        """Match a word to primitive roles — exact first, centroid fallback.

        For known words: returns all roles from keywords with score 1.0.
        For unknown words (≥3 chars): CharacterEncoder centroid matching.

        Args:
            word: The word to match.
            threshold: Minimum cosine similarity for centroid match.

        Returns:
            List of (role_name, score) tuples, sorted by score descending.
        """
        word_lower = word.lower().strip()
        if not word_lower:
            return []

        # Fast path: exact lookup
        exact_roles = self._word_to_roles.get(word_lower)
        if exact_roles:
            return [(role, 1.0) for role in exact_roles]

        # Centroid fallback for unknown words (≥3 chars)
        if not self._role_centroids or len(word_lower) <= 2:
            return []

        word_vec = self._char_encoder.encode_word(word_lower)
        matches: list[tuple[str, float]] = []
        for role, centroid in self._role_centroids.items():
            sim = float(cosine_similarity(word_vec, centroid))
            if sim >= threshold:
                matches.append((role, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def find_similar(
        self,
        query_glyph: Glyph,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[tuple[PrimitiveExemplar, float]]:
        """Find similar exemplar Glyphs — the model pattern.

        Full hierarchy similarity against exemplar Glyphs, same as
        GQL FIND SIMILAR in any glyphh model. Returns exemplars with
        metadata (role, layer, segment) — the path back.

        Args:
            query_glyph: Encoded query/thought Glyph.
            top_k: Maximum results.
            threshold: Minimum similarity.

        Returns:
            List of (exemplar, similarity) sorted by score descending.
        """
        if not self._exemplars:
            return []

        results: list[tuple[PrimitiveExemplar, float]] = []
        query_cortex = query_glyph.global_cortex.data

        for exemplar in self._exemplars:
            if exemplar.glyph is None:
                continue

            # Full hierarchy: role-level weighted similarity
            sim = self._hierarchy_similarity(query_glyph, exemplar.glyph)
            if sim >= threshold:
                results.append((exemplar, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _hierarchy_similarity(self, a: Glyph, b: Glyph) -> float:
        """Compute full hierarchy similarity between two Glyphs.

        Same pattern as pipedream: iterate layers → segments → roles,
        cosine at each role, weighted by layer importance.
        """
        _LAYER_WEIGHTS = {
            "perspective": 0.25, "semantic": 0.30, "relational": 0.25,
            "temporal": 0.10, "direction": 0.10,
        }

        # Only compare activated segments
        a_activated = set(a.metadata.get("_activated_attrs", []))
        b_activated = set(b.metadata.get("_activated_attrs", []))

        total_sim = 0.0
        total_weight = 0.0

        for layer_name, a_layer in a.layers.items():
            if layer_name not in b.layers:
                continue
            b_layer = b.layers[layer_name]
            lw = _LAYER_WEIGHTS.get(layer_name, 0.1)

            for seg_name, a_seg in a_layer.segments.items():
                attr_key = f"{layer_name}_{seg_name}"
                if attr_key not in a_activated or attr_key not in b_activated:
                    continue
                if seg_name not in b_layer.segments:
                    continue
                b_seg = b_layer.segments[seg_name]

                for role_name, a_role in a_seg.roles.items():
                    if role_name not in b_seg.roles:
                        continue
                    rsim = float(cosine_similarity(a_role.data, b_seg.roles[role_name].data))
                    total_sim += rsim * lw
                    total_weight += lw

        return total_sim / total_weight if total_weight > 0 else 0.0

    def route(self, role: str) -> tuple[str, str] | None:
        """Map a role to (layer, segment). Returns None if unmapped."""
        return ROLE_TO_LAYER_SEGMENT.get(role)

    def is_primitive(self, word: str) -> bool:
        """Check if a word is a known primitive."""
        return word.lower().strip() in self._word_to_roles

    def has_role(self, role: str) -> bool:
        """Check if a role exists in ROLE_TO_LAYER_SEGMENT."""
        return role in ROLE_TO_LAYER_SEGMENT

    def get_exemplar(self, role: str) -> PrimitiveExemplar | None:
        """Get the exemplar Glyph for a role."""
        return self._role_to_exemplar.get(role)

    # ── Loading ───────────────────────────────────────────────────────────

    def load(
        self,
        directory: str | Path | None = None,
        files: tuple[str, ...] | None = None,
    ) -> None:
        """Load primitive exemplars from JSONL files.

        Bootstrap sequence:
          1. Load JSONL → word→role mappings + exemplar metadata
          2. Build CharacterEncoder centroids (for unknown word fallback)
          3. Build exemplar Glyphs (requires ThoughtGlyphEncoder)

        Args:
            directory: Override path to lessons directory.
            files: Override which files to load.
        """
        lessons_dir = Path(directory) if directory else _LESSONS_DIR
        if not lessons_dir.exists():
            logger.warning("Lessons directory not found: %s", lessons_dir)
            return

        file_names = files or self.PRIMITIVE_FILES
        for name in file_names:
            path = lessons_dir / name
            if path.exists():
                self._load_jsonl(path)
            else:
                logger.warning("Primitive file not found: %s", path)

        self._build_centroids()
        self._build_exemplar_glyphs()
        self._loaded = True

        exemplar_count = sum(1 for e in self._exemplars if e.glyph is not None)
        logger.info(
            "PrimitiveSpace loaded: %d words → %d roles (%d exemplar Glyphs), %d axioms",
            len(self._word_to_roles),
            len(self._role_to_words),
            exemplar_count,
            len(self._facts),
        )

    # Backward compat alias
    def load_teach_files(
        self,
        directory: str | Path | None = None,
        files: tuple[str, ...] | None = None,
    ) -> None:
        """Backward compat — calls load()."""
        self.load(directory=directory, files=files)

    def _load_jsonl(self, path: Path) -> None:
        """Parse a standard exemplar JSONL file.

        Same format as any glyphh model's exemplars.jsonl.
        """
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                rtype = entry.get("record_type", "")

                if rtype == "primitive":
                    self._load_primitive(entry)
                elif rtype == "axiom":
                    s = entry["subject"].lower()
                    r = entry["relation"].lower()
                    o = entry["object"].lower()
                    self._facts.append((s, r, o))

    def _load_primitive(self, entry: dict) -> None:
        """Load a single primitive entry from JSONL."""
        role = entry["role"]
        layer = entry["layer"]
        segment = entry["segment"]
        keywords = entry.get("keywords", [])
        question = entry.get("question", "")
        description = entry.get("description", "")

        # Build word → roles mapping from keywords
        for word in keywords:
            word_lower = word.lower()
            if word_lower not in self._word_to_roles:
                self._word_to_roles[word_lower] = []
            if role not in self._word_to_roles[word_lower]:
                self._word_to_roles[word_lower].append(role)
            self._role_to_words.setdefault(role, set()).add(word_lower)

        # Create exemplar (Glyph built in _build_exemplar_glyphs)
        exemplar = PrimitiveExemplar(
            role=role,
            layer=layer,
            segment=segment,
            glyph=None,  # built after all keywords loaded
            keywords=keywords,
            question=question,
            description=description,
        )
        self._exemplars.append(exemplar)
        self._role_to_exemplar[role] = exemplar

    def _build_centroids(self) -> None:
        """Build CharacterEncoder centroids for unknown word fallback.

        Each role's centroid = bundle of character n-gram vectors for
        all its keywords. Handles morphological variants of known words.
        """
        from glyphh.core.ops import bundle

        self._role_centroids.clear()
        for role, words in self._role_to_words.items():
            word_vecs = [self._char_encoder.encode_word(w) for w in words]
            if word_vecs:
                self._role_centroids[role] = (
                    bundle(word_vecs) if len(word_vecs) > 1 else word_vecs[0]
                )

    def _build_exemplar_glyphs(self) -> None:
        """Encode exemplar phrases into Glyphs.

        Bootstrap: uses the flat word→role dict (already loaded from
        keywords) to create a ThoughtGlyphEncoder, then encodes each
        exemplar's question field into a full Glyph.
        """
        from glyphh.memory.thought_glyph import ThoughtGlyphEncoder

        # Bootstrap encoder with self (flat word→role already loaded)
        encoder = ThoughtGlyphEncoder(primitives=self)

        for exemplar in self._exemplars:
            if exemplar.question:
                exemplar.glyph = encoder.encode_thought(
                    exemplar.question,
                    speaker="incoming",
                    metadata={
                        "role": exemplar.role,
                        "layer": exemplar.layer,
                        "segment": exemplar.segment,
                    },
                )

        exemplar_count = sum(1 for e in self._exemplars if e.glyph is not None)
        logger.debug(
            "Built %d exemplar Glyphs from %d primitives",
            exemplar_count,
            len(self._exemplars),
        )

    # ── Compound primitives (Level 1+) ────────────────────────────────────

    def add_compound(
        self,
        glyph: "Glyph",
        keywords: list[str],
        source: str = "",
    ) -> PrimitiveExemplar | None:
        """Register a crystallized compound primitive.

        Called by the deep DreamLoop when a structural pattern has been
        re-derived enough times to become permanent. The glyph IS the
        compound — the HDC vector already encodes the structural pattern.
        We just register it as an exemplar so future thoughts can match it.

        Args:
            glyph: The bridge thought's glyph (already encodes the pattern).
            keywords: Content words from the crystallized thoughts.
            source: Human-readable description of where this came from.

        Returns:
            The new PrimitiveExemplar, or None if no keywords.
        """
        from glyphh.core.ops import bundle

        keywords = [w.lower().strip() for w in keywords if w.strip()]
        if not keywords:
            return None

        # Compound role name from keywords
        compound_role = "compound_" + "_".join(sorted(keywords)[:3])

        # Determine primary layer/segment from glyph's activated attrs
        activated = glyph.metadata.get("_activated_attrs", [])
        layer = "semantic"
        segment = "category"
        for attr in activated:
            if attr.startswith("direction_"):
                continue  # skip direction — depends on speaker
            parts = attr.split("_", 1)
            if len(parts) == 2:
                layer, segment = parts
                break  # first non-direction attr

        exemplar = PrimitiveExemplar(
            role=compound_role,
            layer=layer,
            segment=segment,
            glyph=glyph,
            keywords=keywords,
            question=" ".join(keywords),
            description=source or f"crystallized: {' '.join(keywords)}",
        )
        self._exemplars.append(exemplar)

        # Register compound role in the encoder's routing table
        from glyphh.memory.thought_glyph import ROLE_TO_LAYER_SEGMENT
        if compound_role not in ROLE_TO_LAYER_SEGMENT:
            ROLE_TO_LAYER_SEGMENT[compound_role] = (layer, segment)

        # Register keywords so future encoding can match them
        for word in keywords:
            if word not in self._word_to_roles:
                self._word_to_roles[word] = []
            if compound_role not in self._word_to_roles[word]:
                self._word_to_roles[word].append(compound_role)
            self._role_to_words.setdefault(compound_role, set()).add(word)

            # Update centroid
            word_vec = self._char_encoder.encode_word(word)
            if compound_role in self._role_centroids:
                self._role_centroids[compound_role] = bundle([
                    self._role_centroids[compound_role], word_vec,
                ])
            else:
                self._role_centroids[compound_role] = word_vec

        self._version += 1
        logger.info(
            "Crystallized compound primitive: %s (%s) → %s/%s (v%d)",
            compound_role, ", ".join(keywords), layer, segment, self._version,
        )
        return exemplar

    @property
    def version(self) -> int:
        """Incremented each time a compound primitive is crystallized."""
        return self._version

    # ── Introspection ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        exemplar_count = sum(1 for e in self._exemplars if e.glyph is not None)
        return {
            "words": len(self._word_to_roles),
            "roles": len(self._role_to_words),
            "exemplars": exemplar_count,
            "centroids": len(self._role_centroids),
            "axioms": len(self._facts),
            "loaded": self._loaded,
        }

    def words_for_role(self, role: str) -> set[str]:
        """Get all words mapped to a role."""
        return self._role_to_words.get(role, set())

    def all_roles(self) -> list[str]:
        """List all known roles."""
        return sorted(self._role_to_words.keys())
