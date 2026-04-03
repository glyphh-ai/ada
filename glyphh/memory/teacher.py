"""
Teacher — decomposes natural language into atoms and bindings.

Two modes:
  1. Simple parser — handles "X is Y", "X verb Y" patterns without LLM
  2. LLM decomposer — uses Qwen to break complex sentences into primitives

The LLM is the teacher, not the thinker.  It decomposes language into
structured facts that Ada can reason over with pure HDC algebra.

Usage:
    teacher = Teacher(forge, facts)

    # Simple (no LLM needed)
    teacher.parse("Chris is a user")
    # → Fact(subject=chris, relation=is_a, object=user)

    # Complex (LLM decomposes into primitives)
    teacher.decompose("Monetary policy affects inflation through interest rates", engine)
    # → Fact(subject=monetary_policy, relation=affects, object=inflation)
    # → Fact(subject=monetary_policy, relation=uses, object=interest_rates)
    # → Fact(subject=interest_rates, relation=influences, object=inflation)
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from dataclasses import dataclass, field as dc_field

from .atom import AtomForge
from .binding import Fact, FactStore


@dataclass
class LearnResult:
    """Summary of what was learned from a lesson file."""
    atoms_created: int = 0
    pairs_created: int = 0
    compositions_created: int = 0
    facts_created: int = 0
    all_facts: list[Fact] = dc_field(default_factory=list)

    @property
    def total(self) -> int:
        return self.atoms_created + self.pairs_created + self.compositions_created + self.facts_created

if TYPE_CHECKING:
    from glyphh.llm.engine import LLMEngine

logger = logging.getLogger(__name__)


# ── Simple pattern parser ──────────────────────────────────────────────────

# Patterns for basic declarative sentences (no LLM needed)
_IS_A_PATTERN = re.compile(
    r"^(.+?)\s+(?:(?:is|am|are)\s+(?:a|an|the)\s+)(.+)$", re.IGNORECASE
)
_IS_PATTERN = re.compile(
    r"^(.+?)\s+(?:is|am|are)\s+(.+)$", re.IGNORECASE
)
_SVO_PATTERN = re.compile(
    r"^(.+?)\s+(builds?|creates?|uses?|has|makes?|runs?|likes?|loves?|hates?|knows?|wants?|needs?|owns?|writes?|reads?|teaches?|learns?|helps?|affects?|influences?|controls?|manages?|contains?|includes?|requires?|provides?|supports?|enables?|produces?|generates?|processes?|handles?|stores?|tracks?|monitors?|deploys?|encodes?|decodes?|binds?|bundles?)\s+(.+)$",
    re.IGNORECASE,
)
_CALLED_PATTERN = re.compile(
    r"^(.+?)\s+(?:is\s+called|is\s+named)\s+(.+)$", re.IGNORECASE,
)
_HAS_PATTERN = re.compile(
    r"^(.+?)(?:'s|'s)\s+(.+?)\s+is\s+(.+)$", re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Normalize an atom name: lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", "_", text.strip().lower())


def _split_clauses(text: str) -> list[str]:
    """Split compound sentences on 'and'/'but' into individual clauses."""
    # Only split on " and " / " but " that join independent clauses
    parts = re.split(r"\s+(?:and|but)\s+", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _parse_one(text: str) -> list[dict[str, str]]:
    """Try to parse a single clause into role dicts."""
    text = text.strip().rstrip(".")

    # "Chris's name is Christopher"
    m = _HAS_PATTERN.match(text)
    if m:
        return [{
            "subject": _normalize(m.group(1)),
            "relation": _normalize(m.group(2)),
            "object": _normalize(m.group(3)),
        }]

    # "Chris is called Ada"
    m = _CALLED_PATTERN.match(text)
    if m:
        return [{
            "subject": _normalize(m.group(1)),
            "relation": "named",
            "object": _normalize(m.group(2)),
        }]

    # "Chris is a user"
    m = _IS_A_PATTERN.match(text)
    if m:
        return [{
            "subject": _normalize(m.group(1)),
            "relation": "is_a",
            "object": _normalize(m.group(2)),
        }]

    # "HDC is cool" (is without article)
    m = _IS_PATTERN.match(text)
    if m:
        return [{
            "subject": _normalize(m.group(1)),
            "relation": "is",
            "object": _normalize(m.group(2)),
        }]

    # "Chris builds Glyphh"
    m = _SVO_PATTERN.match(text)
    if m:
        verb = _normalize(m.group(2))
        # Strip trailing 's' for simple verb normalization
        if verb.endswith("s") and not verb.endswith("ss"):
            verb = verb[:-1]
        return [{
            "subject": _normalize(m.group(1)),
            "relation": verb,
            "object": _normalize(m.group(3)),
        }]

    return []


def _parse_simple(text: str) -> list[dict[str, str]]:
    """Try to parse a declarative sentence into role dicts.

    Handles compound sentences ("X is Y and Z is W") by splitting
    on conjunctions first, then parsing each clause.
    """
    clauses = _split_clauses(text)
    results = []
    for clause in clauses:
        results.extend(_parse_one(clause))
    return results


# ── LLM decomposition prompt ──────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
Extract factual statements from the text below. Each fact has a subject, \
relation, and object. Output ONLY a JSON array of objects with keys \
"subject", "relation", "object". Use lowercase, underscores for spaces.

Rules:
- Only extract declarative facts. Ignore questions, greetings, commands.
- If the text contains no facts, return an empty array: []
- Keep atoms simple — single words or short compounds with underscores.

Text: {sentence}

JSON:"""


# ── Teacher ────────────────────────────────────────────────────────────────

class Teacher:
    """Decomposes natural language into atoms and structured facts.

    Simple sentences are parsed directly (no LLM).  Complex sentences
    are decomposed by the LLM into primitives.
    """

    def __init__(self, forge: AtomForge, facts: FactStore) -> None:
        self._forge = forge
        self._facts = facts

    def parse(self, text: str) -> list[Fact]:
        """Parse a simple declarative sentence into facts (no LLM).

        Returns list of facts created, or empty list if unparseable.
        """
        role_dicts = _parse_simple(text)
        results = []
        for roles in role_dicts:
            fact = self._facts.teach_frame(roles, source="taught")
            results.append(fact)
        return results

    def decompose(self, text: str, engine: "LLMEngine") -> list[Fact]:
        """Use the LLM to decompose a complex sentence into primitive facts.

        The LLM is the teacher — it breaks complex language into simple
        subject-relation-object triples that Ada can reason over.
        """
        prompt = _DECOMPOSE_PROMPT.format(sentence=text)
        response = engine.generate(prompt, max_tokens=256, temperature=0.1)

        # Parse JSON from response
        facts = self._parse_llm_response(response)

        results = []
        for roles in facts:
            if "subject" in roles and "relation" in roles and "object" in roles:
                clean_roles = {
                    "subject": _normalize(roles["subject"]),
                    "relation": _normalize(roles["relation"]),
                    "object": _normalize(roles["object"]),
                }
                fact = self._facts.teach_frame(clean_roles, source="decomposed")
                results.append(fact)

        logger.info("LLM decomposed '%s' into %d facts", text[:60], len(results))
        return results

    def learn_file(self, path: str | Path) -> LearnResult:
        """Load a .teach file and learn from it.

        .teach format supports three levels of primitives:

          Level 0 — Existence (bare atom, just create it):
            ah
            ee
            buh

          Level 1 — Pairing (two things associated, colon separator):
            a : ay
            b : buh
            mama : her

          Level 2 — Composition (sounds combine into a word, equals sign):
            muh + ah + muh = mama
            kuh + ah + tuh = cat

          Level 3 — Triples (subject relation object, only when relations have meaning):
            cat has fur
            dog has fur

        Lines starting with # are comments.  Blank lines are ignored.
        """
        from pathlib import Path
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Lesson file not found: {path}")

        result = LearnResult()

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" in line:
                    # Level 2: composition — "muh + ah + muh = mama"
                    self._learn_composition(line, result)
                elif ":" in line:
                    # Level 1: pairing — "a : ay"
                    self._learn_pair(line, result)
                elif len(line.split()) == 1:
                    # Level 0: existence — bare atom
                    self._learn_atom(line, result)
                elif len(line.split()) >= 3:
                    # Level 3: triple — "cat has fur"
                    self._learn_triple(line, result)
                elif len(line.split()) == 2:
                    # Treat two tokens as a pair (shorthand without colon)
                    parts = line.split()
                    self._learn_pair(f"{parts[0]} : {parts[1]}", result)
                else:
                    logger.warning("Skipping unrecognized line: %s", line)

        logger.info(
            "Learned from %s: %d atoms, %d pairs, %d compositions, %d facts",
            path.name, result.atoms_created, result.pairs_created,
            result.compositions_created, result.facts_created,
        )
        return result

    def _learn_atom(self, line: str, result: LearnResult) -> None:
        """Level 0: just create the atom. It exists."""
        name = line.strip().lower()
        self._forge.atom(name)
        result.atoms_created += 1

    def _learn_pair(self, line: str, result: LearnResult) -> None:
        """Level 1: associate two things. Implicit binding."""
        parts = [p.strip().lower() for p in line.split(":")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.warning("Malformed pair: %s", line)
            return

        left, right = parts[0], parts[1]
        # Create both atoms
        left_vec = self._forge.atom(left)
        right_vec = self._forge.atom(right)
        # Bind them together — pure association, no named relation
        bound = self._forge.bind(left_vec, right_vec)
        # Store as a fact with implicit "associated_with" relation
        fact = self._facts.teach(left, "associated_with", right, source="pair")
        result.pairs_created += 1
        result.all_facts.append(fact)

    def _learn_composition(self, line: str, result: LearnResult) -> None:
        """Level 2: sounds combine into a word.

        Format: 'muh + ah + muh = mama'
        The components are bundled into a sequence, then bound to the word.
        """
        if "=" not in line:
            return

        parts_side, word_side = line.split("=", 1)
        word = word_side.strip().lower()

        # Parse components (split by +)
        components = [c.strip().lower() for c in parts_side.split("+")]
        components = [c for c in components if c]

        if not components or not word:
            logger.warning("Malformed composition: %s", line)
            return

        # Create atoms for each component and the word
        comp_vecs = [self._forge.atom(c) for c in components]
        word_vec = self._forge.atom(word)

        # Encode the composition as a pathway (ordered sequence)
        # Bind each component to its position, then bundle
        from glyphh.core.ops import generate_symbol
        positional = []
        for i, vec in enumerate(comp_vecs):
            pos = generate_symbol(self._forge._seed, f"seq_pos_{i}", self._forge.dimension)
            positional.append(self._forge.bind(pos, vec))

        composed = self._forge.bundle(positional) if len(positional) > 1 else positional[0]

        # Store the composition as a fact: word composed_of components
        components_str = "+".join(components)
        fact = self._facts.teach(word, "composed_of", components_str, source="composition")
        result.compositions_created += 1
        result.all_facts.append(fact)

        # Also store each component relationship
        for i, comp in enumerate(components):
            pos_fact = self._facts.teach(word, f"sound_{i}", comp, source="composition")
            result.all_facts.append(fact)

    def _learn_triple(self, line: str, result: LearnResult) -> None:
        """Level 3: subject relation object."""
        parts = line.split()
        subject = parts[0].lower()
        relation = parts[1].lower()
        obj = "_".join(parts[2:]).lower()
        fact = self._facts.teach(subject, relation, obj, source="lesson")
        result.facts_created += 1
        result.all_facts.append(fact)

    def learn_lesson(self, name: str) -> LearnResult:
        """Load a built-in lesson by name.

        Built-in lessons live in glyphh/memory/lessons/{name}.teach.
        """
        from pathlib import Path
        lessons_dir = Path(__file__).parent / "lessons"
        path = lessons_dir / f"{name}.teach"
        if not path.exists():
            available = [p.stem for p in lessons_dir.glob("*.teach")]
            raise FileNotFoundError(
                f"Lesson '{name}' not found. Available: {available}"
            )
        return self.learn_file(path)

    @staticmethod
    def _parse_llm_response(response: str) -> list[dict]:
        """Extract JSON array from LLM response, tolerating noise."""
        # Find JSON array in response
        start = response.find("[")
        end = response.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []

        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM decomposition: %s", response[:100])
            return []
