"""
Binding — structured facts composed from atoms.

A Binding is a role-filler frame encoded as a single HDC vector.  It
represents a fact: "chris is_a user", "glyphh uses hdc", "chris builds glyphh".

Each binding is a bundle of role-bound atoms:
    bind(SUBJECT_ROLE, atom("chris")) + bind(RELATION_ROLE, atom("is_a")) + bind(OBJECT_ROLE, atom("user"))

The result is a single vector at the same dimension as the atoms (2048).
This means bindings compose the same way atoms do — you can bind bindings
to bindings, bundle them, chain them.  Complexity from connection, not size.

Variable arity: a binding can have any number of role-filler pairs.  A
simple fact has 3 (subject, relation, object).  A rich event might have 6
(agent, action, patient, instrument, time, location).  The algebra doesn't
care — it's all bind + bundle.

Usage:
    forge = AtomForge(dimension=2048)
    facts = FactStore(forge)

    # Teach a fact
    facts.teach("chris", "is_a", "user")
    facts.teach("chris", "builds", "glyphh")
    facts.teach("glyphh", "uses", "hdc")

    # Query: what does chris build?
    results = facts.query(subject="chris", relation="builds")
    # → [("glyphh", 0.85)]

    # Compose: does chris use hdc?  (never taught directly)
    chain = facts.infer("chris", "uses", depth=2)
    # → [("hdc", via=["chris→builds→glyphh", "glyphh→uses→hdc"])]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity
from .atom import Atom, AtomForge

logger = logging.getLogger(__name__)


# ── Fact ───────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """A structured fact — a role-filler frame encoded as an HDC vector.

    Attributes:
        roles:      Dict of role_name → atom_name (e.g. {"subject": "chris", "relation": "is_a", "object": "user"})
        vector:     The composed HDC vector (bundle of role-bound atoms)
        strength:   Hebbian reinforcement
        created_at: Unix timestamp
        source:     How this fact was created: "taught", "derived", "observed"
    """

    roles: dict[str, str]
    vector: np.ndarray
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    source: str = "taught"

    @property
    def subject(self) -> str | None:
        return self.roles.get("subject")

    @property
    def relation(self) -> str | None:
        return self.roles.get("relation")

    @property
    def object(self) -> str | None:
        return self.roles.get("object")

    def reinforce(self, amount: float = 0.1) -> None:
        self.strength = min(3.0, self.strength + amount / (1.0 + 0.1 * self.strength))

    def __repr__(self) -> str:
        parts = [f"{r}={v}" for r, v in self.roles.items() if not r.startswith("_")]
        return f"Fact({', '.join(parts)}, strength={self.strength:.2f})"


# ── FactStore ──────────────────────────────────────────────────────────────

class FactStore:
    """Stores and queries structured facts composed from atoms.

    Facts are role-filler frames encoded as HDC vectors.  Queries use
    partial binding + cosine similarity to find matching facts.
    Inference follows bind chains to derive unstated knowledge.
    """

    def __init__(self, forge: AtomForge) -> None:
        self._forge = forge
        self._facts: list[Fact] = []
        self._dirty = False

    @property
    def count(self) -> int:
        return len(self._facts)

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts)

    # ── Teaching ───────────────────────────────────────────────────────────

    def teach(
        self,
        subject: str,
        relation: str,
        object: str,
        source: str = "taught",
        **extra_roles: str,
    ) -> Fact:
        """Teach Ada a simple fact: subject → relation → object.

        Extra roles can be added as keyword arguments:
            facts.teach("chris", "builds", "glyphh", time="2026", location="home")
        """
        roles = {"subject": subject, "relation": relation, "object": object}
        roles.update(extra_roles)
        return self.teach_frame(roles, source=source)

    def teach_frame(self, roles: dict[str, str], source: str = "taught") -> Fact:
        """Teach a variable-arity fact from a role dict."""
        # Ensure all atoms exist
        for role_name, atom_name in roles.items():
            kind = "relation" if role_name == "relation" else "entity"
            self._forge.atom(atom_name, kind=kind)

        # Encode: bundle of bind(role_vector, atom_vector) for each role
        vec = self._encode_frame(roles)

        fact = Fact(roles=roles, vector=vec, source=source)
        self._facts.append(fact)
        self._dirty = True
        logger.debug("Taught fact: %s", fact)
        return fact

    def _encode_frame(self, roles: dict[str, str]) -> np.ndarray:
        """Encode a role-filler frame as a single HDC vector."""
        bindings = []
        for role_name, atom_name in roles.items():
            role_vec = self._forge.role(role_name) if role_name in self._forge.ROLES else self._forge.atom(f"_custom_role_{role_name}", kind="role")
            atom_vec = self._forge.atom(atom_name)
            bindings.append(bind(role_vec, atom_vec))

        return bundle(bindings) if len(bindings) > 1 else bindings[0]

    # ── Querying ───────────────────────────────────────────────────────────

    def query(
        self,
        subject: str | None = None,
        relation: str | None = None,
        object: str | None = None,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[tuple[Fact, float]]:
        """Query facts by partial match.

        Provide any combination of subject, relation, object.  The query
        is encoded as the bundle of the provided role-bindings, then
        compared to all stored facts via cosine similarity.

        Returns (fact, score) pairs sorted descending.
        """
        if not self._facts:
            return []

        # Build partial query vector from provided roles
        query_roles = {}
        if subject:
            query_roles["subject"] = subject
        if relation:
            query_roles["relation"] = relation
        if object:
            query_roles["object"] = object

        if not query_roles:
            return []

        query_vec = self._encode_frame(query_roles)

        results = []
        for fact in self._facts:
            sim = float(cosine_similarity(query_vec, fact.vector))
            weighted = sim * fact.strength
            if weighted >= min_score:
                results.append((fact, weighted))

        results.sort(key=lambda x: x[1], reverse=True)

        # Reinforce recalled facts
        for fact, _ in results[:top_k]:
            fact.reinforce()
            self._dirty = True

        return results[:top_k]

    def query_object(
        self,
        subject: str,
        relation: str,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Query: given subject + relation, what is the object?

        Uses algebraic unbinding: encode subject+relation partial,
        then unbind from each fact to recover the object slot,
        then cleanup to nearest known atom.

        Returns (atom_name, score) pairs.
        """
        subj_vec = self._forge.atom(subject)
        rel_vec = self._forge.atom(relation)
        subj_role = self._forge.role("subject")
        rel_role = self._forge.role("relation")
        obj_role = self._forge.role("object")

        results = []
        for fact in self._facts:
            # Check if this fact involves the subject and relation
            if fact.subject and fact.relation:
                subj_sim = self._forge.similarity(
                    self._forge.unbind(fact.vector, subj_role),
                    subj_vec,
                )
                rel_sim = self._forge.similarity(
                    self._forge.unbind(fact.vector, rel_role),
                    rel_vec,
                )

                if subj_sim > 0.05 and rel_sim > 0.05:
                    # Unbind object role to recover the object atom
                    obj_noisy = self._forge.unbind(fact.vector, obj_role)
                    cleaned = self._forge.nearest(obj_noisy, top_k=3, min_sim=0.05)
                    for atom, sim in cleaned:
                        if atom.kind != "role":  # skip structural atoms
                            results.append((atom.name, sim * fact.strength))

        results.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        deduped = []
        for name, score in results:
            if name not in seen:
                seen.add(name)
                deduped.append((name, score))
        return deduped[:top_k]

    # ── Inference (bind chains) ────────────────────────────────────────────

    def infer(
        self,
        subject: str,
        relation: str,
        depth: int = 2,
    ) -> list[tuple[str, float, list[str]]]:
        """Infer unstated facts via bind chains.

        If we know "chris builds glyphh" and "glyphh uses hdc",
        then infer("chris", "uses", depth=2) should find "hdc"
        via the chain chris → builds → glyphh → uses → hdc.

        Returns (object_name, confidence, chain_description) triples.
        """
        results = []

        # Direct facts first (depth=1)
        direct = self.query_object(subject, relation, top_k=5)
        for obj, score in direct:
            results.append((obj, score, [f"{subject} → {relation} → {obj}"]))

        if depth <= 1:
            return results

        # Follow chains: find what subject connects to, then query from there
        # Get all objects connected to subject (any relation)
        connections = self._get_connections(subject)

        for intermediate, via_relation, conn_score in connections:
            # From the intermediate, query for the target relation
            indirect = self.query_object(intermediate, relation, top_k=3)
            for obj, obj_score in indirect:
                combined_score = conn_score * obj_score
                if combined_score > 0.01:
                    chain = [
                        f"{subject} → {via_relation} → {intermediate}",
                        f"{intermediate} → {relation} → {obj}",
                    ]
                    results.append((obj, combined_score, chain))

        # Sort by score, deduplicate
        results.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        deduped = []
        for obj, score, chain in results:
            if obj not in seen:
                seen.add(obj)
                deduped.append((obj, score, chain))
        return deduped

    def _get_connections(self, subject: str) -> list[tuple[str, str, float]]:
        """Find all (object, relation, score) connected to a subject."""
        subj_vec = self._forge.atom(subject)
        subj_role = self._forge.role("subject")
        rel_role = self._forge.role("relation")
        obj_role = self._forge.role("object")

        connections = []
        for fact in self._facts:
            # Check if this fact has the subject
            subj_noisy = self._forge.unbind(fact.vector, subj_role)
            subj_sim = self._forge.similarity(subj_noisy, subj_vec)

            if subj_sim > 0.05:
                # Extract relation and object
                rel_noisy = self._forge.unbind(fact.vector, rel_role)
                rel_match = self._forge.nearest(rel_noisy, top_k=1, min_sim=0.05)

                obj_noisy = self._forge.unbind(fact.vector, obj_role)
                obj_match = self._forge.nearest(obj_noisy, top_k=1, min_sim=0.05)

                if rel_match and obj_match:
                    rel_atom, rel_sim = rel_match[0]
                    obj_atom, obj_sim = obj_match[0]
                    if rel_atom.kind != "role" and obj_atom.kind != "role":
                        score = subj_sim * rel_sim * obj_sim * fact.strength
                        connections.append((obj_atom.name, rel_atom.name, score))

        connections.sort(key=lambda x: x[2], reverse=True)
        return connections

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save facts to disk. Skips if nothing changed."""
        if not self._dirty:
            return

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta_path = path / "facts.jsonl"
        vec_path = path / "fact_vectors.npy"

        with open(meta_path, "w") as f:
            for fact in self._facts:
                record = {
                    "roles": fact.roles,
                    "strength": fact.strength,
                    "created_at": fact.created_at,
                    "source": fact.source,
                }
                f.write(json.dumps(record) + "\n")

        if self._facts:
            np.save(vec_path, np.stack([f.vector for f in self._facts]))

        self._dirty = False
        logger.info("Saved %d facts to %s", len(self._facts), path)

    def load(self, path: str | Path) -> None:
        """Load facts from disk."""
        path = Path(path)
        meta_path = path / "facts.jsonl"
        vec_path = path / "fact_vectors.npy"

        if not meta_path.exists() or not vec_path.exists():
            return

        records = []
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        vectors = np.load(vec_path)
        if len(records) != vectors.shape[0]:
            logger.warning("Fact count mismatch — skipping load")
            return

        for i, rec in enumerate(records):
            # Ensure atoms exist in forge
            for role_name, atom_name in rec["roles"].items():
                kind = "relation" if role_name == "relation" else "entity"
                self._forge.atom(atom_name, kind=kind)

            self._facts.append(Fact(
                roles=rec["roles"],
                vector=vectors[i],
                strength=rec.get("strength", 1.0),
                created_at=rec.get("created_at", 0),
                source=rec.get("source", "loaded"),
            ))

        self._dirty = False
        logger.info("Loaded %d facts from %s", len(self._facts), path)
