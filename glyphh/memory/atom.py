"""
Atom — the neuron-level primitive of Ada's thought.

An Atom is a small (2,048-dim) bipolar HDC vector representing a single
concept: a word, an entity, a relation type.  Atoms are the smallest unit
of meaning — individually simple, but composable via bind and bundle into
arbitrarily complex structures.

Complexity comes from connection, not size.  A binding of two atoms is
itself an atom-sized vector.  A bundle of bindings is atom-sized.  A
pathway through bundles is atom-sized.  Every level of abstraction lives
in the same 2,048-dimensional space — just like every neuron in the brain
is roughly the same size, from V1 to prefrontal cortex.

Encoding uses CharacterEncoder (positional char n-grams) so atoms are
deterministic, handle misspellings, and require no training data.  The
LLM never touches this layer — it's pure HDC.

Usage:
    forge = AtomForge(dimension=2048)

    # Create atoms from words
    chris = forge.atom("chris")
    name  = forge.atom("name")
    is_a  = forge.atom("is_a")

    # Bind into a relation
    relation = forge.bind(is_a, name)       # "is a name"
    fact     = forge.bind(chris, relation)   # "chris is a name"

    # Bundle multiple facts
    scene = forge.bundle([fact1, fact2, fact3])

    # Query: unbind to recover
    recovered = forge.unbind(fact, chris)    # ≈ relation
    forge.nearest(recovered)                 # → "is_a__name" (cleanup)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol
from glyphh.linguistics.character import CharacterEncoder

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_DIM = 2048
DEFAULT_SEED = 42


# ── Atom ───────────────────────────────────────────────────────────────────

@dataclass
class Atom:
    """A single concept — the neuron of Ada's thought.

    Attributes:
        name:     Human-readable label (e.g. "chris", "is_a", "building").
        vector:   Bipolar int8 vector of shape (dimension,).
        kind:     Category — "entity", "relation", "attribute", "role", "derived".
        strength: Hebbian reinforcement [0, 3.0]. Grows with use.
        metadata: Arbitrary key-value pairs.
    """

    name: str
    vector: np.ndarray
    kind: str = "entity"
    strength: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, amount: float = 0.1) -> None:
        """Hebbian strengthening with diminishing returns."""
        self.strength = min(3.0, self.strength + amount / (1.0 + 0.1 * self.strength))

    def weaken(self, factor: float = 0.95) -> None:
        """Decay unused atoms."""
        self.strength = max(0.0, self.strength * factor)


# ── AtomForge ──────────────────────────────────────────────────────────────

class AtomForge:
    """Creates, stores, and retrieves atoms — Ada's concept vocabulary.

    The forge maintains a dictionary of named atoms.  New atoms are encoded
    via CharacterEncoder (positional char n-grams → bipolar HDC vector).
    The dictionary doubles as a cleanup memory: after noisy operations
    (unbind, deep composition), `nearest()` snaps a vector back to the
    closest known atom.

    Args:
        dimension: Vector dimension (default 2048 — neuron-scale).
        seed:      Deterministic seed for symbol generation.
    """

    # ── Role atoms (structural, not content) ───────────────────────────────

    ROLES = (
        "subject", "relation", "object",
        "agent", "action", "patient",
        "attribute", "value",
        "time", "location", "source", "target",
    )

    def __init__(
        self,
        dimension: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._dim = dimension
        self._seed = seed
        self._char_enc = CharacterEncoder(dimension=dimension, seed=seed)
        self._atoms: dict[str, Atom] = {}
        self._dirty = False

        # Pre-seed role atoms — structural scaffolding for bindings
        for role in self.ROLES:
            vec = generate_symbol(seed, f"role__{role}", dimension)
            self._atoms[f"_role_{role}"] = Atom(
                name=f"_role_{role}",
                vector=vec,
                kind="role",
            )

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return len(self._atoms)

    # ── Atom creation ──────────────────────────────────────────────────────

    def atom(self, name: str, kind: str = "entity") -> np.ndarray:
        """Get or create an atom by name. Returns the vector.

        If the atom already exists, returns its vector and reinforces it.
        If new, encodes it via CharacterEncoder and stores it.
        """
        key = name.lower().strip()
        if key in self._atoms:
            self._atoms[key].reinforce()
            return self._atoms[key].vector

        vec = self._char_enc.encode_word(key) if len(key.split()) == 1 else self._char_enc.encode_text(key)
        self._atoms[key] = Atom(name=key, vector=vec, kind=kind)
        self._dirty = True
        logger.debug("Created atom: %s (%s)", key, kind)
        return vec

    def role(self, name: str) -> np.ndarray:
        """Get a role atom by name. Roles are structural, not content."""
        key = f"_role_{name}"
        if key not in self._atoms:
            raise ValueError(f"Unknown role '{name}'. Available: {self.ROLES}")
        return self._atoms[key].vector

    def get(self, name: str) -> Atom | None:
        """Look up an atom by name. Returns None if not found."""
        return self._atoms.get(name.lower().strip())

    def has(self, name: str) -> bool:
        return name.lower().strip() in self._atoms

    # ── HDC operations (convenience wrappers at atom scale) ────────────────

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two vectors — creates a structured association."""
        return bind(a, b)

    def unbind(self, composite: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbind — recover the other component from a binding.

        For bipolar vectors: unbind(bind(a, b), a) ≈ b
        Since bind is element-wise multiply and a*a = 1 for bipolar.
        """
        return bind(composite, key)  # bind IS unbind for bipolar vectors

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Bundle vectors — creates a superposition / set."""
        return bundle(vectors)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(cosine_similarity(a, b))

    # ── Cleanup memory ─────────────────────────────────────────────────────

    def nearest(self, vector: np.ndarray, top_k: int = 1, min_sim: float = 0.05) -> list[tuple[Atom, float]]:
        """Find the nearest known atoms to a (possibly noisy) vector.

        This is the cleanup memory — after noisy operations (unbind, deep
        composition), snap back to the closest clean prototype.

        Returns list of (Atom, similarity) sorted descending.
        """
        if not self._atoms:
            return []

        results = []
        for atom in list(self._atoms.values()):  # snapshot — thread-safe
            sim = float(cosine_similarity(vector, atom.vector))
            if sim >= min_sim:
                results.append((atom, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def cleanup(self, vector: np.ndarray) -> Atom | None:
        """Snap a noisy vector to its nearest known atom.

        Returns None if no atom exceeds min_sim threshold.
        """
        matches = self.nearest(vector, top_k=1, min_sim=0.05)
        return matches[0][0] if matches else None

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save all atoms to disk. Skips if nothing changed."""
        if not self._dirty:
            return

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta_path = path / "atoms.jsonl"
        vec_path = path / "atom_vectors.npy"

        atoms_list = list(self._atoms.values())
        with open(meta_path, "w") as f:
            for atom in atoms_list:
                record = {
                    "name": atom.name,
                    "kind": atom.kind,
                    "strength": atom.strength,
                    "metadata": atom.metadata,
                }
                f.write(json.dumps(record) + "\n")

        if atoms_list:
            np.save(vec_path, np.stack([a.vector for a in atoms_list]))

        self._dirty = False
        logger.info("Saved %d atoms to %s", len(atoms_list), path)

    def load(self, path: str | Path) -> None:
        """Load atoms from disk. Merges with existing atoms."""
        path = Path(path)
        meta_path = path / "atoms.jsonl"
        vec_path = path / "atom_vectors.npy"

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
            logger.warning("Atom count mismatch — skipping load")
            return

        for i, rec in enumerate(records):
            name = rec["name"]
            self._atoms[name] = Atom(
                name=name,
                vector=vectors[i],
                kind=rec.get("kind", "entity"),
                strength=rec.get("strength", 1.0),
                metadata=rec.get("metadata", {}),
            )

        self._dirty = False
        logger.info("Loaded %d atoms from %s", len(records), path)

    # ── Introspection ──────────────────────────────────────────────────────

    def all_atoms(self, kind: str | None = None) -> list[Atom]:
        """List all atoms, optionally filtered by kind."""
        atoms = list(self._atoms.values())
        if kind:
            atoms = [a for a in atoms if a.kind == kind]
        return sorted(atoms, key=lambda a: a.strength, reverse=True)
