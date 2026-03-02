"""
DeductiveLayer — HDC deductive reasoning for implicit prerequisite detection.

A general-purpose predictive coding layer that detects when accumulated context
diverges from the current state, and identifies prerequisites needed to resolve
the gap. This mirrors how the brain works:

  ACCUMULATE → COMPARE → RESOLVE
  (observe)    (mismatch)  (transition)

The layer tracks two independent HDC vectors:
  1. **state_vector**: where we currently ARE (encodes the current state)
  2. **target_vector**: where our actions are DIRECTED TO (decaying superposition
     of states that recent actions point toward)

When target_vector ≠ state_vector (cosine divergence), the layer resolves the
mismatch against a transition library to identify prerequisites. The transition
library is pre-seeded and learns via Hebbian reinforcement.

Three mismatch detection levels (checked in order):
  1. **Explicit**: target-vs-state cosine divergence (classic deductive)
  2. **Temporal stagnation**: state unchanged despite directing actions (temporal)
  3. **Beam divergence**: predicted state trajectory diverges from actual (beam search)

This is domain-agnostic — the same mechanism works for:
  - Auth: unauthenticated != requires auth -> prerequisite authenticate
  - Shopping: no cart != wants to checkout -> prerequisite add_to_cart
  - Navigation: state_x != target state_y -> prerequisite transition
  - Any domain where actions have implicit prerequisites

Load domain knowledge via packs (like IntentExtractor):

    from glyphh.state import DeductiveLayer

    # With domain packs — transitions pre-loaded:
    deductive = DeductiveLayer(dimension=10000, seed=89, packs=["my_domain"])

    # Observe + deduce — transitions are already registered from pack:
    deductive.observe(state="state_x", actions=["move"], targets=["state_y"])
    result = deductive.deduce(query="search the items", current_state="state_x")
    # -> {"prerequisites": ["transition"], "confidence": 0.85, "target": "state_y", ...}

    # Temporal stagnation: state hasn't changed despite directing actions:
    deductive = DeductiveLayer(dimension=10000, seed=89, packs=["my_domain"])
    deductive.observe(state="state_x", actions=[])
    deductive.observe(state="state_x", actions=["create"], targets=["state_y"])
    deductive.observe(state="state_x", actions=["move"], targets=["state_y"])
    result = deductive.deduce(query="search the items", current_state="state_x")
    # -> stagnation detected -> prerequisites: ["transition"]

    # Or register transitions manually:
    deductive = DeductiveLayer(dimension=10000, seed=89)
    deductive.add_transition(
        name="auth_gate",
        directing_actions=["request", "submit"],
        operating_actions=["view", "post", "edit"],
        prerequisite="authenticate",
    )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol

# Data directory — shared with InductiveLayer
_DATA_DIR = Path(__file__).parent / "data"


# ── Helpers ──

def _weighted_bundle(pairs: list[tuple[np.ndarray, float]], dimension: int) -> np.ndarray:
    """Weighted majority vote of bipolar vectors."""
    total = np.zeros(dimension, dtype=np.float32)
    for vec, weight in pairs:
        total += vec.astype(np.float32) * weight
    return np.where(total >= 0, 1, -1).astype(np.int8)


@dataclass
class Transition:
    """A learned prerequisite pattern.

    Encodes the relationship: "if recent actions directed state toward a target,
    and current query wants to operate in that context, this prerequisite is needed."

    strength increases with Hebbian reinforcement (confirmed correct use).
    """

    name: str
    vector: np.ndarray              # HDC signature of the directing→operating pattern
    prerequisite: str               # action to inject (e.g. "navigate", "authenticate")
    strength: float = 1.0           # Hebbian weight
    fire_count: int = 0             # times confirmed correct

    def strengthen(self, amount: float = 0.1) -> None:
        """Hebbian reinforcement with diminishing returns."""
        self.strength = min(3.0, self.strength + amount * (1.0 / self.strength))
        self.fire_count += 1

    def weaken(self, factor: float = 0.85) -> None:
        """Reduce strength but don't drop below 0.3."""
        self.strength = max(0.3, self.strength * factor)


class DeductiveLayer:
    """HDC deductive reasoning for implicit prerequisite detection.

    Domain-agnostic — works with any state/action/target vocabulary.
    Consumers register transition patterns that map (directing_actions,
    operating_actions) → prerequisite. The layer detects when recent
    directing actions have created a target that diverges from the current
    state, and resolves the gap by emitting the prerequisite.

    Three-phase cycle per turn:
      1. ACCUMULATE (observe): Record state + actions + targets
      2. COMPARE (deduce): Detect target ≠ state mismatch via cosine
      3. RESOLVE (deduce): Match against transition library → prerequisites
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: int = 89,
        decay: float = 0.7,
        mismatch_threshold: float = 0.10,
        min_confidence: float = 0.15,
        packs: list[str] | None = None,
        temporal_window: int = 3,
        enable_beam: bool = False,
    ):
        self._dim = dimension
        self._seed = seed
        self._decay = decay
        self._mismatch_threshold = mismatch_threshold
        self._min_confidence = min_confidence

        # State tracking
        self._state_vector: Optional[np.ndarray] = None
        self._target_history: list[np.ndarray] = []
        self._last_target: Optional[str] = None

        # Action history
        self._action_history: list[list[str]] = []

        # Transition library
        self._transitions: dict[str, Transition] = {}

        # Registered action sets for pattern matching
        self._directing_actions: set[str] = set()
        self._operating_verbs: set[str] = set()
        self._resolving_actions: set[str] = set()  # actions that resolve mismatch

        # Temporal stagnation detection — tracks how much state changes per turn.
        # When state is unchanged despite directing actions, stagnation is high.
        self._temporal_window = temporal_window
        self._state_deltas: list[float] = []  # Cosine between consecutive states

        # Beam state prediction — forecasts expected state trajectory using
        # BeamSearchPredictor, detects divergence from actual state.
        self._enable_beam = enable_beam
        self._state_glyph_history: list = []  # Minimal Glyph objects per turn
        self._beam_predictor = None
        if enable_beam:
            from glyphh.temporal import BeamSearchPredictor
            self._beam_predictor = BeamSearchPredictor(
                beam_width=3, drift_reduction=True,
            )

        # Operating action NL mapping — loaded from intent packs (same domain).
        # Maps query words/phrases back to canonical function names
        # so the deductive layer can detect operating intent from NL queries.
        self._operating_phrases: list[tuple[str, str]] = []  # (phrase, canonical)
        self._operating_synonyms: dict[str, str] = {}  # word → canonical

        # Load domain packs — known transitions, no manual registration needed
        if packs:
            self._load_packs(packs)

    def _load_packs(self, packs: list[str]) -> None:
        """Load domain knowledge packs and register their transitions.

        Loads from two sources per pack:
          1. State pack (glyphh/state/data/packs/{name}.json) — transitions
          2. Intent pack (glyphh/intent/data/packs/{name}.json) — NL synonyms

        The intent pack provides the NL-to-function mapping: synonyms and phrases
        for operating actions so the deductive layer can detect operating intent
        from NL queries.
        """
        packs_dir = _DATA_DIR / "packs"
        available = [p.stem for p in packs_dir.glob("*.json")] if packs_dir.exists() else []

        for pack_name in packs:
            pack_path = packs_dir / f"{pack_name}.json"
            if not pack_path.exists():
                raise FileNotFoundError(
                    f"Pack '{pack_name}' not found. Available packs: {available}"
                )
            with open(pack_path) as f:
                pack = json.load(f)

            for transition in pack.get("transitions", []):
                name = transition.get("name")
                directing = transition.get("directing_actions", [])
                operating = transition.get("operating_actions", [])
                prerequisite = transition.get("prerequisite")
                if name and directing and operating and prerequisite:
                    self.add_transition(
                        name=name,
                        directing_actions=directing,
                        operating_actions=operating,
                        prerequisite=prerequisite,
                    )

        # Load NL synonyms from intent packs for operating actions
        self._load_intent_synonyms(packs)

    def _load_intent_synonyms(self, packs: list[str]) -> None:
        """Load NL synonyms for operating actions from intent packs.

        Maps the intent pack's synonym/phrase definitions back to the
        operating action function names registered via transitions. This
        lets the deductive layer recognise NL queries as implying an
        operating action from the registered vocabulary.

        Phrase matching (multi-word) is checked first; single-word
        synonyms serve as fallback.
        """
        intent_packs_dir = Path(__file__).parent.parent / "intent" / "data" / "packs"

        for pack_name in packs:
            intent_path = intent_packs_dir / f"{pack_name}.json"
            if not intent_path.exists():
                continue

            with open(intent_path) as f:
                intent_pack = json.load(f)

            for action_def in intent_pack.get("actions", []):
                canonical = action_def.get("canonical", "")
                if canonical not in self._operating_verbs:
                    continue

                # Multi-word phrases (checked via substring in query)
                for phrase in action_def.get("phrases", []):
                    self._operating_phrases.append((phrase.lower(), canonical))

                # Single-word synonyms (checked via word intersection)
                for syn in action_def.get("synonyms", []):
                    syn_lower = syn.lower()
                    # Multi-word synonyms → also add as phrases
                    if " " in syn_lower:
                        self._operating_phrases.append((syn_lower, canonical))
                    else:
                        self._operating_synonyms[syn_lower] = canonical

        # Sort phrases longest-first for greedy matching
        self._operating_phrases.sort(key=lambda x: len(x[0]), reverse=True)

    def _state_vec(self, state: str) -> np.ndarray:
        """Encode a state label as a deterministic bipolar vector."""
        return generate_symbol(self._seed, f"state_{state}", self._dim)

    def _action_vec(self, action: str) -> np.ndarray:
        """Encode an action name as a deterministic bipolar vector."""
        return generate_symbol(self._seed, f"act_{action}", self._dim)

    def add_transition(
        self,
        name: str,
        directing_actions: list[str],
        operating_actions: list[str],
        prerequisite: str,
        strength: float = 1.0,
    ):
        """Register a transition pattern.

        A transition encodes: "when recent actions included directing_actions
        and the current query implies operating_actions, the prerequisite
        function is likely needed."

        The prerequisite itself is automatically registered as a resolving action —
        when it appears in observed actions, the target is updated to match the
        current state (the mismatch has been resolved).

        Args:
            name: Unique name for this transition.
            directing_actions: Actions that redirect state (e.g. move, copy, create).
            operating_actions: Actions that follow and need the new state (e.g. read, search).
            prerequisite: Function to inject when mismatch detected (e.g. navigate, authenticate).
            strength: Initial Hebbian weight.
        """
        directing_vecs = [self._action_vec(a) for a in sorted(directing_actions)]
        operating_vecs = [self._action_vec(a) for a in sorted(operating_actions)[:8]]

        directing_bundle = bundle(directing_vecs) if len(directing_vecs) > 1 else directing_vecs[0]
        operating_bundle = bundle(operating_vecs) if len(operating_vecs) > 1 else operating_vecs[0]
        transition_vec = bind(directing_bundle, operating_bundle)

        self._transitions[name] = Transition(
            name=name,
            vector=transition_vec,
            prerequisite=prerequisite,
            strength=strength,
        )

        # Track registered action sets
        self._directing_actions.update(directing_actions)
        self._operating_verbs.update(operating_actions)
        # The prerequisite itself resolves the mismatch
        self._resolving_actions.add(prerequisite)

    def observe(
        self,
        state: str,
        actions: list[str],
        targets: Optional[list[str]] = None,
    ):
        """ACCUMULATE: Record what happened this turn.

        When a resolving action (the prerequisite itself) is among the observed
        actions, the target history is updated to the current state — the
        mismatch has been resolved, we arrived at the destination.

        Also tracks temporal state deltas (how much the state changed between
        consecutive turns) and stores state as minimal Glyphs for beam prediction.

        Args:
            state: Current state label after this turn (e.g. context identifier).
            actions: Action names called this turn.
            targets: States that actions were directed toward (e.g. destination
                     contexts, resources that were created).
        """
        new_state_vec = self._state_vec(state)

        # Temporal delta: track how much state changed from previous turn
        if self._state_vector is not None:
            delta = cosine_similarity(new_state_vec, self._state_vector)
            self._state_deltas.append(float(delta))
            if len(self._state_deltas) > 10:
                self._state_deltas = self._state_deltas[-10:]

        self._state_vector = new_state_vec

        # Store state as minimal Glyph for beam prediction
        if self._enable_beam:
            self._store_state_glyph(new_state_vec)

        # If a resolving action was taken, we've arrived at the
        # destination — reset target history to current state so mismatch clears.
        # Old targets are wiped because the transition is complete; new targets
        # from the same turn (below) will still be appended.
        # Also clear temporal stagnation — the state transition was completed.
        if any(a in self._resolving_actions for a in actions):
            self._target_history = [self._state_vec(state)]
            self._last_target = None  # resolved — no outstanding target
            self._state_deltas = []   # stagnation resolved

        if targets:
            for target in targets:
                target_vec = self._state_vec(target)
                self._target_history.append(target_vec)
                self._last_target = target

        # If actions include state-changing ops without explicit targets,
        # the current state itself becomes the target
        if not targets and any(a in self._directing_actions for a in actions):
            if not any(a in self._resolving_actions for a in actions):
                self._target_history.append(self._state_vec(state))

        self._action_history.append(actions)

        # Bound history
        if len(self._target_history) > 10:
            self._target_history = self._target_history[-10:]
        if len(self._action_history) > 10:
            self._action_history = self._action_history[-10:]

    def _get_target_vector(self) -> Optional[np.ndarray]:
        """Decaying superposition of target states. Recent targets dominate."""
        if not self._target_history:
            return None

        n = len(self._target_history)
        pairs = [
            (vec, self._decay ** (n - 1 - i))
            for i, vec in enumerate(self._target_history)
        ]
        return _weighted_bundle(pairs, self._dim)

    def _had_directing_actions(self) -> bool:
        """Check if recent turns included directing actions."""
        for actions in self._action_history[-3:]:
            if any(a in self._directing_actions for a in actions):
                return True
        return False

    def _query_implies_operating(self, query: str) -> bool:
        """Check if the query implies an operating action.

        Uses NL synonyms loaded from intent packs (same domain) to map
        query words back to canonical function names. Three-level check:

          1. Phrase matching — multi-word patterns from pack definitions
          2. Single-word synonyms — pack-defined mappings
          3. Literal function names — exact match against registered operating verbs

        This resolves NL synonyms without the IntentExtractor's generic
        vocabulary override problem.
        """
        query_lower = query.lower()

        # Level 1: phrase matching (longest match first)
        for phrase, canonical in self._operating_phrases:
            if phrase in query_lower:
                return True

        # Level 2: single-word synonym matching from intent packs
        words = set(re.sub(r"[^a-z0-9\s]", "", query_lower).split())
        if words & set(self._operating_synonyms.keys()):
            return True

        # Level 3: literal function-name matching
        return bool(words & self._operating_verbs)

    def _compute_stagnation(self) -> float:
        """Detect state stagnation during directing actions.

        Domain-agnostic: detects when state hasn't changed despite actions
        that typically redirect state (registered via transitions/packs).
        Returns [0, 1] where higher = more stagnant.

        This is a temporal signal — it looks at the TREND of state changes
        across the temporal window, not just the current snapshot.
        """
        if not self._state_deltas or not self._action_history:
            return 0.0

        window = min(self._temporal_window, len(self._state_deltas))
        recent_deltas = self._state_deltas[-window:]
        recent_actions = self._action_history[-window:]

        # Count how many recent turns had directing actions
        directing_turns = sum(
            1 for actions in recent_actions
            if any(a in self._directing_actions for a in actions)
        )

        if directing_turns == 0:
            return 0.0  # No directing actions → stagnation isn't meaningful

        # Average state delta (1.0 = identical = stagnant, ~0.0 = changed)
        avg_delta = sum(recent_deltas) / len(recent_deltas)

        # Stagnation score: high when state is unchanged (avg_delta close to 1.0)
        # AND directing actions have been happening
        stagnation = max(0.0, (avg_delta - 0.3) / 0.7)  # Normalize [0.3, 1.0] → [0, 1]
        directing_ratio = directing_turns / len(recent_actions)

        return min(1.0, stagnation * directing_ratio)

    def _store_state_glyph(self, state_vec: np.ndarray) -> None:
        """Store current state as a minimal Glyph for beam prediction."""
        from datetime import datetime
        from glyphh.core.types import Glyph, Vector

        # Compute a deterministic space_id for this layer's vector space
        space_id = f"deductive_{self._seed}_{self._dim}"

        state_glyph = Glyph(
            identifier=f"state_{len(self._state_glyph_history)}@0#v1",
            name=f"state_{len(self._state_glyph_history)}",
            space_id=space_id,
            global_cortex=Vector(
                data=state_vec.copy(),
                dimension=self._dim,
                space_id=space_id,
            ),
            layers={},
            timestamp=datetime.now(),
        )
        self._state_glyph_history.append(state_glyph)
        if len(self._state_glyph_history) > 10:
            self._state_glyph_history = self._state_glyph_history[-10:]

    def _compute_beam_divergence(self, current_state: str) -> float:
        """Predict expected next state via beam search, compare with actual.

        Returns divergence score [0, 1] — higher means the state trajectory
        suggests we should be somewhere different from where we actually are.

        Requires ≥3 turns of history (2 for deltas + 1 for prediction).
        Domain-agnostic — works with any state labels.
        """
        if (not self._enable_beam
                or self._beam_predictor is None
                or len(self._state_glyph_history) < 3):
            return 0.0

        try:
            pred_result = self._beam_predictor.predict(
                history=self._state_glyph_history,
                time_intervals=1,
                hierarchy_level="cortex",
            )
        except Exception:
            return 0.0

        if not pred_result.predictions or pred_result.predictions[0].confidence < 0.2:
            return 0.0

        predicted_vec = pred_result.predictions[0].vector.data
        actual_vec = self._state_vec(current_state)

        sim = cosine_similarity(predicted_vec, actual_vec)
        divergence = max(0.0, 1.0 - sim)

        return divergence * pred_result.predictions[0].confidence

    def deduce(
        self,
        query: str,
        current_state: str,
    ) -> dict:
        """COMPARE + RESOLVE: Detect state mismatch, identify prerequisites.

        Three-level mismatch detection (checked in order):
          1. Explicit target-vs-state cosine divergence (classic deductive)
          2. Temporal stagnation: state unchanged despite directing actions
          3. Beam divergence: predicted state trajectory diverges from actual

        All three are domain-agnostic — they use registered transitions and
        observed state labels, not hard-coded domain knowledge.

        Args:
            query: The user's natural language query for this turn.
            current_state: Current state label (e.g. context identifier).

        Returns:
            {
                "prerequisites": ["navigate"],
                "mismatch_score": 0.34,
                "target": "state_y",
                "confidence": 0.78,
            }
        """
        result = {
            "prerequisites": [],
            "mismatch_score": 0.0,
            "target": None,
            "confidence": 0.0,
        }

        current_vec = self._state_vec(current_state)

        # ── Level 1: Explicit target-vs-state mismatch ──
        target_vec = self._get_target_vector()
        if target_vec is not None:
            sim = cosine_similarity(target_vec, current_vec)
            mismatch = 1.0 - sim
        else:
            mismatch = 0.0

        result["mismatch_score"] = round(mismatch, 4)

        # ── Level 2: Temporal stagnation fallback ──
        # When explicit mismatch is low (targets not properly fed, or state=target),
        # check if state has been stagnant despite directing actions.
        if mismatch <= self._mismatch_threshold:
            stagnation = self._compute_stagnation()
            if stagnation > 0.5:
                mismatch = stagnation
                result["mismatch_score"] = round(mismatch, 4)

        # ── Level 3: Beam divergence fallback ──
        # When both explicit and stagnation are low, check if beam-predicted
        # state trajectory diverges from actual state.
        if mismatch <= self._mismatch_threshold:
            beam_div = self._compute_beam_divergence(current_state)
            if beam_div > 0.3:
                mismatch = beam_div
                result["mismatch_score"] = round(mismatch, 4)

        # Gate: only trigger when all conditions are met
        if mismatch <= self._mismatch_threshold:
            return result

        if not self._had_directing_actions():
            return result

        if not self._query_implies_operating(query):
            return result

        # ── RESOLVE: match against transition library ──
        recent_action_vecs = []
        for actions in self._action_history[-2:]:
            for a in actions:
                recent_action_vecs.append(self._action_vec(a))
        if not recent_action_vecs:
            return result

        recent_pattern = bundle(recent_action_vecs) if len(recent_action_vecs) > 1 else recent_action_vecs[0]

        # Encode query's operating intent
        query_words = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split())
        query_action_vecs = [
            self._action_vec(w) for w in query_words
            if w in self._operating_verbs
        ]
        if query_action_vecs:
            query_pattern = bundle(query_action_vecs) if len(query_action_vecs) > 1 else query_action_vecs[0]
        else:
            query_pattern = recent_pattern

        best_transition: Optional[str] = None
        best_score = 0.0

        for name, transition in self._transitions.items():
            delta = bind(recent_pattern, query_pattern)
            sim_t = cosine_similarity(delta, transition.vector)
            weighted = abs(sim_t) * transition.strength
            combined = weighted * 0.4 + mismatch * 0.6

            if combined > best_score and combined > self._min_confidence:
                best_score = combined
                best_transition = name

        if best_transition:
            transition = self._transitions[best_transition]
            result["prerequisites"] = [transition.prerequisite]
            result["confidence"] = round(best_score, 4)
            result["target"] = self._last_target

            # One-shot deduction: once the prerequisite is communicated, clear
            # the target so the same deduction doesn't fire on consecutive turns.
            # The caller decides whether to act on it; either way, the deductive
            # cycle resets. New directing actions will rebuild target state.
            self._target_history = [self._state_vec(current_state)]
            self._last_target = None
            self._state_deltas = []  # stagnation resolved by firing

        return result

    def confirm(self, was_correct: bool, prerequisite: str):
        """Hebbian reinforcement: strengthen or weaken the transition that fired."""
        for transition in self._transitions.values():
            if transition.prerequisite == prerequisite:
                if was_correct:
                    transition.strengthen()
                else:
                    transition.weaken()

    @property
    def transitions(self) -> dict[str, Transition]:
        """Access all registered transitions."""
        return dict(self._transitions)

    @property
    def depth(self) -> int:
        """Number of observation steps recorded."""
        return len(self._action_history)

    def reset(self):
        """Reset state for a new conversation. Transitions persist."""
        self._state_vector = None
        self._target_history = []
        self._last_target = None
        self._action_history = []
        # Reset temporal tracking
        self._state_deltas = []
        self._state_glyph_history = []
        # Reset strengths but keep transitions
        for transition in self._transitions.values():
            transition.strength = 1.0
            transition.fire_count = 0
