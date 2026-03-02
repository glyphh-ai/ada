"""
CognitiveLoop — HDC + LLM multi-turn tool calling.

The loop runs per turn:
  1. PERCEIVE  — LLM-primary classification (or IntentExtractor fallback)
  2. RECALL    — IdeaSpace: find similar past ideas
  3. DEDUCE    — DeductiveLayer: check state mismatch → prerequisites
  4. PREDICT   — InductiveLayer: learned pattern classification
  5. RESOLVE   — Map intent → function names (skipped when LLM resolves directly)
  6. SLOT      — SlotExtractor + LLM-provided args merged
  7. CONVERGE  — Confidence gate (HDC + LLM blended)
  8. DECIDE    — Emit CALL (confident + slots filled) or ASK
  9. RECORD    — Store this turn's idea-glyph in memory

With LLM engine: SchemaIntentClassifier does PERCEIVE+RESOLVE+partial SLOT
in a single call. HDC IntentCache learns from LLM decisions for fast repeat
patterns. Without LLM engine: falls back to IntentExtractor (backward compat).

No domain knowledge in the SDK. The config is application.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from glyphh.state import ConversationState, DeductiveLayer, InductiveLayer

from .domain import DomainConfig
from .idea import IdeaSpace, Idea
from .slots import SlotExtractor

if TYPE_CHECKING:
    from glyphh.llm import LLMEngine

logger = logging.getLogger(__name__)

# ── Lightweight keyword extraction (no IntentExtractor needed) ──
_STOP_WORDS: frozenset[str] = frozenset([
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "want", "think",
    "use", "call", "run", "execute", "tool", "function", "api",
    "given", "using", "by", "if", "so", "as", "at", "into",
    "are", "was", "were", "will", "would", "could", "should",
    "may", "might", "shall", "need", "does", "did", "am",
    "you", "your", "he", "she", "they", "us", "his", "her",
    "let", "like", "any", "all", "some", "each", "every",
    "there", "here", "via", "per", "after", "before", "over",
    "again", "more", "much", "many", "less", "next", "last",
    "other", "too", "hey", "ok", "out", "back",
])

_KEYWORD_RE = re.compile(r"[^\w\s#@._\-]")


def _extract_keywords(query: str) -> str:
    """Extract keywords with stop words removed. No vocab files needed."""
    cleaned = _KEYWORD_RE.sub("", query.lower())
    return " ".join(w for w in cleaned.split() if w not in _STOP_WORDS)


@dataclass
class StepResult:
    """Result of one cognitive loop turn."""

    action: str                            # "CALL" or "ASK"
    calls: list[dict] = field(default_factory=list)   # [{func_name: {args}}]
    missing: list[str] = field(default_factory=list)  # What's unclear (ASK)
    confidence: float = 0.0
    signals: dict = field(default_factory=dict)        # Debug info


class CognitiveLoop:
    """HDC + LLM cognitive loop for multi-turn tool calling.

    Domain-agnostic: all domain knowledge comes from the DomainConfig.
    HDC sub-models handle pattern matching, state tracking, and memory.
    Optional LLM engine handles NL understanding at specific pipeline points.
    Emits CALL when confident, ASK when unsure. Never guesses.

    Usage:
        from glyphh.llm import LLMEngine

        config = DomainConfig.from_file("domain/my_domain.json")
        engine = LLMEngine()
        loop = CognitiveLoop(
            packs=["my_pack"],
            domain_config=config,
            llm_engine=engine,
        )
        loop_id = loop.begin(functions=func_defs, initial_state=normalized_state)

        for query in turn_queries:
            result = loop.step(query)
            if result.action == "CALL":
                execute(result.calls)
            else:
                ask_user(result.missing)

            loop.confirm(was_correct, correct_outcome)

        loop.end()
    """

    def __init__(
        self,
        packs: list[str] | None = None,
        domain_config: DomainConfig | None = None,
        dimension: int = 10000,
        confidence_threshold: float = 0.25,
        llm_engine: LLMEngine | None = None,
    ):
        self._dim = dimension
        self._threshold = confidence_threshold
        self._config = domain_config
        self._llm = llm_engine

        # LLM-primary intent classification (when LLM available)
        if llm_engine is not None:
            from .schema_classifier import SchemaIntentClassifier
            self._classifier = SchemaIntentClassifier(
                llm_engine=llm_engine,
                dimension=dimension,
            )
        else:
            self._classifier = None

        # No IntentExtractor dependency — keyword extraction is inlined
        # via _extract_keywords(). LLM handles all intent classification.

        # Sub-models
        self.idea_space = IdeaSpace(dimension=dimension)
        self.deductive = DeductiveLayer(
            dimension=dimension, seed=89, packs=packs,
            temporal_window=3, enable_beam=True,
        )
        self.inductive = InductiveLayer(
            dimension=dimension, seed=97, packs=packs,
        )
        self.conv_state = ConversationState(
            dimension=dimension, seed=73, decay=0.75,
        )
        self.slot_extractor = SlotExtractor(config=domain_config)

        # Function resolution from config
        self._action_to_func = domain_config.action_to_func if domain_config else {}
        self._available_funcs: dict[str, dict] = {}  # name → schema

        # State format from config (no hardcoded separators or defaults)
        self._sep = domain_config.state_format.separator if domain_config else "/"
        self._default_primary = domain_config.state_format.default_primary if domain_config else ""
        self._tree_key = domain_config.state_format.tree_key if domain_config else "_tree"

        # Loop state
        self._loop_id: str | None = None
        self._turn: int = 0
        self._recent_actions: list[str] = []
        self._state: dict = {}
        self._last_step_idea_vec: np.ndarray | None = None
        self._last_step_funcs: list[str] = []

    def begin(
        self,
        functions: list[dict],
        initial_state: dict | None = None,
    ) -> str:
        """Start a new cognitive loop. Returns loop_id.

        Args:
            functions: Available function schemas [{name, description, parameters}]
            initial_state: Pre-normalized state dict with keys:
                           primary, collections, and optionally a tree key
        """
        self._loop_id = str(uuid.uuid4())[:8]
        self._turn = 0
        self._recent_actions = []

        # Register available functions
        self._available_funcs = {f["name"]: f for f in functions}

        # State is pre-normalized by the caller — no parsing here
        if initial_state:
            self._state = initial_state
        else:
            self._state = {
                "primary": self._default_primary,
                "collections": {},
            }

        # Configure LLM-primary classifier with function schemas
        if self._classifier is not None:
            self._classifier.configure(functions, self._action_to_func)

        # Observe initial state in deductive layer
        self.deductive.observe(
            state=self._state.get("primary", self._default_primary),
            actions=[],
        )

        return self._loop_id

    def step(self, query: str) -> StepResult:
        """Process one turn through the cognitive loop.

        Pipeline:
          PERCEIVE → RECALL → DEDUCE → RESOLVE → SLOT → CONVERGE → DECIDE → RECORD
        """
        signals: dict[str, Any] = {}

        # ── 1. PERCEIVE: intent classification ──
        llm_functions: list[str] = []
        llm_arguments: dict[str, dict] = {}
        llm_confidence: float = 0.0

        if self._classifier is not None:
            # LLM-primary path: function schemas define the intent space
            classification = self._classifier.classify(
                query=query,
                state=self._state,
                recent_actions=self._recent_actions[-3:],
            )
            signals["classification"] = classification
            signals["classification_source"] = classification.get("source", "llm")

            llm_functions = classification.get("functions", [])
            llm_arguments = classification.get("arguments", {})
            llm_confidence = classification.get("confidence", 0.0)

            # Back-derive action for downstream compatibility (IdeaEncoder needs it)
            func_to_action = {v: k for k, v in self._action_to_func.items()}
            action = func_to_action.get(llm_functions[0], "") if llm_functions else ""
            target = ""

            # Lightweight keyword extraction — inlined, no IntentExtractor needed
            keywords = _extract_keywords(query)
            intent = {"action": action, "target": target, "domain": "", "keywords": keywords}
            signals["intent"] = intent
        else:
            # No-LLM fallback: keyword-only extraction (no IntentExtractor)
            keywords = _extract_keywords(query)
            intent = {"action": "", "target": "", "domain": "", "keywords": keywords}
            signals["intent"] = intent
            action = ""
            target = ""

        # ── 2. RECALL: episodic memory lookup ──
        idea_vec = self.idea_space.encoder.encode_from_query(
            query=query,
            intent=intent,
            state=self._state.get("primary", self._default_primary),
            recent_actions=self._recent_actions[-3:],
        )
        self._last_step_idea_vec = idea_vec

        recalled = self.idea_space.recall(idea_vec, top_k=3)
        signals["recall"] = [
            (idea.label or "?", round(sim, 3))
            for idea, sim in recalled
        ]

        # ── 3. DEDUCE: state mismatch → prerequisites ──
        deduction = self.deductive.deduce(
            query=query,
            current_state=self._state.get("primary", self._default_primary),
        )
        signals["deduction"] = deduction

        # ── 4. RESOLVE: map intent → function(s) ──
        if llm_functions and llm_confidence > 0.3:
            # LLM already resolved functions — validate and apply rules
            functions = [f for f in llm_functions if f in self._available_funcs]
            functions = self._apply_exclusion_rules(functions)
            signals["resolve_source"] = "llm_direct"
        else:
            # Traditional path: action → function mapping
            functions = self._resolve_functions(
                action, target, query, deduction, recalled,
            )
            signals["resolve_source"] = "rule_based"
        signals["resolved_functions"] = functions

        if not functions:
            # No function could be resolved — ASK
            self._record_turn([], idea_vec)
            return StepResult(
                action="ASK",
                missing=[f"Could not resolve action '{action}' to a function"],
                confidence=0.0,
                signals=signals,
            )

        # ── 5. SLOT: extract arguments ──
        # Pass full state through — the slot extractor reads whatever
        # hints the caller has injected (no hardcoded key names here)
        slot_state = dict(self._state)
        slot_state["_query_lower"] = query.lower()
        filled = self.slot_extractor.extract(
            query, functions, self._available_funcs, slot_state,
        )

        # Merge LLM-provided arguments (from SchemaIntentClassifier)
        if llm_arguments:
            for fname, args in llm_arguments.items():
                if fname in self._available_funcs and isinstance(args, dict):
                    filled.setdefault(fname, {}).update(args)
            signals["llm_arguments"] = llm_arguments

        # Check for missing required slots
        missing = self.slot_extractor.missing_required(
            functions, self._available_funcs, filled,
        )

        signals["filled_slots"] = filled
        signals["missing_slots"] = missing

        # ── 6. CONFIDENCE GATE ──
        confidence = self._compute_confidence(
            intent, functions, deduction, recalled, filled, missing,
        )

        # Blend with LLM classification confidence
        if self._classifier is not None and llm_confidence > 0:
            confidence = 0.4 * confidence + 0.6 * llm_confidence

        signals["confidence"] = confidence

        if confidence < self._threshold and not recalled:
            # Low confidence AND no episodic memory match — ASK
            missing_list = []
            for fname, params in missing.items():
                for p in params:
                    missing_list.append(f"{fname}.{p}")
            self._record_turn([], idea_vec)
            return StepResult(
                action="ASK",
                missing=missing_list if missing_list else ["Low confidence"],
                confidence=confidence,
                signals=signals,
            )

        # ── 7. BUILD CALLS ──
        calls = []
        for fname in functions:
            args = filled.get(fname, {})
            calls.append({fname: args})

        # ── 8. RECORD: store idea + update state ──
        self._record_turn(functions, idea_vec)
        self._update_state(calls)

        self._last_step_funcs = functions

        return StepResult(
            action="CALL",
            calls=calls,
            confidence=confidence,
            signals=signals,
        )

    def confirm(
        self,
        was_correct: bool,
        correct_outcome: list[dict] | None = None,
    ) -> None:
        """Hebbian reinforcement on the last step.

        If correct: strengthen recalled idea + pathways.
        If incorrect + correct_outcome: store new idea for future recall.
        """
        # Reinforce episodic memory
        self.idea_space.reinforce_last(was_correct)

        # Reinforce LLM intent cache
        if self._classifier is not None:
            self._classifier.confirm(was_correct)

        # Reinforce deductive layer for trigger function
        trigger_func = self._get_trigger_func()
        if trigger_func and trigger_func in self._last_step_funcs:
            self.deductive.confirm(was_correct, trigger_func)

        # Reinforce inductive layer
        if trigger_func:
            label_needed = f"{trigger_func}_needed"
            label_not = f"{trigger_func}_not_needed"
            if trigger_func in self._last_step_funcs:
                self.inductive.confirm(was_correct=was_correct, label=label_needed)
            else:
                self.inductive.confirm(
                    was_correct=trigger_func not in (self._last_step_funcs or []),
                    label=label_not,
                )

        # If incorrect and we have the correct answer, store it as a new idea
        if not was_correct and correct_outcome and self._last_step_idea_vec is not None:
            self.idea_space.store(
                self._last_step_idea_vec,
                correct_outcome,
                strength=1.0,
                label="corrected",
            )

    def end(self) -> None:
        """End the loop. Working state clears. Long-term ideas persist."""
        self._loop_id = None
        self._turn = 0
        self._recent_actions = []
        self._state = {}
        self._last_step_idea_vec = None
        self._last_step_funcs = []
        # Reset working state but keep long-term memory
        self.deductive.reset()
        self.conv_state.reset()

    # ── Internal ──

    def _get_trigger_func(self) -> str | None:
        """Get the trigger function from config (e.g. navigation trigger)."""
        if self._config and self._config.navigation_trigger:
            return self._config.navigation_trigger.get("trigger_func")
        return None

    def _resolve_functions(
        self,
        action: str,
        target: str,
        query: str,
        deduction: dict,
        recalled: list[tuple[Idea, float]],
    ) -> list[str]:
        """Map intent → ordered list of function names to call.

        Resolution order:
        1. Strong episodic recall (similarity > 0.5) → use stored outcome
        2. Direct action→func mapping from config
        3. Multi-action detection from config keywords
        4. Deductive prerequisites prepended
        5. Inductive prediction
        6. Exclusion rules
        7. Trigger suppression (from config)
        """
        functions: list[str] = []

        # 1. Check episodic recall
        if recalled and recalled[0][1] > 0.5:
            best_idea, sim = recalled[0]
            for call in best_idea.outcome:
                if isinstance(call, dict):
                    functions.extend(call.keys())
            if functions:
                return functions

        # 2. Direct action → function mapping from config
        primary_func = self._action_to_func.get(action)
        action_is_trigger = False
        trigger_func = self._get_trigger_func()
        if primary_func and primary_func in self._available_funcs:
            functions.append(primary_func)
            if primary_func == trigger_func:
                action_is_trigger = True

        # 3. Multi-action detection from config keywords
        query_lower = query.lower()
        additional = self._detect_additional_actions(query_lower, functions)
        trigger_has_keyword = trigger_func in additional if trigger_func else False
        functions.extend(additional)

        # Track whether the trigger has keyword/action evidence
        trigger_has_evidence = action_is_trigger or trigger_has_keyword

        # 4. Deductive prerequisites
        if deduction.get("prerequisites"):
            for prereq in deduction["prerequisites"]:
                if prereq in self._available_funcs and prereq not in functions:
                    functions.insert(0, prereq)

        # 5. Inductive prediction for trigger function
        if (trigger_func and trigger_func not in functions
                and self._available_funcs.get(trigger_func)):
            query_tokens = " ".join(
                re.sub(r"[^a-z0-9\s]", "", query_lower).split()[:20]
            )
            ind_result = self.inductive.predict(features={
                "query_tokens": query_tokens,
            })
            label_needed = f"{trigger_func}_needed"
            if (ind_result["label"] == label_needed
                    and ind_result["confidence"] > 0.15):
                functions.insert(0, trigger_func)

        # 6. Apply exclusion rules (more specific function removes generic one)
        functions = self._apply_exclusion_rules(functions)

        # 7. Suppress trigger from deductive/inductive if no keyword evidence
        #    Predictions without query support are unreliable.
        if (trigger_func and trigger_func in functions
                and not trigger_has_evidence
                and not self._trigger_has_keyword_support(trigger_func, query_lower)):
            functions.remove(trigger_func)

        # 8. Suppress trigger via config-driven context rules
        if (trigger_func and trigger_func in functions
                and not action_is_trigger
                and self._should_suppress_trigger(query_lower)):
            functions.remove(trigger_func)

        return functions

    def _apply_exclusion_rules(self, functions: list[str]) -> list[str]:
        """Remove functions that are superseded by more specific ones.

        Reads exclusion_rules from config: {specific_func: [generic_funcs_to_remove]}.
        """
        if not self._config or not self._config.exclusion_rules:
            return functions

        to_remove: set[str] = set()
        for specific_func, generic_funcs in self._config.exclusion_rules.items():
            if specific_func in functions:
                for gf in generic_funcs:
                    if gf in functions:
                        to_remove.add(gf)

        if to_remove:
            functions = [f for f in functions if f not in to_remove]

        return functions

    def _should_suppress_trigger(self, query_lower: str) -> bool:
        """Check if the trigger function should be suppressed.

        All suppression logic is driven by the domain config's
        trigger_suppression section. The SDK has zero domain knowledge.
        """
        if not self._config:
            return False

        ts = self._config.trigger_suppression

        # Strong intent patterns — never suppress when these match
        for pat in ts.strong_intent_patterns:
            if re.search(pat, query_lower):
                return False

        # Current context phrases — suppress when these match
        for phrase in ts.current_context_phrases:
            if phrase in query_lower:
                return True

        # Primary match — suppress if query mentions the current primary by name
        if ts.primary_match_pattern:
            primary = self._state.get("primary", self._default_primary)
            sep = self._sep
            primary_name = primary.rstrip(sep).split(sep)[-1].lower() if primary else ""
            if primary_name:
                pat = ts.primary_match_pattern.replace("{primary_name}", re.escape(primary_name))
                if re.search(pat, query_lower):
                    return True

        # File collections — suppress if all quoted strings are in these collections
        if ts.file_collections:
            collections = self._state.get("collections", {})
            all_items: set[str] = set()
            for coll_name in ts.file_collections:
                all_items.update(collections.get(coll_name, []))
            if all_items:
                quoted = re.findall(r"""['"]([^'"]+)['"]""", query_lower)
                if quoted and all(q in all_items for q in quoted):
                    return True

        return False

    def _trigger_has_keyword_support(
        self,
        trigger_func: str,
        query_lower: str,
    ) -> bool:
        """Check if the trigger function has any keyword match in the query.

        Used to gate deductive/inductive predictions: if no keyword
        evidence supports the trigger, the prediction is unreliable.
        """
        if not self._config:
            return False
        patterns = self._config.multi_action_keywords.get(trigger_func, [])
        for pat in patterns:
            if re.search(pat, query_lower):
                return True
        return False

    def _detect_additional_actions(
        self,
        query_lower: str,
        already: list[str],
    ) -> list[str]:
        """Detect multiple actions in a single query using config keywords."""
        if not self._config:
            return []

        additional = []
        for fname, patterns in self._config.multi_action_keywords.items():
            if fname in already or fname not in self._available_funcs:
                continue
            for pat in patterns:
                if re.search(pat, query_lower):
                    additional.append(fname)
                    break

        return additional

    def _compute_confidence(
        self,
        intent: dict,
        functions: list[str],
        deduction: dict,
        recalled: list[tuple[Idea, float]],
        filled: dict[str, dict],
        missing: dict[str, list[str]],
    ) -> float:
        """Compute overall confidence from all signals.

        Factors:
        - Intent action was resolved to a function
        - Episodic recall similarity
        - Slot fill completeness
        - Deductive confidence
        """
        score = 0.0

        # Base: did we resolve at least one function?
        if functions:
            score += 0.3

        # Episodic recall: strong match → high confidence
        if recalled:
            best_sim = recalled[0][1]
            score += min(0.3, best_sim * 0.4)

        # Deductive: prerequisites detected with confidence
        if deduction.get("prerequisites"):
            score += min(0.15, deduction.get("confidence", 0) * 0.2)

        # Slot completeness: fewer missing required → higher confidence
        total_missing = sum(len(v) for v in missing.values())
        if total_missing == 0 and functions:
            score += 0.25
        elif total_missing <= 1:
            score += 0.10

        return min(1.0, score)

    def _record_turn(
        self,
        functions: list[str],
        idea_vec: np.ndarray,
    ) -> None:
        """Record this turn in working memory."""
        self._turn += 1
        self.idea_space.tick()

        if functions:
            self._recent_actions = (
                self._recent_actions + functions
            )[-10:]  # Keep last 10

        # Observe in deductive layer
        self.deductive.observe(
            state=self._state.get("primary", self._default_primary),
            actions=functions,
        )

    def _update_state(self, calls: list[dict]) -> None:
        """Update internal state using declarative effects from config."""
        if not self._config:
            return

        for call in calls:
            for fname, args in call.items():
                if not isinstance(args, dict):
                    continue
                effects = self._config.state_effects.get(fname, [])
                for effect in effects:
                    self._apply_effect(effect, args)

    def _apply_effect(self, effect: Any, args: dict) -> None:
        """Apply a single state effect.

        Uses the separator from config.state_format for hierarchical
        state values — no hardcoded path format.
        """
        arg_value = args.get(effect.from_arg, "")
        if not arg_value:
            return

        sep = self._sep

        if effect.op == "set_primary":
            if arg_value == effect.parent_keyword:
                # Navigate up one level in the state hierarchy
                primary = self._state.get("primary", self._default_primary)
                parts = primary.rstrip(sep).split(sep)
                self._state["primary"] = sep.join(parts[:-1]) or self._default_primary
            else:
                # Navigate into a child
                primary = self._state.get("primary", self._default_primary)
                self._state["primary"] = primary.rstrip(sep) + sep + str(arg_value)
            if effect.refresh_collections:
                self._refresh_collections()

        elif effect.op == "add_to_collection":
            collections = self._state.get("collections", {})
            coll = collections.get(effect.collection, [])
            if arg_value not in coll:
                coll.append(arg_value)
                collections[effect.collection] = coll
                self._state["collections"] = collections

        elif effect.op == "remove_from_collection":
            collections = self._state.get("collections", {})
            coll = collections.get(effect.collection, [])
            if arg_value in coll:
                coll.remove(arg_value)

    def _refresh_collections(self) -> None:
        """Refresh collections from the tree for the current primary state."""
        primary = self._state.get("primary", self._default_primary)
        tree = self._state.get(self._tree_key, {})
        node = tree.get(primary, {})
        collections = self._state.get("collections", {})
        # Refresh any collections that have corresponding tree keys
        for key in list(collections.keys()):
            if key in node:
                collections[key] = list(node[key])  # copy to avoid mutation
        self._state["collections"] = collections

