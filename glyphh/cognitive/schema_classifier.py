"""Intent classification from function schemas — three-tier waterfall.

Classification pipeline:
  1. ModelScorer    (domain HDC encoder, sub-ms — when provided)
  2. IntentCache    (learned HDC patterns from previous LLM decisions)
  3. LLM           (structured generation, full classification)

When a ModelScorer is provided, it acts as the primary classification path.
The LLM handles argument extraction and arbitrates uncertain scorer results.
Over time, the IntentCache learns from confirmed decisions.

Without a ModelScorer, the pipeline is: IntentCache → LLM (original behavior).
Without an LLM, the pipeline is: ModelScorer → IntentCache (HDC-only).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .intent_cache import IntentCache
from .model_scorer import ModelScorer, ScorerResult

if TYPE_CHECKING:
    from glyphh.llm.engine import LLMEngine

logger = logging.getLogger(__name__)


class SchemaIntentClassifier:
    """Three-tier intent classifier: ModelScorer → IntentCache → LLM.

    Usage:
        classifier = SchemaIntentClassifier(
            llm_engine=engine,       # optional
            model_scorer=scorer,     # optional
            dimension=10000,
        )
        classifier.configure(functions, action_to_func)  # at begin() time

        result = classifier.classify(query, state, recent_actions)
        # result = {functions, arguments, confidence, source}

        classifier.confirm(correct=True)  # Hebbian reinforcement
    """

    # Confidence tiers for model scorer gating
    _HIGH_CONFIDENCE = 0.50     # Trust scorer directly, LLM extracts args only
    _UNCERTAIN_CONFIDENCE = 0.20  # LLM arbitrates (both scorer and LLM vote)

    def __init__(
        self,
        llm_engine: LLMEngine | None = None,
        dimension: int = 10000,
        cache_threshold: float = 0.85,
        model_scorer: ModelScorer | None = None,
    ):
        self._llm = llm_engine
        self._scorer = model_scorer
        self._cache = IntentCache(
            dimension=dimension,
            threshold=cache_threshold,
        )

        # Set at configure() time
        self._func_schemas: dict[str, dict] = {}
        self._action_to_func: dict[str, str] = {}
        self._system_prompt: str = ""
        self._classify_tools: list[dict] = []
        self._configured = False

    def configure(
        self,
        functions: list[dict[str, Any]],
        action_to_func: dict[str, str],
    ) -> None:
        """Configure the classifier with available function schemas.

        Called from CognitiveLoop.begin(). Builds the LLM system prompt
        and tool schema from the function definitions. Called once per
        session, not per query.

        Args:
            functions: Function schemas [{name, description, parameters}]
            action_to_func: Mapping from action verbs to function names
        """
        from glyphh.llm.prompts import SCHEMA_CLASSIFY_SYSTEM
        from glyphh.llm.structured import (
            build_classify_tools,
            format_function_descriptions,
        )

        self._func_schemas = {f["name"]: f for f in functions}
        self._action_to_func = action_to_func

        # Build the system prompt — concise, function descriptions
        # are provided via the tool schemas themselves
        self._system_prompt = (
            "You are a function-calling assistant. "
            "Given the user's query and current context, call the appropriate function(s). "
            "If the query requires multiple function calls, make all of them. "
            "If the query doesn't match any function, do not make any calls."
        )

        # Pass the actual function schemas as tools — Qwen3 naturally
        # calls them directly, which is more reliable than a meta-tool
        self._classify_tools = []
        for func in functions:
            self._classify_tools.append({
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                },
            })

        # Configure model scorer with function definitions
        if self._scorer is not None:
            self._scorer.configure(functions)

        # Clear cache from previous session
        self._cache.clear()
        self._configured = True

    def classify(
        self,
        query: str,
        state: dict[str, Any],
        recent_actions: list[str],
    ) -> dict[str, Any]:
        """Classify intent via three-tier waterfall: ModelScorer → Cache → LLM.

        Returns:
            {
                "functions": ["func_name", ...],
                "arguments": {"func_name": {"param": "value"}},
                "confidence": 0.0-1.0,
                "source": "model_scorer" | "model_scorer+llm" | "hdc_cache" | "llm",
            }
        """
        if not self._configured:
            return {
                "functions": [],
                "arguments": {},
                "confidence": 0.0,
                "source": "unconfigured",
            }

        state_primary = state.get("primary", "")

        # ── Tier 1: Model scorer (domain HDC, sub-ms) ──
        if self._scorer is not None:
            scorer_result = self._scorer.score(query)

            if scorer_result.is_irrelevant:
                logger.debug("ModelScorer: irrelevant (conf=%.3f)", scorer_result.confidence)
                return {
                    "functions": [],
                    "arguments": {},
                    "confidence": scorer_result.confidence,
                    "source": "model_scorer_irrelevant",
                    "all_scores": scorer_result.all_scores,
                }

            if scorer_result.confidence >= self._HIGH_CONFIDENCE:
                # HIGH: trust scorer for routing, LLM extracts args only
                logger.debug(
                    "ModelScorer HIGH (conf=%.3f): %s",
                    scorer_result.confidence, scorer_result.functions,
                )
                result = {
                    "functions": scorer_result.functions,
                    "arguments": scorer_result.arguments,
                    "confidence": scorer_result.confidence,
                    "source": "model_scorer",
                    "all_scores": scorer_result.all_scores,
                }
                # Use LLM for argument extraction if available
                if self._llm is not None and scorer_result.functions:
                    llm_args = self._llm_extract_args(
                        query, scorer_result.functions, state, recent_actions,
                    )
                    if llm_args:
                        result["arguments"] = llm_args
                return result

            if scorer_result.confidence >= self._UNCERTAIN_CONFIDENCE:
                # UNCERTAIN: LLM arbitrates (both score)
                logger.debug(
                    "ModelScorer UNCERTAIN (conf=%.3f): %s",
                    scorer_result.confidence, scorer_result.functions,
                )
                if self._llm is not None:
                    llm_result = self._llm_classify(query, state, recent_actions)
                    # Check overlap between scorer and LLM
                    scorer_set = set(scorer_result.functions)
                    llm_set = set(llm_result.get("functions", []))
                    overlap = scorer_set & llm_set
                    if overlap:
                        # Both agree — boost confidence
                        llm_result["confidence"] = min(
                            1.0,
                            max(llm_result["confidence"], scorer_result.confidence) + 0.1,
                        )
                        llm_result["source"] = "model_scorer+llm"
                    else:
                        llm_result["source"] = "llm"
                    llm_result["all_scores"] = scorer_result.all_scores
                    return llm_result
                else:
                    # No LLM — return scorer result as-is
                    return {
                        "functions": scorer_result.functions,
                        "arguments": scorer_result.arguments,
                        "confidence": scorer_result.confidence,
                        "source": "model_scorer",
                        "all_scores": scorer_result.all_scores,
                    }

            # LOW confidence from scorer — fall through to cache/LLM
            logger.debug(
                "ModelScorer LOW (conf=%.3f), falling through",
                scorer_result.confidence,
            )

        # ── Tier 2: IntentCache (learned HDC patterns, fast path) ──
        cached = self._cache.lookup(query, state_primary)
        if cached is not None:
            logger.debug("IntentCache hit (sim=%.3f)", cached.get("cache_similarity", 0))
            cached["source"] = "hdc_cache"
            return cached

        # ── Tier 3: LLM classification (generative, full) ──
        if self._llm is not None:
            result = self._llm_classify(query, state, recent_actions)

            # Store in cache for future fast path
            if result["confidence"] > 0.3:
                self._cache.store(query, state_primary, {
                    "functions": result["functions"],
                    "arguments": result["arguments"],
                    "confidence": result["confidence"],
                })

            result["source"] = "llm"
            return result

        # ── No classification source available ──
        return {
            "functions": [],
            "arguments": {},
            "confidence": 0.0,
            "source": "none",
        }

    def confirm(self, correct: bool) -> None:
        """Hebbian reinforcement on the most recently cached entry."""
        self._cache.reinforce(correct)

    @property
    def cache_size(self) -> int:
        """Number of entries in the HDC cache."""
        return self._cache.size

    def _llm_extract_args(
        self,
        query: str,
        functions: list[str],
        state: dict[str, Any],
        recent_actions: list[str],
    ) -> dict[str, dict]:
        """Targeted LLM call for argument extraction only.

        Used when the ModelScorer has high confidence on routing but
        doesn't extract arguments. Constrains the tool set to only
        the scorer-selected functions for faster/cheaper LLM call.
        """
        # Build constrained tool set — only the functions the scorer selected
        constrained_tools = [
            t for t in self._classify_tools
            if t.get("function", {}).get("name") in functions
        ]
        if not constrained_tools:
            return {}

        state_summary = self._summarize_state(state)
        recent_str = ", ".join(recent_actions) if recent_actions else "none"

        system = (
            "You are a function-calling assistant. "
            "The function to call has already been determined. "
            "Extract the correct arguments from the user's query."
        )
        user_prompt = (
            f"Context: {state_summary}\n"
            f"Recent: {recent_str}\n"
            f"Query: {query}"
        )

        try:
            result = self._llm.structured_generate(
                system=system,
                user=user_prompt,
                tools=constrained_tools,
                max_tokens=512,
            )
            data = result.data
            arguments: dict[str, dict] = {}

            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    fname = tc.get("name", "")
                    fargs = tc.get("arguments", {})
                    if fname in self._func_schemas and isinstance(fargs, dict):
                        arguments[fname] = fargs
            elif "name" in data and data["name"] in self._func_schemas:
                fargs = data.get("arguments", {})
                if isinstance(fargs, dict):
                    arguments[data["name"]] = fargs

            return arguments
        except Exception as e:
            logger.warning("LLM arg extraction failed: %s", e)
            return {}

    def _llm_classify(
        self,
        query: str,
        state: dict[str, Any],
        recent_actions: list[str],
    ) -> dict[str, Any]:
        """Call the LLM to classify intent from function schemas."""
        from glyphh.llm.prompts import SCHEMA_CLASSIFY_USER

        # Build user prompt with per-query context
        state_summary = self._summarize_state(state)
        recent_str = ", ".join(recent_actions) if recent_actions else "none"

        user_prompt = SCHEMA_CLASSIFY_USER.format(
            state_summary=state_summary,
            recent_actions=recent_str,
            query=query,
        )

        try:
            result = self._llm.structured_generate(
                system=self._system_prompt,
                user=user_prompt,
                tools=self._classify_tools,
                max_tokens=1024,
            )

            data = result.data
            functions = []
            arguments = {}
            confidence = 0.0

            # Handle three response formats from the LLM:
            #
            # 1. Meta-tool format (classify_and_call):
            #    {"functions": [...], "arguments": {...}, "confidence": 0.9}
            #
            # 2. Single direct tool call (Qwen3 prefers calling functions directly):
            #    {"name": "func_name", "arguments": {"param": "value"}}
            #
            # 3. Multiple direct tool calls:
            #    {"tool_calls": [{"name": "f1", "arguments": {...}}, ...]}

            if "tool_calls" in data:
                # Multiple direct tool calls
                for tc in data["tool_calls"]:
                    fname = tc.get("name", "")
                    fargs = tc.get("arguments", {})
                    if fname in self._func_schemas:
                        functions.append(fname)
                        if isinstance(fargs, dict):
                            arguments[fname] = fargs
                confidence = 0.9 if functions else 0.0

            elif "name" in data and data["name"] in self._func_schemas:
                # Single direct tool call
                fname = data["name"]
                fargs = data.get("arguments", {})
                functions = [fname]
                arguments = {fname: fargs} if isinstance(fargs, dict) else {}
                confidence = 0.9

            elif "functions" in data:
                # Meta-tool format
                functions = data.get("functions", [])
                arguments = data.get("arguments", {})
                confidence = float(data.get("confidence", 0.0))

            # Validate function names against registered schemas
            valid_functions = [f for f in functions if f in self._func_schemas]

            # Validate arguments: only keep args for valid functions
            valid_arguments = {}
            for fname, args in arguments.items():
                if fname in self._func_schemas and isinstance(args, dict):
                    valid_arguments[fname] = args

            return {
                "functions": valid_functions,
                "arguments": valid_arguments,
                "confidence": confidence if valid_functions else 0.0,
                "latency_ms": result.latency_ms,
            }

        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            return {
                "functions": [],
                "arguments": {},
                "confidence": 0.0,
                "error": str(e),
            }

    def _summarize_state(self, state: dict[str, Any]) -> str:
        """Build a concise state summary for the LLM prompt."""
        parts = []
        primary = state.get("primary", "")
        if primary:
            parts.append(f"context={primary}")

        collections = state.get("collections", {})
        for name, items in collections.items():
            if isinstance(items, list) and items:
                sample = items[:5]
                if len(items) > 5:
                    parts.append(f"{name}=[{', '.join(str(s) for s in sample)}, ... ({len(items)} total)]")
                else:
                    parts.append(f"{name}=[{', '.join(str(s) for s in sample)}]")

        return ", ".join(parts) if parts else "empty"
