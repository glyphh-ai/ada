"""LLM-primary intent classification from function schemas.

Replaces IntentExtractor within CognitiveLoop when an LLM engine is
available. Uses function schemas as the intent space definition —
no static vocabulary files or domain packs needed.

Pipeline:
  1. HDC cache check (fast path for repeat patterns)
  2. LLM structured classification (primary)
  3. HDC cache store (learn from LLM for future fast path)

The LLM does PERCEIVE + RESOLVE + partial SLOT in a single call:
it selects function(s), extracts arguments, and reports confidence.
Over time, HDC learns from LLM decisions and handles common patterns
without calling the LLM (sub-millisecond).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .intent_cache import IntentCache

if TYPE_CHECKING:
    from glyphh.llm.engine import LLMEngine

logger = logging.getLogger(__name__)


class SchemaIntentClassifier:
    """LLM-primary intent classifier driven by function schemas.

    Usage:
        classifier = SchemaIntentClassifier(llm_engine, dimension=10000)
        classifier.configure(functions, action_to_func)  # at begin() time

        result = classifier.classify(query, state, recent_actions)
        # result = {functions, arguments, confidence, source}

        classifier.confirm(correct=True)  # Hebbian reinforcement
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        dimension: int = 10000,
        cache_threshold: float = 0.85,
    ):
        self._llm = llm_engine
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
        func_names = list(self._func_schemas.keys())

        # Build the system prompt with function descriptions baked in
        func_desc = format_function_descriptions(functions)
        self._system_prompt = SCHEMA_CLASSIFY_SYSTEM.format(
            function_descriptions=func_desc,
        )

        # Build tool schema with function names as enum constraint
        self._classify_tools = build_classify_tools(func_names)

        # Clear cache from previous session
        self._cache.clear()
        self._configured = True

    def classify(
        self,
        query: str,
        state: dict[str, Any],
        recent_actions: list[str],
    ) -> dict[str, Any]:
        """Classify intent from query using HDC cache or LLM.

        Returns:
            {
                "functions": ["func_name", ...],
                "arguments": {"func_name": {"param": "value"}},
                "confidence": 0.0-1.0,
                "source": "hdc_cache" | "llm",
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

        # 1. HDC cache check (fast path)
        cached = self._cache.lookup(query, state_primary)
        if cached is not None:
            logger.debug("IntentCache hit (sim=%.3f)", cached.get("cache_similarity", 0))
            cached["source"] = "hdc_cache"
            return cached

        # 2. LLM classification (primary)
        result = self._llm_classify(query, state, recent_actions)

        # 3. Store in cache for future fast path
        if result["confidence"] > 0.3:
            self._cache.store(query, state_primary, {
                "functions": result["functions"],
                "arguments": result["arguments"],
                "confidence": result["confidence"],
            })

        result["source"] = "llm"
        return result

    def confirm(self, correct: bool) -> None:
        """Hebbian reinforcement on the most recently cached entry."""
        self._cache.reinforce(correct)

    @property
    def cache_size(self) -> int:
        """Number of entries in the HDC cache."""
        return self._cache.size

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
                max_tokens=256,
            )

            data = result.data
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
