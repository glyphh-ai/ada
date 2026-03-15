"""
SlotExtractor — data-driven argument extraction from query + state.

All extraction logic is driven by the domain config's slot_definitions.
Each parameter has an ordered list of strategies; the first match wins.

Generic extraction strategies (no domain knowledge):
  quoted / quoted:N         — Nth quoted string from query
  quoted_filtered:collection — first quoted string NOT in the named collection
  state_match:collection    — match against a named state collection
  state_hint:key            — read a key from state dict
  number                    — first number in query
  bool_triggers             — config trigger words → boolean
  keyword_map               — config keyword → enum value
  fallback_words            — config keyword → literal value
  positional_before_transfer — source text before "to"/"into"
  positional_after_transfer  — destination text after "to"/"into"
  implicit_single:collection — auto-fill if collection has exactly 1 item
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .domain import DomainConfig

# Regex patterns for extraction
# Match "double quoted" or 'single quoted' — but for single quotes, require
# word boundary before the opening quote to avoid matching contractions
# like it's, don't, she'd etc.
_DOUBLE_QUOTED_RE = re.compile(r'"([^"]+)"')
_SINGLE_QUOTED_RE = re.compile(r"(?<!\w)'([^']+)'(?!\w)")
_NUMBER_RE = re.compile(r"\b(\d+)\b")

# Words that separate source from destination
_TRANSFER_WORDS = {"to", "into", "onto", "as", "under"}


class SlotExtractor:
    """Data-driven slot extraction from query text + state.

    Driven entirely by the domain config's slot_definitions.
    The extractor has zero domain knowledge — it only knows
    generic extraction strategies.
    """

    def __init__(self, config: DomainConfig | None = None):
        self._config = config
        self._slot_defs = config.slot_definitions if config else {}

    def extract(
        self,
        query: str,
        functions: list[str],
        func_schemas: dict[str, dict],
        state: dict,
        claim_aware: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Extract argument values for each function.

        Args:
            query: Raw user query text
            functions: List of function names to fill slots for
            func_schemas: {func_name: schema_dict} with parameters.properties
            state: Normalized state dict with keys: primary, collections, etc.
            claim_aware: When True, extracted string values are "claimed" so
                subsequent functions in the list won't reuse them. Useful when
                a single query produces multiple function calls.

        Returns:
            {func_name: {param_name: value}} for each function
        """
        # Pre-compute shared extraction context once
        ctx = self._build_context(query, state)

        result = {}
        cross_claimed: set[str] = set()

        for fname in functions:
            schema = func_schemas.get(fname, {})
            params = schema.get("parameters", {}).get("properties", {})

            # Intra-function claiming: params within same function don't reuse values.
            # Cross-function claiming (claim_aware): values persist across functions.
            ctx["_claimed"] = set(cross_claimed) if claim_aware else set()

            args: dict[str, Any] = {}
            for pname in params:
                slot_def = self._slot_defs.get(pname)
                if slot_def is None:
                    continue  # No extraction rule for this param

                val = self._fill_param(slot_def, ctx)
                if val is not None:
                    args[pname] = val
                    # Claim within this function so next param skips it
                    if isinstance(val, str):
                        ctx["_claimed"].add(val)

            result[fname] = args

            # Track cross-function claims
            if claim_aware:
                cross_claimed = set(ctx["_claimed"])

        return result

    def missing_required(
        self,
        functions: list[str],
        func_schemas: dict[str, dict],
        filled_slots: dict[str, dict],
    ) -> dict[str, list[str]]:
        """Check which required slots are still unfilled.

        Returns {func_name: [missing_param_names]}.
        """
        missing = {}
        for fname in functions:
            schema = func_schemas.get(fname, {})
            required = set(schema.get("parameters", {}).get("required", []))
            filled = set(filled_slots.get(fname, {}).keys())
            unfilled = required - filled
            if unfilled:
                missing[fname] = sorted(unfilled)
        return missing

    # ── Internal ──

    def _build_context(self, query: str, state: dict) -> dict:
        """Pre-compute shared extraction data once per call."""
        query_lower = query.lower()
        # Double quotes first (higher confidence), then single quotes
        quoted = _DOUBLE_QUOTED_RE.findall(query) + _SINGLE_QUOTED_RE.findall(query)
        collections = state.get("collections", {})

        # Pre-compute transfer split
        src_candidates, dst_candidates = self._split_transfer(
            query_lower, quoted,
        )

        return {
            "query": query,
            "query_lower": query_lower,
            "words": query_lower.split(),
            "quoted": quoted,
            "numbers": [int(n) for n in _NUMBER_RE.findall(query)],
            "state": state,
            "collections": collections,
            "primary": state.get("primary", ""),
            "src_candidates": src_candidates,
            "dst_candidates": dst_candidates,
        }

    def _fill_param(self, slot_def: Any, ctx: dict) -> Any:
        """Walk the strategy list for a param; return first match."""
        for strategy_spec in slot_def.strategies:
            strategy, _, arg = strategy_spec.partition(":")
            val = self._execute_strategy(strategy, arg, slot_def, ctx)
            if val is not None:
                return val
        return None

    def _execute_strategy(
        self,
        strategy: str,
        arg: str,
        slot_def: Any,
        ctx: dict,
    ) -> Any:
        """Dispatch to individual extraction strategy."""
        if strategy == "quoted":
            return self._extract_quoted(arg, ctx)
        elif strategy == "quoted_filtered":
            return self._extract_quoted_filtered(arg, ctx)
        elif strategy == "state_match":
            return self._extract_state_match(arg, ctx)
        elif strategy == "state_hint":
            return self._extract_state_hint(arg, ctx)
        elif strategy == "number":
            return self._extract_number(ctx)
        elif strategy == "bool_triggers":
            return self._extract_bool_triggers(slot_def, ctx)
        elif strategy == "keyword_map":
            return self._extract_keyword_map(slot_def, ctx)
        elif strategy == "fallback_words":
            return self._extract_fallback_words(slot_def, ctx)
        elif strategy == "positional_before_transfer":
            return self._extract_positional(before=True, ctx=ctx)
        elif strategy == "positional_after_transfer":
            return self._extract_positional(before=False, ctx=ctx)
        elif strategy == "implicit_single":
            return self._extract_implicit_single(arg, ctx)
        return None

    # ── Strategy implementations ──

    def _extract_quoted(self, arg: str, ctx: dict) -> str | None:
        """Extract a quoted string. If arg is a digit index, return that index.

        For indexed access (quoted:N), uses original positions but skips
        if the value is claimed. For unindexed access, returns first unclaimed.
        """
        quoted = ctx["quoted"]
        if not quoted:
            return None
        claimed = ctx.get("_claimed", set())
        if arg and arg.isdigit():
            idx = int(arg)
            if idx < len(quoted) and quoted[idx] not in claimed:
                return quoted[idx]
            return None
        # Unindexed: first unclaimed
        for q in quoted:
            if q not in claimed:
                return q
        return None

    def _extract_quoted_filtered(self, exclude_collection: str, ctx: dict) -> str | None:
        """Return first quoted string NOT found in the exclude collection.

        Useful for type-aware disambiguation: e.g. for a 'folder' param,
        use 'quoted_filtered:items_here' to skip file names.
        """
        exclude = set(ctx["collections"].get(exclude_collection, []))
        claimed = ctx.get("_claimed", set())
        for q in ctx["quoted"]:
            if q not in exclude and q not in claimed:
                return q
        return None

    def _extract_state_match(self, collection_name: str, ctx: dict) -> str | None:
        """Match quoted strings or query words against a state collection."""
        collection = ctx["collections"].get(collection_name, [])
        if not collection:
            return None

        collection_set = set(collection)
        claimed = ctx.get("_claimed", set())
        query_lower = ctx["query_lower"]
        words = ctx["words"]

        # Check quoted strings first (exact match)
        for q in ctx["quoted"]:
            if q in collection_set and q not in claimed:
                return q

        # Check if collection items appear as substrings in query
        for item in collection:
            item_lower = item.lower()
            if item_lower in query_lower and item not in claimed:
                return item

        # Fuzzy: check if query words match collection items with
        # singular/plural tolerance (e.g. "document" matches "documents")
        collection_lower = {c.lower(): c for c in collection}
        for word in words:
            # Exact word match
            if word in collection_lower and collection_lower[word] not in claimed:
                return collection_lower[word]
            # Word + 's' matches collection item (singular query → plural item)
            if (word + "s") in collection_lower and collection_lower[word + "s"] not in claimed:
                return collection_lower[word + "s"]
            # Word without 's' matches collection item (plural query → singular item)
            if word.endswith("s") and word[:-1] in collection_lower and collection_lower[word[:-1]] not in claimed:
                return collection_lower[word[:-1]]

        return None

    def _extract_state_hint(self, hint_key: str, ctx: dict) -> str | None:
        """Read a specific hint key from the state dict."""
        val = ctx["state"].get(hint_key)
        return val if val else None

    def _extract_number(self, ctx: dict) -> int | None:
        """Extract the first number from the query."""
        numbers = ctx["numbers"]
        return numbers[0] if numbers else None

    def _extract_bool_triggers(self, slot_def: Any, ctx: dict) -> bool | None:
        """Check config trigger words against query words."""
        triggers = slot_def.triggers
        if not triggers:
            return None
        for word in ctx["words"]:
            if word in triggers:
                return triggers[word]
        return None

    def _extract_keyword_map(self, slot_def: Any, ctx: dict) -> str | None:
        """Check config keyword→value map against query words."""
        kw_map = slot_def.keyword_map
        if not kw_map:
            return None
        for word in ctx["words"]:
            if word in kw_map:
                return kw_map[word]
        return None

    def _extract_fallback_words(self, slot_def: Any, ctx: dict) -> str | None:
        """Check config keyword→literal value map (e.g. 'parent' → '..')."""
        fallback = slot_def.fallback_words
        if not fallback:
            return None
        for word in ctx["words"]:
            if word in fallback:
                return fallback[word]
        return None

    def _extract_positional(self, before: bool, ctx: dict) -> str | None:
        """Extract source (before) or destination (after) from transfer split."""
        candidates = ctx["src_candidates"] if before else ctx["dst_candidates"]
        claimed = ctx.get("_claimed", set())
        for c in candidates:
            if c not in claimed:
                return c
        return None

    def _extract_implicit_single(self, collection_name: str, ctx: dict) -> str | None:
        """Auto-fill if a state collection has exactly one item."""
        collection = ctx["collections"].get(collection_name, [])
        return collection[0] if len(collection) == 1 else None

    # ── Transfer word splitting (generic) ──

    def _split_transfer(
        self,
        query_lower: str,
        quoted: list[str],
    ) -> tuple[list[str], list[str]]:
        """Split quoted strings into source vs destination.

        Uses positional context: things before "to"/"into" are sources,
        things after are destinations.
        """
        if len(quoted) < 2:
            return quoted[:], []

        # Find transfer word position
        words = query_lower.split()
        transfer_idx = -1
        for i, w in enumerate(words):
            if w in _TRANSFER_WORDS:
                transfer_idx = i
                break

        if transfer_idx == -1:
            # No transfer word: first is source, last is dest
            return [quoted[0]], [quoted[-1]]

        # Use character position of transfer word
        transfer_pos = query_lower.find(f" {words[transfer_idx]} ")
        if transfer_pos == -1:
            transfer_pos = query_lower.find(words[transfer_idx])

        src = []
        dst = []
        for q in quoted:
            q_pos = query_lower.find(q.lower())
            if q_pos < transfer_pos:
                src.append(q)
            else:
                dst.append(q)

        return src, dst
