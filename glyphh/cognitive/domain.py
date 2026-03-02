"""
DomainConfig — load and validate domain configuration for CognitiveLoop.

The domain config defines all domain-specific knowledge:
  - action_to_func: canonical action → API function name
  - slot_definitions: per-param extraction strategy pipeline
  - multi_action_keywords: multi-action detection patterns
  - state_effects: how function calls change state
  - exclusion_rules: which functions supersede others
  - trigger_suppression: when to suppress the trigger function
  - state_format: how primary state values are structured
  - navigation_patterns: ConversationState pathway seeding
  - navigation_trigger: which function triggers nav context
  - target_extraction_rules: for DeductiveLayer target tracking

JSON files live model-side. The model handler loads the file,
passes it to CognitiveLoop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlotDefinition:
    """Extraction pipeline for a single parameter.

    Strategies are tried in order; first match wins.
    Strategy format: "strategy_name" or "strategy_name:argument"
      e.g. "quoted", "quoted:0", "state_match:collection_name",
           "implicit_single:collection_name", "state_hint:hint_key"
    """

    name: str
    type: str = "text"                              # entity, text, number, boolean, enum
    strategies: list[str] = field(default_factory=list)
    fallback_words: dict[str, str] = field(default_factory=dict)
    triggers: dict[str, bool] = field(default_factory=dict)
    keyword_map: dict[str, str] = field(default_factory=dict)


@dataclass
class StateEffect:
    """One state mutation triggered by a function call.

    Operations:
      set_primary           — change the primary state key
      add_to_collection     — append a value to a named collection
      remove_from_collection — remove a value from a named collection
    """

    op: str = ""                    # "set_primary", "add_to_collection", "remove_from_collection"
    from_arg: str = ""              # which function arg provides the value
    collection: str = ""            # target collection (for add/remove ops)
    parent_keyword: str = ""        # value that means "go up" in the state hierarchy
    refresh_collections: bool = False  # refresh collections from tree after set_primary


@dataclass
class TriggerSuppression:
    """Config for when to suppress the trigger function (e.g. navigation).

    The trigger function (from navigation_trigger) is often added by keyword
    patterns or prediction. These rules let the domain config specify when
    that trigger should be suppressed — e.g., when the query is about
    the current context and doesn't need navigation.
    """

    strong_intent_patterns: list[str] = field(default_factory=list)
    current_context_phrases: list[str] = field(default_factory=list)
    primary_match_pattern: str = ""
    file_collections: list[str] = field(default_factory=list)


@dataclass
class StateFormat:
    """Config for how primary state values are structured.

    Allows the domain config to define the separator used in hierarchical
    state values and how parent navigation works — so the SDK doesn't
    need to hardcode any path format.
    """

    separator: str = "/"
    default_primary: str = ""
    tree_key: str = "_tree"


@dataclass
class DomainConfig:
    """Parsed domain configuration for CognitiveLoop.

    All domain-specific knowledge lives here. The SDK cognitive
    classes read this config — they contain zero domain logic.
    """

    domain: str = "unknown"
    action_to_func: dict[str, str] = field(default_factory=dict)
    slot_definitions: dict[str, SlotDefinition] = field(default_factory=dict)
    multi_action_keywords: dict[str, list[str]] = field(default_factory=dict)
    state_effects: dict[str, list[StateEffect]] = field(default_factory=dict)
    exclusion_rules: dict[str, list[str]] = field(default_factory=dict)
    trigger_suppression: TriggerSuppression = field(default_factory=TriggerSuppression)
    state_format: StateFormat = field(default_factory=StateFormat)
    navigation_patterns: list[dict] = field(default_factory=list)
    navigation_trigger: dict = field(default_factory=dict)
    target_extraction_rules: dict[str, dict] = field(default_factory=dict)
    initial_state_parser: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainConfig:
        """Parse a domain config dict into a DomainConfig."""
        # Parse slot definitions
        slot_defs: dict[str, SlotDefinition] = {}
        for pname, pdef in data.get("slot_definitions", {}).items():
            slot_defs[pname] = SlotDefinition(
                name=pname,
                type=pdef.get("type", "text"),
                strategies=pdef.get("strategies", []),
                fallback_words=pdef.get("fallback_words", {}),
                triggers=pdef.get("triggers", {}),
                keyword_map=pdef.get("keyword_map", {}),
            )

        # Parse state effects
        effects: dict[str, list[StateEffect]] = {}
        for fname, effect_list in data.get("state_effects", {}).items():
            effects[fname] = [
                StateEffect(
                    op=e.get("op", ""),
                    from_arg=e.get("from_arg", ""),
                    collection=e.get("collection", ""),
                    parent_keyword=e.get("parent_keyword", ""),
                    refresh_collections=e.get("refresh_collections", False),
                )
                for e in effect_list
            ]

        # Parse trigger suppression
        ts_data = data.get("trigger_suppression", {})
        trigger_supp = TriggerSuppression(
            strong_intent_patterns=ts_data.get("strong_intent_patterns", []),
            current_context_phrases=ts_data.get("current_context_phrases", []),
            primary_match_pattern=ts_data.get("primary_match_pattern", ""),
            file_collections=ts_data.get("file_collections", []),
        )

        # Parse state format
        sf_data = data.get("state_format", {})
        state_fmt = StateFormat(
            separator=sf_data.get("separator", "/"),
            default_primary=sf_data.get("default_primary", ""),
            tree_key=sf_data.get("tree_key", "_tree"),
        )

        return cls(
            domain=data.get("domain", "unknown"),
            action_to_func=data.get("action_to_func", {}),
            slot_definitions=slot_defs,
            multi_action_keywords=data.get("multi_action_keywords", {}),
            state_effects=effects,
            exclusion_rules=data.get("exclusion_rules", {}),
            trigger_suppression=trigger_supp,
            state_format=state_fmt,
            navigation_patterns=data.get("navigation_patterns", []),
            navigation_trigger=data.get("navigation_trigger", {}),
            target_extraction_rules=data.get("target_extraction_rules", {}),
            initial_state_parser=data.get("initial_state_parser", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> DomainConfig:
        """Load a domain config from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
