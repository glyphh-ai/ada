"""Shared fixtures for glyphh.cognitive tests.

Uses a generic "test_domain" — no filesystem or BFCL references.
Proves the SDK is truly domain-agnostic.
"""

import pytest

from glyphh.cognitive.domain import (
    DomainConfig,
    SlotDefinition,
    StateEffect,
    StateFormat,
    TriggerSuppression,
)


# ── Minimal domain config for testing ──

DOMAIN_DICT = {
    "domain": "test_domain",

    "action_to_func": {
        "go": "navigate",
        "search": "lookup",
        "read": "display",
        "create": "make",
        "list": "display",
        "remove": "delete",
    },

    "slot_definitions": {
        "location": {
            "type": "entity",
            "strategies": [
                "fallback_words",
                "state_hint:query_mentions_child",
                "state_match:locations_here",
                "quoted",
            ],
            "fallback_words": {"back": "..", "up": "..", "parent": ".."},
        },
        "item_name": {
            "type": "entity",
            "strategies": ["quoted", "state_match:items_here", "implicit_single:items_here"],
        },
        "item_name1": {
            "type": "entity",
            "strategies": ["positional_before_transfer", "quoted:0", "state_match:items_here"],
        },
        "item_name2": {
            "type": "entity",
            "strategies": ["positional_after_transfer", "quoted:1", "state_match:items_here"],
        },
        "pattern": {
            "type": "text",
            "strategies": ["quoted"],
        },
        "count": {
            "type": "number",
            "strategies": ["number"],
        },
        "verbose": {
            "type": "boolean",
            "strategies": ["bool_triggers"],
            "triggers": {"verbose": True, "detailed": True, "all": True},
        },
        "sort_by": {
            "type": "enum",
            "strategies": ["keyword_map"],
            "keyword_map": {"alpha": "a", "alphabetical": "a", "size": "s", "date": "d"},
        },
    },

    "multi_action_keywords": {
        "navigate": [
            "go to", "navigate to", "switch to.*context",
            "enter (?:the )?.*(?:zone|area|context)",
        ],
        "make": [
            "create.*(?:item|entry)", "new (?:item|entry)",
            "(?:^|\\. )make.*(?:item|entry)",
        ],
        "lookup": [
            "search for", "find.*in the",
            "look up.*in", "investigate",
        ],
        "display": [
            "show (?:the )?content", "read (?:the )?(?:item|entry)",
            "view (?:the )?(?:item|entry)", "display",
        ],
        "delete": [
            "remove.*item", "delete.*(?:item|entry)",
        ],
        "preview": [
            "last \\d+ entries", "preview.*(?:item|entry)",
        ],
    },

    "exclusion_rules": {
        "preview": ["display"],
        "lookup": ["display"],
    },

    "state_effects": {
        "navigate": [
            {
                "op": "set_primary",
                "from_arg": "location",
                "parent_keyword": "..",
                "refresh_collections": True,
            }
        ],
        "make": [
            {
                "op": "add_to_collection",
                "collection": "items_here",
                "from_arg": "item_name",
            }
        ],
        "delete": [
            {
                "op": "remove_from_collection",
                "collection": "items_here",
                "from_arg": "item_name",
            }
        ],
    },

    "trigger_suppression": {
        "strong_intent_patterns": [
            "navigate to", "go to ", "switch to",
            "enter (?:the )?.*(?:zone|area|context)",
        ],
        "current_context_phrases": [
            "current context", "this context",
            "this area", "in this zone",
        ],
        "primary_match_pattern": "(?:in|within) (?:the )?(?:'{primary_name}'|{primary_name}) (?:zone|area|context)",
        "file_collections": ["items_here"],
    },

    "state_format": {
        "separator": ".",
        "default_primary": "root",
        "tree_key": "_tree",
    },

    "navigation_patterns": [
        {"name": "nav_lookup", "sequence": ["navigate", "lookup"]},
        {"name": "nav_display", "sequence": ["navigate", "display"]},
    ],

    "navigation_trigger": {
        "name": "nav_context",
        "trigger_func": "navigate",
        "continuation_funcs": ["lookup", "display", "make", "delete", "preview"],
    },

    "target_extraction_rules": {
        "make": {"arg": "item_name", "format": "{primary}.{arg_value}"},
    },

    "initial_state_parser": {
        "root_key": "TestSystem",
        "tree_key": "root",
    },
}


# ── Minimal function schemas for the test domain ──

FUNC_SCHEMAS = [
    {
        "name": "navigate",
        "description": "Navigate to a context",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "lookup",
        "description": "Search for a pattern in an item",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "item_name": {"type": "string"},
            },
            "required": ["pattern", "item_name"],
        },
    },
    {
        "name": "display",
        "description": "Display an item's content",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string"},
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "make",
        "description": "Create a new item",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string"},
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "delete",
        "description": "Remove an item",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string"},
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "preview",
        "description": "Preview last N entries of an item",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["item_name"],
        },
    },
]

# As a dict keyed by name (convenience)
FUNC_SCHEMAS_DICT = {f["name"]: f for f in FUNC_SCHEMAS}


def make_state(
    primary: str = "root",
    items: list[str] | None = None,
    locations: list[str] | None = None,
    tree: dict | None = None,
) -> dict:
    """Build a normalized state dict for tests."""
    state = {
        "primary": primary,
        "collections": {
            "items_here": items or [],
            "locations_here": locations or [],
        },
    }
    if tree:
        state["_tree"] = tree
    return state


@pytest.fixture
def domain_config():
    """Parsed DomainConfig for the test domain."""
    return DomainConfig.from_dict(DOMAIN_DICT)


@pytest.fixture
def func_schemas():
    """Function schema list."""
    return FUNC_SCHEMAS


@pytest.fixture
def func_schemas_dict():
    """Function schemas as dict keyed by name."""
    return FUNC_SCHEMAS_DICT
