"""Tests for DomainConfig parsing and validation."""

import json
import tempfile
from pathlib import Path

import pytest

from glyphh.cognitive.domain import (
    DomainConfig,
    SlotDefinition,
    StateEffect,
    StateFormat,
    TriggerSuppression,
)

from .conftest import DOMAIN_DICT


class TestDomainConfigFromDict:
    """DomainConfig.from_dict() parsing."""

    def test_domain_name(self, domain_config):
        assert domain_config.domain == "test_domain"

    def test_action_to_func(self, domain_config):
        assert domain_config.action_to_func["go"] == "navigate"
        assert domain_config.action_to_func["search"] == "lookup"
        assert domain_config.action_to_func["create"] == "make"

    def test_slot_definitions_parsed(self, domain_config):
        assert "location" in domain_config.slot_definitions
        assert "item_name" in domain_config.slot_definitions
        assert "pattern" in domain_config.slot_definitions

    def test_slot_definition_type(self, domain_config):
        loc = domain_config.slot_definitions["location"]
        assert isinstance(loc, SlotDefinition)
        assert loc.type == "entity"
        assert loc.name == "location"

    def test_slot_definition_strategies(self, domain_config):
        loc = domain_config.slot_definitions["location"]
        assert loc.strategies == [
            "fallback_words",
            "state_hint:query_mentions_child",
            "state_match:locations_here",
            "quoted",
        ]

    def test_slot_definition_fallback_words(self, domain_config):
        loc = domain_config.slot_definitions["location"]
        assert loc.fallback_words["back"] == ".."
        assert loc.fallback_words["parent"] == ".."

    def test_slot_definition_triggers(self, domain_config):
        verbose = domain_config.slot_definitions["verbose"]
        assert verbose.triggers["verbose"] is True
        assert verbose.triggers["detailed"] is True

    def test_slot_definition_keyword_map(self, domain_config):
        sort_by = domain_config.slot_definitions["sort_by"]
        assert sort_by.keyword_map["alpha"] == "a"
        assert sort_by.keyword_map["size"] == "s"

    def test_state_effects_parsed(self, domain_config):
        assert "navigate" in domain_config.state_effects
        assert "make" in domain_config.state_effects
        assert "delete" in domain_config.state_effects

    def test_state_effect_navigate(self, domain_config):
        effects = domain_config.state_effects["navigate"]
        assert len(effects) == 1
        e = effects[0]
        assert isinstance(e, StateEffect)
        assert e.op == "set_primary"
        assert e.from_arg == "location"
        assert e.parent_keyword == ".."
        assert e.refresh_collections is True

    def test_state_effect_add_collection(self, domain_config):
        effects = domain_config.state_effects["make"]
        e = effects[0]
        assert e.op == "add_to_collection"
        assert e.collection == "items_here"
        assert e.from_arg == "item_name"

    def test_state_effect_remove_collection(self, domain_config):
        effects = domain_config.state_effects["delete"]
        e = effects[0]
        assert e.op == "remove_from_collection"
        assert e.collection == "items_here"

    def test_exclusion_rules(self, domain_config):
        assert domain_config.exclusion_rules["preview"] == ["display"]
        assert domain_config.exclusion_rules["lookup"] == ["display"]

    def test_trigger_suppression(self, domain_config):
        ts = domain_config.trigger_suppression
        assert isinstance(ts, TriggerSuppression)
        assert "navigate to" in ts.strong_intent_patterns
        assert "current context" in ts.current_context_phrases
        assert ts.primary_match_pattern != ""
        assert "items_here" in ts.file_collections

    def test_state_format(self, domain_config):
        sf = domain_config.state_format
        assert isinstance(sf, StateFormat)
        assert sf.separator == "."
        assert sf.default_primary == "root"
        assert sf.tree_key == "_tree"

    def test_navigation_trigger(self, domain_config):
        nt = domain_config.navigation_trigger
        assert nt["trigger_func"] == "navigate"
        assert "lookup" in nt["continuation_funcs"]

    def test_navigation_patterns(self, domain_config):
        pats = domain_config.navigation_patterns
        assert len(pats) == 2
        assert pats[0]["name"] == "nav_lookup"
        assert pats[0]["sequence"] == ["navigate", "lookup"]


class TestDomainConfigFromFile:
    """DomainConfig.from_file() I/O."""

    def test_loads_from_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(DOMAIN_DICT, f)
            f.flush()
            config = DomainConfig.from_file(f.name)

        assert config.domain == "test_domain"
        assert "location" in config.slot_definitions

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            DomainConfig.from_file("/nonexistent/path.json")


class TestDomainConfigDefaults:
    """DomainConfig with missing/empty fields."""

    def test_empty_dict(self):
        config = DomainConfig.from_dict({})
        assert config.domain == "unknown"
        assert config.action_to_func == {}
        assert config.slot_definitions == {}
        assert config.state_effects == {}
        assert config.exclusion_rules == {}
        assert config.state_format.separator == "/"
        assert config.state_format.default_primary == ""

    def test_partial_dict(self):
        config = DomainConfig.from_dict({
            "domain": "partial",
            "action_to_func": {"do": "something"},
        })
        assert config.domain == "partial"
        assert config.action_to_func["do"] == "something"
        assert config.slot_definitions == {}

    def test_slot_definition_defaults(self):
        config = DomainConfig.from_dict({
            "slot_definitions": {
                "bare": {"type": "text"},
            }
        })
        slot = config.slot_definitions["bare"]
        assert slot.name == "bare"
        assert slot.strategies == []
        assert slot.fallback_words == {}
        assert slot.triggers == {}
        assert slot.keyword_map == {}
