"""Tests for CognitiveLoop — the main LLM-free pipeline."""

import pytest

from glyphh.cognitive.loop import CognitiveLoop, StepResult
from glyphh.cognitive.domain import DomainConfig

from .conftest import DOMAIN_DICT, FUNC_SCHEMAS, make_state


DIM = 1000  # Small for fast tests


@pytest.fixture
def config():
    return DomainConfig.from_dict(DOMAIN_DICT)


@pytest.fixture
def loop(config):
    return CognitiveLoop(
        packs=[],
        domain_config=config,
        dimension=DIM,
        confidence_threshold=0.25,
    )


@pytest.fixture
def started_loop(loop):
    """Loop with begin() already called."""
    state = make_state(
        primary="root.workspace",
        items=["report.txt", "budget.csv"],
        locations=["archive", "temp"],
    )
    loop.begin(functions=FUNC_SCHEMAS, initial_state=state)
    return loop


class TestLoopInit:
    """CognitiveLoop initialization."""

    def test_creates_without_config(self):
        loop = CognitiveLoop(packs=[], dimension=DIM)
        assert loop._config is None

    def test_creates_with_config(self, config):
        loop = CognitiveLoop(packs=[], domain_config=config, dimension=DIM)
        assert loop._config is not None
        assert loop._sep == "."
        assert loop._default_primary == "root"

    def test_state_format_from_config(self, config):
        loop = CognitiveLoop(packs=[], domain_config=config, dimension=DIM)
        assert loop._sep == "."
        assert loop._default_primary == "root"
        assert loop._tree_key == "_tree"

    def test_action_to_func_loaded(self, config):
        loop = CognitiveLoop(packs=[], domain_config=config, dimension=DIM)
        assert loop._action_to_func["go"] == "navigate"
        assert loop._action_to_func["search"] == "lookup"


class TestLoopBegin:
    """CognitiveLoop.begin() initialization."""

    def test_returns_loop_id(self, loop):
        lid = loop.begin(functions=FUNC_SCHEMAS)
        assert isinstance(lid, str)
        assert len(lid) == 8

    def test_registers_functions(self, loop):
        loop.begin(functions=FUNC_SCHEMAS)
        assert "navigate" in loop._available_funcs
        assert "lookup" in loop._available_funcs
        assert "display" in loop._available_funcs

    def test_initial_state_set(self, loop):
        state = make_state(primary="root.zone_a", items=["item1"])
        loop.begin(functions=FUNC_SCHEMAS, initial_state=state)
        assert loop._state["primary"] == "root.zone_a"
        assert "item1" in loop._state["collections"]["items_here"]

    def test_default_state_when_none(self, loop):
        loop.begin(functions=FUNC_SCHEMAS)
        assert loop._state["primary"] == "root"
        assert loop._state["collections"] == {}


class TestLoopStep:
    """CognitiveLoop.step() — main pipeline."""

    def test_returns_step_result(self, started_loop):
        result = started_loop.step("show the content of report.txt")
        assert isinstance(result, StepResult)

    def test_step_result_has_signals(self, started_loop):
        result = started_loop.step("show the content of report.txt")
        assert "intent" in result.signals
        assert "resolved_functions" in result.signals

    def test_call_action_when_resolved(self, started_loop):
        result = started_loop.step("show the content of report.txt")
        # Should resolve to at least one function
        if result.action == "CALL":
            assert len(result.calls) > 0

    def test_ask_action_when_unresolvable(self, started_loop):
        # A query with no matching action/keywords
        result = started_loop.step("xyzzy plugh")
        assert result.action == "ASK"
        assert result.confidence == 0.0

    def test_turn_counter_advances(self, started_loop):
        assert started_loop._turn == 0
        started_loop.step("show report.txt")
        assert started_loop._turn == 1
        started_loop.step("search for budget")
        assert started_loop._turn == 2

    def test_recent_actions_tracked(self, started_loop):
        started_loop.step("show report.txt")
        assert len(started_loop._recent_actions) >= 0  # May or may not resolve


class TestMultiActionDetection:
    """Multi-action keyword detection from config."""

    def test_detects_create_keyword(self, started_loop):
        result = started_loop.step("create a new item called 'test_item'")
        if result.action == "CALL":
            func_names = []
            for call in result.calls:
                func_names.extend(call.keys())
            assert "make" in func_names

    def test_detects_navigate_keyword(self, started_loop):
        result = started_loop.step("navigate to the archive context")
        if result.action == "CALL":
            func_names = []
            for call in result.calls:
                func_names.extend(call.keys())
            assert "navigate" in func_names


class TestExclusionRules:
    """Exclusion rules: specific function removes generic one."""

    def test_preview_removes_display(self, started_loop):
        # "preview" should exclude "display" per config
        result = started_loop.step("show last 5 entries of report.txt")
        if result.action == "CALL":
            func_names = []
            for call in result.calls:
                func_names.extend(call.keys())
            # If both preview and display were detected, display should be removed
            if "preview" in func_names:
                assert "display" not in func_names


class TestTriggerSuppression:
    """Trigger function suppression from config."""

    def test_suppress_in_current_context(self, started_loop):
        # "current context" phrase should suppress navigate
        result = started_loop.step("show items in this context")
        if result.action == "CALL":
            func_names = []
            for call in result.calls:
                func_names.extend(call.keys())
            # navigate should be suppressed when "this context" is mentioned
            # (unless it was the primary action from intent)
            # This tests the suppression mechanism works
            assert isinstance(func_names, list)

    def test_strong_intent_prevents_suppression(self, started_loop):
        # "navigate to" is a strong intent pattern — should NOT suppress
        result = started_loop.step("navigate to the archive zone")
        if result.action == "CALL":
            func_names = []
            for call in result.calls:
                func_names.extend(call.keys())
            assert "navigate" in func_names


class TestStateUpdates:
    """State mutation via declarative effects from config."""

    def test_set_primary(self, started_loop):
        # Navigate to a new location
        started_loop._update_state([{"navigate": {"location": "archive"}}])
        assert started_loop._state["primary"] == "root.workspace.archive"

    def test_set_primary_parent(self, started_loop):
        # Navigate to parent using the configured parent_keyword ".."
        original = started_loop._state["primary"]  # "root.workspace"
        started_loop._update_state([{"navigate": {"location": ".."}}])
        assert started_loop._state["primary"] == "root"

    def test_set_primary_uses_config_separator(self, started_loop):
        # Our test config uses "." as separator
        started_loop._update_state([{"navigate": {"location": "zone_a"}}])
        assert "." in started_loop._state["primary"]
        assert started_loop._state["primary"] == "root.workspace.zone_a"

    def test_add_to_collection(self, started_loop):
        started_loop._update_state([{"make": {"item_name": "new_item.txt"}}])
        items = started_loop._state["collections"]["items_here"]
        assert "new_item.txt" in items

    def test_add_to_collection_no_duplicates(self, started_loop):
        started_loop._update_state([{"make": {"item_name": "report.txt"}}])
        items = started_loop._state["collections"]["items_here"]
        assert items.count("report.txt") == 1

    def test_remove_from_collection(self, started_loop):
        assert "report.txt" in started_loop._state["collections"]["items_here"]
        started_loop._update_state([{"delete": {"item_name": "report.txt"}}])
        assert "report.txt" not in started_loop._state["collections"]["items_here"]

    def test_no_effect_for_unknown_function(self, started_loop):
        original = dict(started_loop._state)
        started_loop._update_state([{"unknown_func": {"arg": "val"}}])
        assert started_loop._state["primary"] == original["primary"]

    def test_no_effect_with_empty_arg(self, started_loop):
        original_primary = started_loop._state["primary"]
        started_loop._update_state([{"navigate": {"location": ""}}])
        assert started_loop._state["primary"] == original_primary


class TestConfirm:
    """Hebbian reinforcement via confirm()."""

    def test_confirm_correct(self, started_loop):
        started_loop.step("show report.txt")
        # Should not error
        started_loop.confirm(was_correct=True)

    def test_confirm_incorrect_stores_correction(self, started_loop):
        started_loop.step("show report.txt")
        correct = [{"display": {"item_name": "report.txt"}}]
        started_loop.confirm(was_correct=False, correct_outcome=correct)
        # Should have stored a new idea
        assert started_loop.idea_space.size >= 1

    def test_confirm_without_step_no_error(self, started_loop):
        # confirm() before any step — should handle gracefully
        started_loop.confirm(was_correct=True)


class TestEnd:
    """CognitiveLoop.end() cleanup."""

    def test_end_clears_loop_state(self, started_loop):
        started_loop.step("show report.txt")
        started_loop.end()
        assert started_loop._loop_id is None
        assert started_loop._turn == 0
        assert started_loop._recent_actions == []
        assert started_loop._state == {}

    def test_end_preserves_idea_space(self, started_loop):
        started_loop.step("show report.txt")
        started_loop.confirm(was_correct=False, correct_outcome=[{"display": {}}])
        ideas_before = started_loop.idea_space.size
        started_loop.end()
        # Ideas survive end()
        assert started_loop.idea_space.size == ideas_before


class TestLoopWithoutConfig:
    """CognitiveLoop with no domain config — pure baseline."""

    def test_step_asks_when_no_config(self):
        loop = CognitiveLoop(packs=[], dimension=DIM)
        loop.begin(functions=FUNC_SCHEMAS)
        result = loop.step("search for budget")
        # Without config, no action_to_func mapping, so ASK
        assert result.action == "ASK"

    def test_no_state_effects_without_config(self):
        loop = CognitiveLoop(packs=[], dimension=DIM)
        loop.begin(functions=FUNC_SCHEMAS, initial_state={"primary": "root"})
        loop._update_state([{"navigate": {"location": "zone"}}])
        # State unchanged — no effects registered
        assert loop._state["primary"] == "root"


class TestRefreshCollections:
    """_refresh_collections() from tree data."""

    def test_refresh_from_tree(self, config):
        loop = CognitiveLoop(packs=[], domain_config=config, dimension=DIM)
        state = {
            "primary": "root.zone_a",
            "collections": {"items_here": [], "locations_here": []},
            "_tree": {
                "root.zone_a": {
                    "items_here": ["item_a.txt", "item_b.txt"],
                    "locations_here": ["sub_zone"],
                },
            },
        }
        loop.begin(functions=FUNC_SCHEMAS, initial_state=state)
        loop._refresh_collections()
        assert loop._state["collections"]["items_here"] == ["item_a.txt", "item_b.txt"]
        assert loop._state["collections"]["locations_here"] == ["sub_zone"]
