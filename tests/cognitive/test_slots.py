"""Tests for SlotExtractor — strategy-driven argument extraction."""

import pytest

from glyphh.cognitive.slots import SlotExtractor

from .conftest import DOMAIN_DICT, FUNC_SCHEMAS_DICT, make_state, DomainConfig


@pytest.fixture
def extractor(domain_config):
    return SlotExtractor(config=domain_config)


@pytest.fixture
def state():
    return make_state(
        primary="root.workspace",
        items=["report.txt", "budget.csv", "notes.md"],
        locations=["archive", "temp"],
    )


class TestQuotedStrategy:
    """'quoted' and 'quoted:N' strategies."""

    def test_quoted_string_extracted(self, extractor, state):
        result = extractor.extract(
            query="search for 'budget' in the item",
            functions=["lookup"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["lookup"]["pattern"] == "budget"

    def test_quoted_index_0(self, extractor, state):
        result = extractor.extract(
            query="compare 'alpha' to 'beta'",
            functions=["lookup"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        # item_name1 uses quoted:0, item_name2 uses quoted:1
        # But lookup only has pattern + item_name, not item_name1/item_name2
        assert result["lookup"]["pattern"] == "alpha"

    def test_double_quotes(self, extractor, state):
        result = extractor.extract(
            query='search for "budget" in the item',
            functions=["lookup"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["lookup"]["pattern"] == "budget"


class TestStateMatchStrategy:
    """'state_match:collection_name' strategy."""

    def test_matches_item_in_collection(self, extractor, state):
        result = extractor.extract(
            query="show the content of report.txt",
            functions=["display"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["display"]["item_name"] == "report.txt"

    def test_matches_quoted_item_in_collection(self, extractor, state):
        result = extractor.extract(
            query="show 'budget.csv'",
            functions=["display"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["display"]["item_name"] == "budget.csv"

    def test_no_match_empty_collection(self, extractor):
        empty_state = make_state(items=[], locations=[])
        result = extractor.extract(
            query="show content of something",
            functions=["display"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=empty_state,
        )
        assert "item_name" not in result["display"]


class TestFallbackWordsStrategy:
    """'fallback_words' strategy."""

    def test_back_maps_to_parent(self, extractor, state):
        result = extractor.extract(
            query="go back",
            functions=["navigate"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["navigate"]["location"] == ".."

    def test_up_maps_to_parent(self, extractor, state):
        result = extractor.extract(
            query="go up",
            functions=["navigate"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["navigate"]["location"] == ".."

    def test_parent_maps_to_parent(self, extractor, state):
        result = extractor.extract(
            query="go to parent",
            functions=["navigate"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["navigate"]["location"] == ".."


class TestStateHintStrategy:
    """'state_hint:key' strategy."""

    def test_reads_hint_from_state(self, extractor):
        state = make_state()
        state["query_mentions_child"] = "child_zone"
        result = extractor.extract(
            query="go to the child zone",
            functions=["navigate"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["navigate"]["location"] == "child_zone"


class TestNumberStrategy:
    """'number' strategy."""

    def test_extracts_number(self, extractor, state):
        result = extractor.extract(
            query="show last 5 entries of report.txt",
            functions=["preview"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert result["preview"]["count"] == 5

    def test_no_number(self, extractor, state):
        result = extractor.extract(
            query="show entries of report.txt",
            functions=["preview"],
            func_schemas=FUNC_SCHEMAS_DICT,
            state=state,
        )
        assert "count" not in result["preview"]


class TestBoolTriggersStrategy:
    """'bool_triggers' strategy."""

    def test_trigger_word_sets_true(self, extractor, state):
        # 'verbose' param has triggers: verbose, detailed, all
        # Need a function that uses the verbose param — let's test directly
        # via the slot extractor with a custom schema
        schemas = {
            "display": {
                "name": "display",
                "parameters": {
                    "properties": {"item_name": {}, "verbose": {}},
                    "required": ["item_name"],
                },
            }
        }
        result = extractor.extract(
            "show all content of report.txt",
            ["display"],
            schemas,
            state,
        )
        assert result["display"].get("verbose") is True

    def test_no_trigger_word(self, extractor, state):
        schemas = {
            "display": {
                "name": "display",
                "parameters": {
                    "properties": {"item_name": {}, "verbose": {}},
                    "required": ["item_name"],
                },
            }
        }
        result = extractor.extract(
            "show content of report.txt",
            ["display"],
            schemas,
            state,
        )
        assert "verbose" not in result["display"]


class TestKeywordMapStrategy:
    """'keyword_map' strategy."""

    def test_keyword_maps_to_value(self, extractor, state):
        schemas = {
            "display": {
                "name": "display",
                "parameters": {
                    "properties": {"item_name": {}, "sort_by": {}},
                    "required": ["item_name"],
                },
            }
        }
        result = extractor.extract(
            "show report.txt sorted by size",
            ["display"],
            schemas,
            state,
        )
        assert result["display"]["sort_by"] == "s"


class TestPositionalTransferStrategy:
    """'positional_before_transfer' / 'positional_after_transfer' strategies."""

    def test_source_and_dest(self, extractor, state):
        schemas = {
            "lookup": {
                "name": "lookup",
                "parameters": {
                    "properties": {"item_name1": {}, "item_name2": {}},
                    "required": ["item_name1", "item_name2"],
                },
            }
        }
        result = extractor.extract(
            "compare 'report.txt' to 'budget.csv'",
            ["lookup"],
            schemas,
            state,
        )
        assert result["lookup"]["item_name1"] == "report.txt"
        assert result["lookup"]["item_name2"] == "budget.csv"


class TestImplicitSingleStrategy:
    """'implicit_single:collection' strategy."""

    def test_auto_fill_single_item(self, extractor):
        state = make_state(items=["only_item.txt"])
        result = extractor.extract(
            "show the content",
            ["display"],
            FUNC_SCHEMAS_DICT,
            state,
        )
        assert result["display"]["item_name"] == "only_item.txt"

    def test_no_auto_fill_multiple_items(self, extractor):
        state = make_state(items=["a.txt", "b.txt"])
        result = extractor.extract(
            "show the content",
            ["display"],
            FUNC_SCHEMAS_DICT,
            state,
        )
        # With multiple items and no other match, should not auto-fill
        # (state_match might still match if query contains item name)
        # but implicit_single should NOT fire
        assert result["display"].get("item_name") is None


class TestMissingRequired:
    """SlotExtractor.missing_required() checks."""

    def test_no_missing(self, extractor):
        filled = {"display": {"item_name": "report.txt"}}
        missing = extractor.missing_required(["display"], FUNC_SCHEMAS_DICT, filled)
        assert missing == {}

    def test_detects_missing(self, extractor):
        filled = {"lookup": {"pattern": "budget"}}  # Missing item_name
        missing = extractor.missing_required(["lookup"], FUNC_SCHEMAS_DICT, filled)
        assert "lookup" in missing
        assert "item_name" in missing["lookup"]

    def test_multiple_functions(self, extractor):
        filled = {
            "navigate": {},  # Missing location
            "display": {"item_name": "report.txt"},
        }
        missing = extractor.missing_required(
            ["navigate", "display"], FUNC_SCHEMAS_DICT, filled,
        )
        assert "navigate" in missing
        assert "display" not in missing


class TestExtractorWithoutConfig:
    """SlotExtractor with no domain config."""

    def test_returns_empty_args(self):
        extractor = SlotExtractor(config=None)
        result = extractor.extract(
            "search for budget",
            ["lookup"],
            FUNC_SCHEMAS_DICT,
            {},
        )
        assert result["lookup"] == {}
