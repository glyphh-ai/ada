"""
Unit tests for the IntentInferrer class.

Tests the __init__() method and IntentKeywords/InferredIntent dataclasses.

Validates: Requirement 4 - Intent Inference from Matches
"""

import pytest
from glyphh.nl.intent_inferrer import (
    IntentInferrer,
    IntentKeywords,
    InferredIntent,
    VALID_INTENT_TYPES,
)


class TestIntentKeywords:
    """Tests for the IntentKeywords dataclass."""
    
    def test_default_keywords(self):
        """Test default IntentKeywords values.
        
        Validates: Requirements 4.2, 4.3, 4.4
        """
        keywords = IntentKeywords()
        
        # Count keywords
        assert "how many" in keywords.count
        assert "count" in keywords.count
        assert "number of" in keywords.count
        
        # Find keywords
        assert "find" in keywords.find
        assert "show" in keywords.find
        assert "get" in keywords.find
        assert "list" in keywords.find
        assert "search" in keywords.find
        
        # Filter keywords
        assert "greater" in keywords.filter
        assert "less" in keywords.filter
        assert "between" in keywords.filter
        assert "more" in keywords.filter
        assert "fewer" in keywords.filter
        
        # Similar keywords
        assert "similar" in keywords.similar
        assert "like" in keywords.similar
        assert "related" in keywords.similar
    
    def test_custom_keywords(self):
        """Test custom IntentKeywords values."""
        keywords = IntentKeywords(
            count={"count", "total", "sum"},
            find={"find", "locate", "retrieve"},
            filter={"where", "having"},
            similar={"like", "matching"}
        )
        
        assert "total" in keywords.count
        assert "locate" in keywords.find
        assert "where" in keywords.filter
        assert "matching" in keywords.similar
    
    def test_keywords_normalized_to_lowercase(self):
        """Test that keywords are normalized to lowercase."""
        keywords = IntentKeywords(
            count={"COUNT", "Total"},
            find={"FIND", "Show"},
            filter={"GREATER", "Less"},
            similar={"SIMILAR", "Like"}
        )
        
        assert "count" in keywords.count
        assert "total" in keywords.count
        assert "find" in keywords.find
        assert "show" in keywords.find
        assert "greater" in keywords.filter
        assert "less" in keywords.filter
        assert "similar" in keywords.similar
        assert "like" in keywords.similar
    
    def test_keywords_list_converted_to_set(self):
        """Test that keyword lists are converted to sets."""
        keywords = IntentKeywords(
            count=["count", "total"],
            find=["find", "show"],
            filter=["greater", "less"],
            similar=["similar", "like"]
        )
        
        assert isinstance(keywords.count, set)
        assert isinstance(keywords.find, set)
        assert isinstance(keywords.filter, set)
        assert isinstance(keywords.similar, set)
    
    def test_keywords_invalid_type_raises(self):
        """Test that invalid keyword type raises TypeError."""
        with pytest.raises(TypeError, match="count must be a set or iterable"):
            IntentKeywords(count="invalid")
    
    def test_get_all_keywords(self):
        """Test get_all_keywords() returns all keywords."""
        keywords = IntentKeywords()
        all_kw = keywords.get_all_keywords()
        
        assert "find" in all_kw
        assert "count" in all_kw
        assert "greater" in all_kw
        assert "similar" in all_kw
    
    def test_get_intent_for_keyword(self):
        """Test get_intent_for_keyword() returns correct intent."""
        keywords = IntentKeywords()
        
        assert keywords.get_intent_for_keyword("find") == "find"
        assert keywords.get_intent_for_keyword("count") == "count"
        assert keywords.get_intent_for_keyword("how many") == "count"
        assert keywords.get_intent_for_keyword("greater") == "filter"
        assert keywords.get_intent_for_keyword("similar") == "similar"
        assert keywords.get_intent_for_keyword("unknown") is None
    
    def test_get_intent_for_keyword_case_insensitive(self):
        """Test get_intent_for_keyword() is case-insensitive."""
        keywords = IntentKeywords()
        
        assert keywords.get_intent_for_keyword("FIND") == "find"
        assert keywords.get_intent_for_keyword("Count") == "count"
        assert keywords.get_intent_for_keyword("HOW MANY") == "count"
    
    def test_keywords_repr(self):
        """Test IntentKeywords __repr__."""
        keywords = IntentKeywords()
        repr_str = repr(keywords)
        
        assert "IntentKeywords" in repr_str
        assert "count=" in repr_str
        assert "find=" in repr_str
        assert "filter=" in repr_str
        assert "similar=" in repr_str


class TestInferredIntent:
    """Tests for the InferredIntent dataclass."""
    
    def test_inferred_intent_creation(self):
        """Test basic InferredIntent creation."""
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find", "show"],
            supporting_matches=[]
        )
        
        assert intent.intent_type == "find"
        assert intent.confidence == 0.85
        assert intent.matched_keywords == ["find", "show"]
        assert intent.supporting_matches == []
    
    def test_inferred_intent_valid_types(self):
        """Test that all valid intent types are accepted."""
        for intent_type in VALID_INTENT_TYPES:
            intent = InferredIntent(
                intent_type=intent_type,
                confidence=0.5,
                matched_keywords=[],
                supporting_matches=[]
            )
            assert intent.intent_type == intent_type
    
    def test_inferred_intent_invalid_type_raises(self):
        """Test that invalid intent type raises ValueError."""
        with pytest.raises(ValueError, match="intent_type must be one of"):
            InferredIntent(
                intent_type="invalid",
                confidence=0.5,
                matched_keywords=[],
                supporting_matches=[]
            )
    
    def test_inferred_intent_confidence_bounds(self):
        """Test that confidence must be in [0.0, 1.0].
        
        Validates: Requirement 4.6 - THE SDK SHALL return confidence scores for inferred intents
        """
        # Valid confidence values
        InferredIntent(intent_type="find", confidence=0.0, matched_keywords=[], supporting_matches=[])
        InferredIntent(intent_type="find", confidence=0.5, matched_keywords=[], supporting_matches=[])
        InferredIntent(intent_type="find", confidence=1.0, matched_keywords=[], supporting_matches=[])
        
        # Invalid confidence values
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            InferredIntent(intent_type="find", confidence=-0.1, matched_keywords=[], supporting_matches=[])
        
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            InferredIntent(intent_type="find", confidence=1.1, matched_keywords=[], supporting_matches=[])
    
    def test_inferred_intent_confidence_not_number_raises(self):
        """Test that non-numeric confidence raises TypeError."""
        with pytest.raises(TypeError, match="confidence must be a number"):
            InferredIntent(
                intent_type="find",
                confidence="high",
                matched_keywords=[],
                supporting_matches=[]
            )
    
    def test_inferred_intent_matched_keywords_not_list_raises(self):
        """Test that non-list matched_keywords raises TypeError."""
        with pytest.raises(TypeError, match="matched_keywords must be a list"):
            InferredIntent(
                intent_type="find",
                confidence=0.5,
                matched_keywords="find",
                supporting_matches=[]
            )
    
    def test_inferred_intent_supporting_matches_not_list_raises(self):
        """Test that non-list supporting_matches raises TypeError."""
        with pytest.raises(TypeError, match="supporting_matches must be a list"):
            InferredIntent(
                intent_type="find",
                confidence=0.5,
                matched_keywords=[],
                supporting_matches="matches"
            )
    
    def test_is_high_confidence(self):
        """Test is_high_confidence() method."""
        high_intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        low_intent = InferredIntent(
            intent_type="find",
            confidence=0.5,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        assert high_intent.is_high_confidence() is True
        assert high_intent.is_high_confidence(threshold=0.9) is False
        assert low_intent.is_high_confidence() is False
        assert low_intent.is_high_confidence(threshold=0.4) is True
    
    def test_has_keyword_support(self):
        """Test has_keyword_support() method."""
        with_keywords = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        without_keywords = InferredIntent(
            intent_type="find",
            confidence=0.5,
            matched_keywords=[],
            supporting_matches=[]
        )
        
        assert with_keywords.has_keyword_support() is True
        assert without_keywords.has_keyword_support() is False
    
    def test_inferred_intent_repr(self):
        """Test InferredIntent __repr__."""
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        repr_str = repr(intent)
        
        assert "InferredIntent" in repr_str
        assert "find" in repr_str
        assert "0.85" in repr_str


class TestIntentInferrerInit:
    """Tests for the IntentInferrer.__init__() method.
    
    Validates: Requirements 4.2, 4.3, 4.4, 4.5
    """
    
    def test_init_with_default_keywords(self):
        """Test IntentInferrer initialization with default keywords.
        
        Validates: Requirements 4.2, 4.3, 4.4
        """
        inferrer = IntentInferrer()
        
        # Verify default count keywords
        assert "how many" in inferrer.keywords.count
        assert "count" in inferrer.keywords.count
        assert "number of" in inferrer.keywords.count
        
        # Verify default find keywords
        assert "find" in inferrer.keywords.find
        assert "show" in inferrer.keywords.find
        assert "get" in inferrer.keywords.find
        assert "list" in inferrer.keywords.find
        assert "search" in inferrer.keywords.find
        
        # Verify default filter keywords
        assert "greater" in inferrer.keywords.filter
        assert "less" in inferrer.keywords.filter
        assert "between" in inferrer.keywords.filter
        assert "more" in inferrer.keywords.filter
        assert "fewer" in inferrer.keywords.filter
        
        # Verify default similar keywords
        assert "similar" in inferrer.keywords.similar
        assert "like" in inferrer.keywords.similar
        assert "related" in inferrer.keywords.similar
    
    def test_init_with_none_uses_defaults(self):
        """Test that passing None uses default keywords."""
        inferrer = IntentInferrer(keywords=None)
        
        assert "find" in inferrer.keywords.find
        assert "how many" in inferrer.keywords.count
        assert "greater" in inferrer.keywords.filter
        assert "similar" in inferrer.keywords.similar
    
    def test_init_with_custom_keywords(self):
        """Test IntentInferrer initialization with custom keywords.
        
        Validates: Requirement 4.5 - THE SDK SHALL support configurable intent keywords per model
        """
        custom_keywords = IntentKeywords(
            count={"count", "total", "sum"},
            find={"find", "locate", "retrieve"},
            filter={"where", "having"},
            similar={"like", "matching"}
        )
        inferrer = IntentInferrer(keywords=custom_keywords)
        
        assert "total" in inferrer.keywords.count
        assert "locate" in inferrer.keywords.find
        assert "where" in inferrer.keywords.filter
        assert "matching" in inferrer.keywords.similar
        
        # Default keywords should not be present
        assert "how many" not in inferrer.keywords.count
        assert "show" not in inferrer.keywords.find
        assert "greater" not in inferrer.keywords.filter
        assert "related" not in inferrer.keywords.similar
    
    def test_init_with_invalid_keywords_type_raises(self):
        """Test that invalid keywords type raises TypeError."""
        with pytest.raises(TypeError, match="keywords must be an IntentKeywords instance or None"):
            IntentInferrer(keywords="invalid")
        
        with pytest.raises(TypeError, match="keywords must be an IntentKeywords instance or None"):
            IntentInferrer(keywords={"count": {"count"}})
        
        with pytest.raises(TypeError, match="keywords must be an IntentKeywords instance or None"):
            IntentInferrer(keywords=123)
    
    def test_init_stores_keywords_reference(self):
        """Test that init stores the keywords reference correctly."""
        custom_keywords = IntentKeywords(
            count={"count"},
            find={"find"},
            filter={"filter"},
            similar={"similar"}
        )
        inferrer = IntentInferrer(keywords=custom_keywords)
        
        # Should be the same object
        assert inferrer.keywords is custom_keywords
    
    def test_init_creates_new_default_keywords(self):
        """Test that init creates a new IntentKeywords instance when None is passed."""
        inferrer1 = IntentInferrer()
        inferrer2 = IntentInferrer()
        
        # Should be different instances
        assert inferrer1.keywords is not inferrer2.keywords
        
        # But with the same default values
        assert inferrer1.keywords.count == inferrer2.keywords.count
        assert inferrer1.keywords.find == inferrer2.keywords.find
    
    def test_init_with_empty_keyword_sets(self):
        """Test that init accepts IntentKeywords with empty sets."""
        empty_keywords = IntentKeywords(
            count=set(),
            find=set(),
            filter=set(),
            similar=set()
        )
        inferrer = IntentInferrer(keywords=empty_keywords)
        
        assert len(inferrer.keywords.count) == 0
        assert len(inferrer.keywords.find) == 0
        assert len(inferrer.keywords.filter) == 0
        assert len(inferrer.keywords.similar) == 0
    
    def test_init_with_partial_custom_keywords(self):
        """Test that init works with partially customized keywords."""
        # Only customize count keywords, use defaults for others
        partial_keywords = IntentKeywords(
            count={"count", "total"},
            # find, filter, similar use defaults
        )
        inferrer = IntentInferrer(keywords=partial_keywords)
        
        # Custom count keywords
        assert "total" in inferrer.keywords.count
        assert "count" in inferrer.keywords.count
        
        # Default find keywords
        assert "find" in inferrer.keywords.find
        assert "show" in inferrer.keywords.find
    
    def test_inferrer_repr(self):
        """Test IntentInferrer __repr__."""
        inferrer = IntentInferrer()
        repr_str = repr(inferrer)
        
        assert "IntentInferrer" in repr_str
        assert "keywords=" in repr_str
    
    def test_keywords_attribute_is_intent_keywords_instance(self):
        """Test that keywords attribute is always an IntentKeywords instance."""
        inferrer_default = IntentInferrer()
        inferrer_none = IntentInferrer(keywords=None)
        inferrer_custom = IntentInferrer(keywords=IntentKeywords())
        
        assert isinstance(inferrer_default.keywords, IntentKeywords)
        assert isinstance(inferrer_none.keywords, IntentKeywords)
        assert isinstance(inferrer_custom.keywords, IntentKeywords)


class TestDetectKeywords:
    """Tests for the IntentInferrer.detect_keywords() method.
    
    Validates: Requirements 4.2, 4.3, 4.4
    """
    
    def test_detect_count_keywords(self):
        """Test detection of count keywords.
        
        Validates: Requirement 4.2 - WHEN a query contains "how many" or "count", 
        THE SDK SHALL infer intent=count
        """
        inferrer = IntentInferrer()
        
        # Test "how many"
        result = inferrer.detect_keywords("How many Toyota cars?")
        assert "count" in result
        assert "how many" in result["count"]
        
        # Test "count"
        result = inferrer.detect_keywords("Count the cars")
        assert "count" in result
        assert "count" in result["count"]
        
        # Test "number of"
        result = inferrer.detect_keywords("What is the number of items?")
        assert "count" in result
        assert "number of" in result["count"]
    
    def test_detect_find_keywords(self):
        """Test detection of find keywords.
        
        Validates: Requirement 4.3 - WHEN a query contains "find", "show", "get", 
        THE SDK SHALL infer intent=find
        """
        inferrer = IntentInferrer()
        
        # Test "find"
        result = inferrer.detect_keywords("Find Toyota cars")
        assert "find" in result
        assert "find" in result["find"]
        
        # Test "show"
        result = inferrer.detect_keywords("Show me the cars")
        assert "find" in result
        assert "show" in result["find"]
        
        # Test "get"
        result = inferrer.detect_keywords("Get all items")
        assert "find" in result
        assert "get" in result["find"]
        
        # Test "list"
        result = inferrer.detect_keywords("List all products")
        assert "find" in result
        assert "list" in result["find"]
        
        # Test "search"
        result = inferrer.detect_keywords("Search for Toyota")
        assert "find" in result
        assert "search" in result["find"]
    
    def test_detect_filter_keywords(self):
        """Test detection of filter keywords.
        
        Validates: Requirement 4.4 - WHEN a query contains comparison operators, 
        THE SDK SHALL infer intent=filter
        """
        inferrer = IntentInferrer()
        
        # Test "greater"
        result = inferrer.detect_keywords("Price greater than 100")
        assert "filter" in result
        assert "greater" in result["filter"]
        
        # Test "less"
        result = inferrer.detect_keywords("Price less than 50")
        assert "filter" in result
        assert "less" in result["filter"]
        
        # Test "between"
        result = inferrer.detect_keywords("Price between 10 and 100")
        assert "filter" in result
        assert "between" in result["filter"]
        
        # Test "more"
        result = inferrer.detect_keywords("More than 5 items")
        assert "filter" in result
        assert "more" in result["filter"]
        
        # Test "fewer"
        result = inferrer.detect_keywords("Fewer than 10 results")
        assert "filter" in result
        assert "fewer" in result["filter"]
    
    def test_detect_similar_keywords(self):
        """Test detection of similar keywords."""
        inferrer = IntentInferrer()
        
        # Test "similar"
        result = inferrer.detect_keywords("Find similar items")
        assert "similar" in result
        assert "similar" in result["similar"]
        
        # Test "like"
        result = inferrer.detect_keywords("Items like this one")
        assert "similar" in result
        assert "like" in result["similar"]
        
        # Test "related"
        result = inferrer.detect_keywords("Show related products")
        assert "similar" in result
        assert "related" in result["similar"]
    
    def test_detect_multiple_intent_types(self):
        """Test detection of keywords from multiple intent types."""
        inferrer = IntentInferrer()
        
        # Query with find and similar keywords
        result = inferrer.detect_keywords("Find similar Toyota")
        assert "find" in result
        assert "similar" in result
        assert "find" in result["find"]
        assert "similar" in result["similar"]
        
        # Query with find and filter keywords
        result = inferrer.detect_keywords("Show me cars with greater price")
        assert "find" in result
        assert "filter" in result
        assert "show" in result["find"]
        assert "greater" in result["filter"]
    
    def test_detect_no_keywords(self):
        """Test that queries without keywords return empty dict."""
        inferrer = IntentInferrer()
        
        result = inferrer.detect_keywords("Hello world")
        assert result == {}
        
        result = inferrer.detect_keywords("Toyota cars")
        assert result == {}
    
    def test_detect_case_insensitive(self):
        """Test that keyword detection is case-insensitive."""
        inferrer = IntentInferrer()
        
        # Uppercase
        result = inferrer.detect_keywords("FIND TOYOTA")
        assert "find" in result
        assert "find" in result["find"]
        
        # Mixed case
        result = inferrer.detect_keywords("How Many Cars?")
        assert "count" in result
        assert "how many" in result["count"]
        
        # All caps multi-word
        result = inferrer.detect_keywords("HOW MANY items?")
        assert "count" in result
        assert "how many" in result["count"]
    
    def test_detect_multi_word_keywords(self):
        """Test detection of multi-word keywords like 'how many' and 'number of'."""
        inferrer = IntentInferrer()
        
        # "how many" should be detected as a single keyword
        result = inferrer.detect_keywords("How many Toyota cars are there?")
        assert "count" in result
        assert "how many" in result["count"]
        assert len(result["count"]) == 1  # Only one keyword matched
        
        # "number of" should be detected as a single keyword
        result = inferrer.detect_keywords("What is the number of items?")
        assert "count" in result
        assert "number of" in result["count"]
        assert len(result["count"]) == 1
    
    def test_detect_word_boundaries(self):
        """Test that keywords are matched at word boundaries only."""
        inferrer = IntentInferrer()
        
        # "find" should not match "finding"
        result = inferrer.detect_keywords("I am finding cars")
        assert "find" not in result
        
        # "count" should not match "counting"
        result = inferrer.detect_keywords("I am counting items")
        assert "count" not in result
        
        # "show" should not match "showing"
        result = inferrer.detect_keywords("I am showing results")
        assert "find" not in result
        
        # But "find" should match when it's a complete word
        result = inferrer.detect_keywords("Find the cars")
        assert "find" in result
        assert "find" in result["find"]
    
    def test_detect_multiple_keywords_same_intent(self):
        """Test detection of multiple keywords for the same intent type."""
        inferrer = IntentInferrer()
        
        # Multiple find keywords
        result = inferrer.detect_keywords("Find and show me the list of cars")
        assert "find" in result
        # Should contain multiple keywords
        assert len(result["find"]) >= 2
        assert "find" in result["find"]
        assert "show" in result["find"]
    
    def test_detect_empty_query(self):
        """Test that empty query returns empty dict."""
        inferrer = IntentInferrer()
        
        result = inferrer.detect_keywords("")
        assert result == {}
    
    def test_detect_with_custom_keywords(self):
        """Test detection with custom keywords."""
        custom_keywords = IntentKeywords(
            count={"total", "sum"},
            find={"locate", "retrieve"},
            filter={"where", "having"},
            similar={"matching"}
        )
        inferrer = IntentInferrer(keywords=custom_keywords)
        
        # Custom count keyword
        result = inferrer.detect_keywords("What is the total?")
        assert "count" in result
        assert "total" in result["count"]
        
        # Custom find keyword
        result = inferrer.detect_keywords("Locate the item")
        assert "find" in result
        assert "locate" in result["find"]
        
        # Default keyword should not match
        result = inferrer.detect_keywords("Find the item")
        assert "find" not in result
    
    def test_detect_keywords_order_preserved(self):
        """Test that matched keywords are returned in order of appearance."""
        inferrer = IntentInferrer()
        
        # Keywords should be in order of appearance in query
        result = inferrer.detect_keywords("Show me the list and search for items")
        assert "find" in result
        # "show" appears before "list" and "search"
        keywords = result["find"]
        show_idx = keywords.index("show") if "show" in keywords else -1
        list_idx = keywords.index("list") if "list" in keywords else -1
        search_idx = keywords.index("search") if "search" in keywords else -1
        
        if show_idx >= 0 and list_idx >= 0:
            assert show_idx < list_idx
        if list_idx >= 0 and search_idx >= 0:
            assert list_idx < search_idx
    
    def test_detect_keywords_with_punctuation(self):
        """Test keyword detection with punctuation around keywords."""
        inferrer = IntentInferrer()
        
        # Keyword followed by punctuation
        result = inferrer.detect_keywords("Find, show, and list items")
        assert "find" in result
        assert "find" in result["find"]
        assert "show" in result["find"]
        assert "list" in result["find"]
        
        # Keyword with question mark
        result = inferrer.detect_keywords("How many?")
        assert "count" in result
        assert "how many" in result["count"]
    
    def test_detect_keywords_at_boundaries(self):
        """Test keyword detection at start and end of query."""
        inferrer = IntentInferrer()
        
        # Keyword at start
        result = inferrer.detect_keywords("Find items")
        assert "find" in result
        assert "find" in result["find"]
        
        # Keyword at end
        result = inferrer.detect_keywords("Items to find")
        assert "find" in result
        assert "find" in result["find"]
        
        # Only keyword
        result = inferrer.detect_keywords("find")
        assert "find" in result
        assert "find" in result["find"]


class TestComputeIntentConfidence:
    """Tests for the IntentInferrer.compute_intent_confidence() method.
    
    Validates: Requirement 4.6 - THE SDK SHALL return confidence scores for inferred intents
    """
    
    def test_confidence_with_single_keyword(self):
        """Test confidence calculation with a single keyword.
        
        Single keyword should give 0.3 confidence.
        """
        inferrer = IntentInferrer()
        
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find"],
            matches=[]
        )
        
        assert confidence == 0.3
    
    def test_confidence_with_two_keywords(self):
        """Test confidence calculation with two keywords.
        
        Two keywords should give 0.6 confidence (max from keywords).
        """
        inferrer = IntentInferrer()
        
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find", "show"],
            matches=[]
        )
        
        assert confidence == 0.6
    
    def test_confidence_with_three_keywords_capped(self):
        """Test that keyword confidence is capped at 0.6.
        
        Three keywords would be 0.9, but should be capped at 0.6.
        """
        inferrer = IntentInferrer()
        
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find", "show", "get"],
            matches=[]
        )
        
        # Should be capped at 0.6 (max keyword confidence)
        assert confidence == 0.6
    
    def test_confidence_with_no_keywords_or_matches(self):
        """Test confidence is 0.0 with no keywords or matches."""
        inferrer = IntentInferrer()
        
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=[],
            matches=[]
        )
        
        assert confidence == 0.0
    
    def test_confidence_with_matches_adds_bonus(self):
        """Test that schema matches add to confidence.
        
        Each match adds 0.1, capped at 0.3.
        """
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock token and schema vector for TokenMatch
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create a mock vector (dimension 100 for testing)
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        schema_vec = SchemaVector(
            key="make=Toyota",
            vector=mock_vector,
            element_type="value",
            role_path="vehicle.identity.make",
            original_value="Toyota"
        )
        
        # Create a single match with low similarity (below 0.7 threshold)
        match = TokenMatch(
            token=token,
            schema_vector=schema_vec,
            similarity=0.5,
            match_type="exact"
        )
        
        # Single keyword (0.3) + single match (0.1) = 0.4
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find"],
            matches=[match]
        )
        
        assert confidence == 0.4
    
    def test_confidence_with_multiple_matches(self):
        """Test confidence with multiple schema matches."""
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock tokens and schema vectors
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        matches = []
        for i in range(3):
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            schema_vec = SchemaVector(
                key=f"key{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"path{i}",
                original_value=f"Value{i}"
            )
            match = TokenMatch(
                token=token,
                schema_vector=schema_vec,
                similarity=0.5,  # Below high-quality threshold
                match_type="partial"
            )
            matches.append(match)
        
        # Single keyword (0.3) + 3 matches (0.3, capped) = 0.6
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find"],
            matches=matches
        )
        
        assert confidence == 0.6
    
    def test_confidence_match_bonus_capped(self):
        """Test that match bonus is capped at 0.3."""
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock tokens and schema vectors
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        # Create 5 matches (would be 0.5, but should be capped at 0.3)
        matches = []
        for i in range(5):
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            schema_vec = SchemaVector(
                key=f"key{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"path{i}",
                original_value=f"Value{i}"
            )
            match = TokenMatch(
                token=token,
                schema_vector=schema_vec,
                similarity=0.5,  # Below high-quality threshold
                match_type="partial"
            )
            matches.append(match)
        
        # No keywords (0.0) + 5 matches (0.3, capped) = 0.3
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=[],
            matches=matches
        )
        
        assert confidence == 0.3
    
    def test_confidence_high_quality_bonus(self):
        """Test that high-quality matches (avg similarity > 0.7) add bonus."""
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock token and schema vector
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        schema_vec = SchemaVector(
            key="make=Toyota",
            vector=mock_vector,
            element_type="value",
            role_path="vehicle.identity.make",
            original_value="Toyota"
        )
        
        # Create a high-quality match (similarity > 0.7)
        match = TokenMatch(
            token=token,
            schema_vector=schema_vec,
            similarity=0.85,  # Above high-quality threshold
            match_type="exact"
        )
        
        # Single keyword (0.3) + single match (0.1) + high-quality bonus (0.1) = 0.5
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find"],
            matches=[match]
        )
        
        assert confidence == 0.5
    
    def test_confidence_no_high_quality_bonus_below_threshold(self):
        """Test that no bonus is added when avg similarity <= 0.7."""
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock token and schema vector
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        schema_vec = SchemaVector(
            key="make=Toyota",
            vector=mock_vector,
            element_type="value",
            role_path="vehicle.identity.make",
            original_value="Toyota"
        )
        
        # Create a match exactly at threshold (0.7 - should NOT get bonus)
        match = TokenMatch(
            token=token,
            schema_vector=schema_vec,
            similarity=0.7,  # Exactly at threshold, not above
            match_type="exact"
        )
        
        # Single keyword (0.3) + single match (0.1) + no bonus = 0.4
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find"],
            matches=[match]
        )
        
        assert confidence == 0.4
    
    def test_confidence_clamped_to_max_1(self):
        """Test that confidence is clamped to maximum 1.0."""
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock tokens and schema vectors
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        # Create multiple high-quality matches
        matches = []
        for i in range(5):
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            schema_vec = SchemaVector(
                key=f"key{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"path{i}",
                original_value=f"Value{i}"
            )
            match = TokenMatch(
                token=token,
                schema_vector=schema_vec,
                similarity=0.9,  # High quality
                match_type="exact"
            )
            matches.append(match)
        
        # Max keywords (0.6) + max matches (0.3) + high-quality bonus (0.1) = 1.0
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find", "show", "get"],  # 3 keywords
            matches=matches
        )
        
        # Use approximate comparison due to floating point precision
        assert abs(confidence - 1.0) < 1e-10
        assert confidence <= 1.0  # Must be clamped to max 1.0
    
    def test_confidence_always_in_valid_range(self):
        """Test that confidence is always in [0.0, 1.0] range.
        
        Validates: Requirement 4.6
        """
        inferrer = IntentInferrer()
        
        # Test various combinations
        test_cases = [
            ([], []),  # No keywords, no matches
            (["find"], []),  # Single keyword
            (["find", "show"], []),  # Two keywords
            (["find", "show", "get", "list"], []),  # Many keywords
        ]
        
        for keywords, matches in test_cases:
            confidence = inferrer.compute_intent_confidence(
                intent_type="find",
                keywords=keywords,
                matches=matches
            )
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range for {keywords}"
    
    def test_confidence_all_intent_types(self):
        """Test confidence calculation works for all valid intent types."""
        inferrer = IntentInferrer()
        
        for intent_type in ["count", "find", "filter", "similar"]:
            confidence = inferrer.compute_intent_confidence(
                intent_type=intent_type,
                keywords=["keyword"],
                matches=[]
            )
            assert 0.0 <= confidence <= 1.0
            assert confidence == 0.3  # Single keyword
    
    def test_confidence_invalid_intent_type_raises(self):
        """Test that invalid intent type raises ValueError."""
        inferrer = IntentInferrer()
        
        with pytest.raises(ValueError, match="intent_type must be one of"):
            inferrer.compute_intent_confidence(
                intent_type="invalid",
                keywords=["find"],
                matches=[]
            )
    
    def test_confidence_keywords_not_list_raises(self):
        """Test that non-list keywords raises TypeError."""
        inferrer = IntentInferrer()
        
        with pytest.raises(TypeError, match="keywords must be a list"):
            inferrer.compute_intent_confidence(
                intent_type="find",
                keywords="find",  # Should be a list
                matches=[]
            )
    
    def test_confidence_matches_not_list_raises(self):
        """Test that non-list matches raises TypeError."""
        inferrer = IntentInferrer()
        
        with pytest.raises(TypeError, match="matches must be a list"):
            inferrer.compute_intent_confidence(
                intent_type="find",
                keywords=["find"],
                matches="match"  # Should be a list
            )
    
    def test_confidence_with_mixed_quality_matches(self):
        """Test confidence with a mix of high and low quality matches."""
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock tokens and schema vectors
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        # Create matches with mixed similarity scores
        # Average will be (0.9 + 0.5) / 2 = 0.7, which is NOT > 0.7
        matches = []
        for i, sim in enumerate([0.9, 0.5]):
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            schema_vec = SchemaVector(
                key=f"key{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"path{i}",
                original_value=f"Value{i}"
            )
            match = TokenMatch(
                token=token,
                schema_vector=schema_vec,
                similarity=sim,
                match_type="partial"
            )
            matches.append(match)
        
        # Single keyword (0.3) + 2 matches (0.2) + no bonus (avg=0.7, not >0.7) = 0.5
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find"],
            matches=matches
        )
        
        assert confidence == 0.5
    
    def test_confidence_with_high_avg_similarity(self):
        """Test confidence when average similarity is above threshold."""
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock tokens and schema vectors
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        # Create matches with high similarity scores
        # Average will be (0.9 + 0.8) / 2 = 0.85, which IS > 0.7
        matches = []
        for i, sim in enumerate([0.9, 0.8]):
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            schema_vec = SchemaVector(
                key=f"key{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"path{i}",
                original_value=f"Value{i}"
            )
            match = TokenMatch(
                token=token,
                schema_vector=schema_vec,
                similarity=sim,
                match_type="partial"
            )
            matches.append(match)
        
        # Single keyword (0.3) + 2 matches (0.2) + high-quality bonus (0.1) = 0.6
        confidence = inferrer.compute_intent_confidence(
            intent_type="find",
            keywords=["find"],
            matches=matches
        )
        
        assert confidence == 0.6


class TestInferIntent:
    """Tests for the IntentInferrer.infer_intent() method.
    
    Validates: Requirements 4.1, 4.5
    """
    
    def test_infer_intent_with_find_keyword(self):
        """Test intent inference with find keyword.
        
        Validates: Requirement 4.1 - THE SDK SHALL infer intent based on query keywords
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="Find Toyota cars", token_matches=[])
        
        intents = inferrer.infer_intent("Find Toyota cars", match_result)
        
        assert len(intents) == 1
        assert intents[0].intent_type == "find"
        assert "find" in intents[0].matched_keywords
        assert intents[0].confidence > 0
    
    def test_infer_intent_with_count_keyword(self):
        """Test intent inference with count keyword.
        
        Validates: Requirement 4.1
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="How many Toyota cars?", token_matches=[])
        
        intents = inferrer.infer_intent("How many Toyota cars?", match_result)
        
        assert len(intents) == 1
        assert intents[0].intent_type == "count"
        assert "how many" in intents[0].matched_keywords
    
    def test_infer_intent_with_filter_keyword(self):
        """Test intent inference with filter keyword.
        
        Validates: Requirement 4.1
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="Price greater than 100", token_matches=[])
        
        intents = inferrer.infer_intent("Price greater than 100", match_result)
        
        assert len(intents) == 1
        assert intents[0].intent_type == "filter"
        assert "greater" in intents[0].matched_keywords
    
    def test_infer_intent_with_similar_keyword(self):
        """Test intent inference with similar keyword.
        
        Validates: Requirement 4.1
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="Find similar items", token_matches=[])
        
        intents = inferrer.infer_intent("Find similar items", match_result)
        
        assert len(intents) == 2
        # Should have both find and similar intents
        intent_types = {i.intent_type for i in intents}
        assert "find" in intent_types
        assert "similar" in intent_types
    
    def test_infer_intent_multiple_intents_sorted_by_confidence(self):
        """Test that multiple intents are sorted by confidence (descending).
        
        Validates: Requirement 4.1
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="Find similar Toyota", token_matches=[])
        
        intents = inferrer.infer_intent("Find similar Toyota", match_result)
        
        assert len(intents) >= 2
        # Verify sorted by confidence (descending)
        for i in range(len(intents) - 1):
            assert intents[i].confidence >= intents[i + 1].confidence
    
    def test_infer_intent_no_keywords_returns_empty_list(self):
        """Test that queries without keywords return empty list.
        
        Validates: Requirement 4.1
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="Toyota cars", token_matches=[])
        
        intents = inferrer.infer_intent("Toyota cars", match_result)
        
        assert intents == []
    
    def test_infer_intent_with_supporting_matches(self):
        """Test that supporting matches are included in inferred intents.
        
        Validates: Requirement 4.1
        """
        from glyphh.nl.schema_matcher import MatchResult, TokenMatch
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock token match
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=5,
            is_stop_word=False,
            ngram_size=1
        )
        schema_vec = SchemaVector(
            key="make=Toyota",
            vector=mock_vector,
            element_type="value",
            role_path="vehicle.identity.make",
            original_value="Toyota"
        )
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vec,
            similarity=0.85,
            match_type="exact"
        )
        
        match_result = MatchResult(
            query="Find Toyota cars",
            token_matches=[token_match]
        )
        
        intents = inferrer.infer_intent("Find Toyota cars", match_result)
        
        assert len(intents) == 1
        assert intents[0].intent_type == "find"
        assert len(intents[0].supporting_matches) == 1
        assert intents[0].supporting_matches[0] is token_match
    
    def test_infer_intent_confidence_increases_with_matches(self):
        """Test that confidence increases when supporting matches are present.
        
        Validates: Requirement 4.1
        """
        from glyphh.nl.schema_matcher import MatchResult, TokenMatch
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.core.types import Vector
        import numpy as np
        
        inferrer = IntentInferrer()
        
        # Create mock token match
        mock_data = np.ones(100, dtype=np.int8)
        mock_vector = Vector(data=mock_data, dimension=100, space_id="test")
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=5,
            is_stop_word=False,
            ngram_size=1
        )
        schema_vec = SchemaVector(
            key="make=Toyota",
            vector=mock_vector,
            element_type="value",
            role_path="vehicle.identity.make",
            original_value="Toyota"
        )
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vec,
            similarity=0.85,
            match_type="exact"
        )
        
        # Without matches
        match_result_no_matches = MatchResult(
            query="Find Toyota cars",
            token_matches=[]
        )
        intents_no_matches = inferrer.infer_intent("Find Toyota cars", match_result_no_matches)
        
        # With matches
        match_result_with_matches = MatchResult(
            query="Find Toyota cars",
            token_matches=[token_match]
        )
        intents_with_matches = inferrer.infer_intent("Find Toyota cars", match_result_with_matches)
        
        # Confidence should be higher with matches
        assert intents_with_matches[0].confidence > intents_no_matches[0].confidence
    
    def test_infer_intent_with_custom_keywords(self):
        """Test intent inference with custom keywords.
        
        Validates: Requirement 4.5 - THE SDK SHALL support configurable intent keywords per model
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        custom_keywords = IntentKeywords(
            count={"total", "sum"},
            find={"locate", "retrieve"},
            filter={"where", "having"},
            similar={"matching"}
        )
        inferrer = IntentInferrer(keywords=custom_keywords)
        match_result = MatchResult(query="Locate the item", token_matches=[])
        
        intents = inferrer.infer_intent("Locate the item", match_result)
        
        assert len(intents) == 1
        assert intents[0].intent_type == "find"
        assert "locate" in intents[0].matched_keywords
    
    def test_infer_intent_custom_keywords_default_not_matched(self):
        """Test that default keywords don't match when custom keywords are used.
        
        Validates: Requirement 4.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        custom_keywords = IntentKeywords(
            count={"total"},
            find={"locate"},
            filter={"where"},
            similar={"matching"}
        )
        inferrer = IntentInferrer(keywords=custom_keywords)
        match_result = MatchResult(query="Find the item", token_matches=[])
        
        # "find" is not in custom keywords, so no intent should be inferred
        intents = inferrer.infer_intent("Find the item", match_result)
        
        assert intents == []
    
    def test_infer_intent_query_not_string_raises(self):
        """Test that non-string query raises TypeError."""
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="test", token_matches=[])
        
        with pytest.raises(TypeError, match="query must be a string"):
            inferrer.infer_intent(123, match_result)
        
        with pytest.raises(TypeError, match="query must be a string"):
            inferrer.infer_intent(None, match_result)
    
    def test_infer_intent_match_result_not_match_result_raises(self):
        """Test that non-MatchResult raises TypeError."""
        inferrer = IntentInferrer()
        
        with pytest.raises(TypeError, match="match_result must be a MatchResult instance"):
            inferrer.infer_intent("Find Toyota", "not a match result")
        
        with pytest.raises(TypeError, match="match_result must be a MatchResult instance"):
            inferrer.infer_intent("Find Toyota", None)
        
        with pytest.raises(TypeError, match="match_result must be a MatchResult instance"):
            inferrer.infer_intent("Find Toyota", {"query": "test"})
    
    def test_infer_intent_empty_query(self):
        """Test intent inference with empty query."""
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="", token_matches=[])
        
        intents = inferrer.infer_intent("", match_result)
        
        assert intents == []
    
    def test_infer_intent_case_insensitive(self):
        """Test that intent inference is case-insensitive."""
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        
        # Uppercase
        match_result = MatchResult(query="FIND TOYOTA", token_matches=[])
        intents = inferrer.infer_intent("FIND TOYOTA", match_result)
        assert len(intents) == 1
        assert intents[0].intent_type == "find"
        
        # Mixed case
        match_result = MatchResult(query="How Many Cars?", token_matches=[])
        intents = inferrer.infer_intent("How Many Cars?", match_result)
        assert len(intents) == 1
        assert intents[0].intent_type == "count"
    
    def test_infer_intent_multiple_keywords_same_intent(self):
        """Test intent inference with multiple keywords for same intent."""
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="Find and show me the list", token_matches=[])
        
        intents = inferrer.infer_intent("Find and show me the list", match_result)
        
        assert len(intents) == 1
        assert intents[0].intent_type == "find"
        # Should have multiple keywords
        assert len(intents[0].matched_keywords) >= 2
    
    def test_infer_intent_returns_inferred_intent_objects(self):
        """Test that infer_intent returns InferredIntent objects."""
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        match_result = MatchResult(query="Find Toyota", token_matches=[])
        
        intents = inferrer.infer_intent("Find Toyota", match_result)
        
        assert len(intents) == 1
        assert isinstance(intents[0], InferredIntent)
        assert hasattr(intents[0], 'intent_type')
        assert hasattr(intents[0], 'confidence')
        assert hasattr(intents[0], 'matched_keywords')
        assert hasattr(intents[0], 'supporting_matches')
    
    def test_infer_intent_confidence_in_valid_range(self):
        """Test that all inferred intent confidences are in [0.0, 1.0]."""
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        
        # Test various queries
        queries = [
            "Find Toyota",
            "How many cars?",
            "Find similar items",
            "Show me the list and search for Toyota",
            "Count items greater than 100",
        ]
        
        for query in queries:
            match_result = MatchResult(query=query, token_matches=[])
            intents = inferrer.infer_intent(query, match_result)
            
            for intent in intents:
                assert 0.0 <= intent.confidence <= 1.0, \
                    f"Confidence {intent.confidence} out of range for query '{query}'"
    
    def test_infer_intent_all_intent_types(self):
        """Test that all intent types can be inferred."""
        from glyphh.nl.schema_matcher import MatchResult
        
        inferrer = IntentInferrer()
        
        # Test each intent type
        test_cases = [
            ("Find Toyota", "find"),
            ("How many cars?", "count"),
            ("Price greater than 100", "filter"),
            ("Find similar items", "similar"),
        ]
        
        for query, expected_intent in test_cases:
            match_result = MatchResult(query=query, token_matches=[])
            intents = inferrer.infer_intent(query, match_result)
            
            intent_types = {i.intent_type for i in intents}
            assert expected_intent in intent_types, \
                f"Expected '{expected_intent}' intent for query '{query}'"
