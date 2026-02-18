"""
Property-based tests for IntentInferrer.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for intent inference.

**Validates: Property 10** - Intent Keyword Detection
For any query containing intent keywords (count keywords → count intent,
find keywords → find intent, filter keywords → filter intent), the inferred
intent SHALL match the keyword category with confidence > 0.

**Validates: Requirements 4.2, 4.3, 4.4**
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from glyphh.nl.intent_inferrer import IntentInferrer, IntentKeywords, InferredIntent
from glyphh.nl.schema_matcher import MatchResult


# Generator Strategies

# Intent keyword categories with their default keywords
COUNT_KEYWORDS = ["how many", "count", "number of"]
FIND_KEYWORDS = ["find", "show", "get", "list", "search"]
FILTER_KEYWORDS = ["greater", "less", "between", "more", "fewer"]
SIMILAR_KEYWORDS = ["similar", "like", "related"]

# Strategy for generating random text that doesn't contain intent keywords
non_keyword_text = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'Zs')),
    min_size=0, max_size=30
).filter(lambda x: not any(
    kw in x.lower() for kw in COUNT_KEYWORDS + FIND_KEYWORDS + FILTER_KEYWORDS + SIMILAR_KEYWORDS
))

# Strategy for generating random filler words
filler_words = st.sampled_from([
    "the", "a", "an", "some", "all", "my", "your", "our",
    "Toyota", "Honda", "cars", "items", "products", "data",
    "red", "blue", "large", "small", "new", "old",
    "please", "now", "today", "quickly", "slowly"
])


class TestIntentKeywordDetection:
    """
    Property tests for Intent Keyword Detection (Property 10).
    
    **Validates: Property 10** - Intent Keyword Detection
    For any query containing intent keywords (count keywords → count intent,
    find keywords → find intent, filter keywords → filter intent), the inferred
    intent SHALL match the keyword category with confidence > 0.
    
    **Validates: Requirements 4.2, 4.3, 4.4**
    """
    
    @given(
        count_keyword=st.sampled_from(COUNT_KEYWORDS),
        prefix=st.lists(filler_words, min_size=0, max_size=3),
        suffix=st.lists(filler_words, min_size=0, max_size=3)
    )
    @settings(max_examples=100)
    def test_count_keywords_produce_count_intent(
        self, count_keyword: str, prefix: list, suffix: list
    ):
        """
        Property test: Queries with count keywords produce count intent.
        
        For any query containing a count keyword ("how many", "count", "number of"),
        the inferred intent SHALL include a count intent with confidence > 0.
        
        **Validates: Property 10**
        **Validates: Requirement 4.2**
        """
        # Build query with count keyword
        query_parts = prefix + [count_keyword] + suffix
        query = " ".join(query_parts)
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create empty match result (no schema matches needed for keyword detection)
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: Count keyword should produce count intent with confidence > 0
        count_intents = [i for i in intents if i.intent_type == "count"]
        assert len(count_intents) >= 1, \
            f"Query '{query}' with count keyword '{count_keyword}' should produce count intent"
        
        # Verify confidence > 0
        assert count_intents[0].confidence > 0, \
            f"Count intent confidence should be > 0, got {count_intents[0].confidence}"
        
        # Verify the keyword was matched
        assert count_keyword in count_intents[0].matched_keywords, \
            f"Count keyword '{count_keyword}' should be in matched_keywords"
    
    @given(
        find_keyword=st.sampled_from(FIND_KEYWORDS),
        prefix=st.lists(filler_words, min_size=0, max_size=3),
        suffix=st.lists(filler_words, min_size=0, max_size=3)
    )
    @settings(max_examples=100)
    def test_find_keywords_produce_find_intent(
        self, find_keyword: str, prefix: list, suffix: list
    ):
        """
        Property test: Queries with find keywords produce find intent.
        
        For any query containing a find keyword ("find", "show", "get", "list", "search"),
        the inferred intent SHALL include a find intent with confidence > 0.
        
        **Validates: Property 10**
        **Validates: Requirement 4.3**
        """
        # Build query with find keyword
        query_parts = prefix + [find_keyword] + suffix
        query = " ".join(query_parts)
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: Find keyword should produce find intent with confidence > 0
        find_intents = [i for i in intents if i.intent_type == "find"]
        assert len(find_intents) >= 1, \
            f"Query '{query}' with find keyword '{find_keyword}' should produce find intent"
        
        # Verify confidence > 0
        assert find_intents[0].confidence > 0, \
            f"Find intent confidence should be > 0, got {find_intents[0].confidence}"
        
        # Verify the keyword was matched
        assert find_keyword in find_intents[0].matched_keywords, \
            f"Find keyword '{find_keyword}' should be in matched_keywords"
    
    @given(
        filter_keyword=st.sampled_from(FILTER_KEYWORDS),
        prefix=st.lists(filler_words, min_size=0, max_size=3),
        suffix=st.lists(filler_words, min_size=0, max_size=3)
    )
    @settings(max_examples=100)
    def test_filter_keywords_produce_filter_intent(
        self, filter_keyword: str, prefix: list, suffix: list
    ):
        """
        Property test: Queries with filter keywords produce filter intent.
        
        For any query containing a filter keyword ("greater", "less", "between", "more", "fewer"),
        the inferred intent SHALL include a filter intent with confidence > 0.
        
        **Validates: Property 10**
        **Validates: Requirement 4.4**
        """
        # Build query with filter keyword
        query_parts = prefix + [filter_keyword] + suffix
        query = " ".join(query_parts)
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: Filter keyword should produce filter intent with confidence > 0
        filter_intents = [i for i in intents if i.intent_type == "filter"]
        assert len(filter_intents) >= 1, \
            f"Query '{query}' with filter keyword '{filter_keyword}' should produce filter intent"
        
        # Verify confidence > 0
        assert filter_intents[0].confidence > 0, \
            f"Filter intent confidence should be > 0, got {filter_intents[0].confidence}"
        
        # Verify the keyword was matched
        assert filter_keyword in filter_intents[0].matched_keywords, \
            f"Filter keyword '{filter_keyword}' should be in matched_keywords"
    
    @given(
        similar_keyword=st.sampled_from(SIMILAR_KEYWORDS),
        prefix=st.lists(filler_words, min_size=0, max_size=3),
        suffix=st.lists(filler_words, min_size=0, max_size=3)
    )
    @settings(max_examples=100)
    def test_similar_keywords_produce_similar_intent(
        self, similar_keyword: str, prefix: list, suffix: list
    ):
        """
        Property test: Queries with similar keywords produce similar intent.
        
        For any query containing a similar keyword ("similar", "like", "related"),
        the inferred intent SHALL include a similar intent with confidence > 0.
        
        **Validates: Property 10**
        **Validates: Requirements 4.2, 4.3, 4.4** (similar intent follows same pattern)
        """
        # Build query with similar keyword
        query_parts = prefix + [similar_keyword] + suffix
        query = " ".join(query_parts)
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: Similar keyword should produce similar intent with confidence > 0
        similar_intents = [i for i in intents if i.intent_type == "similar"]
        assert len(similar_intents) >= 1, \
            f"Query '{query}' with similar keyword '{similar_keyword}' should produce similar intent"
        
        # Verify confidence > 0
        assert similar_intents[0].confidence > 0, \
            f"Similar intent confidence should be > 0, got {similar_intents[0].confidence}"
        
        # Verify the keyword was matched
        assert similar_keyword in similar_intents[0].matched_keywords, \
            f"Similar keyword '{similar_keyword}' should be in matched_keywords"
    
    @given(
        intent_type=st.sampled_from(["count", "find", "filter", "similar"]),
        prefix=st.lists(filler_words, min_size=0, max_size=2),
        suffix=st.lists(filler_words, min_size=0, max_size=2)
    )
    @settings(max_examples=100)
    def test_all_detected_intents_have_positive_confidence(
        self, intent_type: str, prefix: list, suffix: list
    ):
        """
        Property test: All detected intents have confidence > 0.
        
        For any query containing intent keywords, all inferred intents
        SHALL have confidence > 0.
        
        **Validates: Property 10**
        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        # Select keyword based on intent type
        keyword_map = {
            "count": COUNT_KEYWORDS[0],  # "how many"
            "find": FIND_KEYWORDS[0],    # "find"
            "filter": FILTER_KEYWORDS[0], # "greater"
            "similar": SIMILAR_KEYWORDS[0] # "similar"
        }
        keyword = keyword_map[intent_type]
        
        # Build query
        query_parts = prefix + [keyword] + suffix
        query = " ".join(query_parts)
        
        # Create inferrer
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: All detected intents must have confidence > 0
        for intent in intents:
            assert intent.confidence > 0, \
                f"Intent '{intent.intent_type}' should have confidence > 0, got {intent.confidence}"
    
    @given(
        count_keyword=st.sampled_from(COUNT_KEYWORDS),
        find_keyword=st.sampled_from(FIND_KEYWORDS)
    )
    @settings(max_examples=100)
    def test_multiple_keywords_produce_multiple_intents(
        self, count_keyword: str, find_keyword: str
    ):
        """
        Property test: Multiple keywords from different categories produce multiple intents.
        
        For any query containing keywords from multiple intent categories,
        the inferred intents SHALL include all matching categories with confidence > 0.
        
        **Validates: Property 10**
        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        # Build query with both count and find keywords
        query = f"{find_keyword} {count_keyword} Toyota cars"
        
        # Create inferrer
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: Both count and find intents should be detected
        intent_types = {i.intent_type for i in intents}
        
        assert "count" in intent_types, \
            f"Query '{query}' should produce count intent"
        assert "find" in intent_types, \
            f"Query '{query}' should produce find intent"
        
        # Verify all have confidence > 0
        for intent in intents:
            assert intent.confidence > 0, \
                f"Intent '{intent.intent_type}' should have confidence > 0"
    
    @given(
        keyword=st.sampled_from(COUNT_KEYWORDS + FIND_KEYWORDS + FILTER_KEYWORDS + SIMILAR_KEYWORDS)
    )
    @settings(max_examples=100)
    def test_keyword_case_insensitivity(self, keyword: str):
        """
        Property test: Keyword detection is case-insensitive.
        
        For any intent keyword, the detection SHALL work regardless of case
        (uppercase, lowercase, mixed case).
        
        **Validates: Property 10**
        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        # Create inferrer
        inferrer = IntentInferrer()
        
        # Test with different cases
        test_cases = [
            keyword.lower(),
            keyword.upper(),
            keyword.capitalize(),
            keyword.title()
        ]
        
        for test_keyword in test_cases:
            query = f"{test_keyword} Toyota cars"
            match_result = MatchResult(query=query, token_matches=[])
            
            # Infer intent
            intents = inferrer.infer_intent(query, match_result)
            
            # Property: Should detect intent regardless of case
            assert len(intents) >= 1, \
                f"Query '{query}' with keyword '{test_keyword}' should produce at least one intent"
            
            # Verify confidence > 0
            assert intents[0].confidence > 0, \
                f"Intent confidence should be > 0 for keyword '{test_keyword}'"
    
    @given(
        keyword=st.sampled_from(COUNT_KEYWORDS + FIND_KEYWORDS + FILTER_KEYWORDS + SIMILAR_KEYWORDS),
        position=st.sampled_from(["start", "middle", "end"])
    )
    @settings(max_examples=100)
    def test_keyword_position_in_query(self, keyword: str, position: str):
        """
        Property test: Keywords are detected regardless of position in query.
        
        For any intent keyword, the detection SHALL work regardless of where
        the keyword appears in the query (start, middle, or end).
        
        **Validates: Property 10**
        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        # Build query with keyword at different positions
        if position == "start":
            query = f"{keyword} Toyota cars please"
        elif position == "middle":
            query = f"Please {keyword} Toyota cars"
        else:  # end
            query = f"Toyota cars {keyword}"
        
        # Create inferrer
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: Should detect intent regardless of position
        assert len(intents) >= 1, \
            f"Query '{query}' with keyword '{keyword}' at {position} should produce intent"
        
        # Verify confidence > 0
        assert intents[0].confidence > 0, \
            f"Intent confidence should be > 0 for keyword at {position}"
    
    @given(
        custom_count_keywords=st.lists(
            st.text(min_size=2, max_size=15, alphabet=st.characters(whitelist_categories=('L',))),
            min_size=1, max_size=3, unique=True
        ),
        custom_find_keywords=st.lists(
            st.text(min_size=2, max_size=15, alphabet=st.characters(whitelist_categories=('L',))),
            min_size=1, max_size=3, unique=True
        )
    )
    @settings(max_examples=100)
    def test_custom_keywords_produce_correct_intent(
        self, custom_count_keywords: list, custom_find_keywords: list
    ):
        """
        Property test: Custom keywords produce correct intent.
        
        For any custom keyword configuration, queries containing those keywords
        SHALL produce the corresponding intent with confidence > 0.
        
        **Validates: Property 10**
        **Validates: Requirement 4.5** (configurable intent keywords)
        """
        # Filter out empty strings and ensure keywords are valid
        custom_count_keywords = [k.strip().lower() for k in custom_count_keywords if k.strip()]
        custom_find_keywords = [k.strip().lower() for k in custom_find_keywords if k.strip()]
        
        # Skip if no valid keywords
        assume(len(custom_count_keywords) > 0)
        assume(len(custom_find_keywords) > 0)
        
        # Ensure no overlap between keyword sets
        assume(not set(custom_count_keywords) & set(custom_find_keywords))
        
        # Create custom keywords
        custom_keywords = IntentKeywords(
            count=set(custom_count_keywords),
            find=set(custom_find_keywords),
            filter={"customfilter"},
            similar={"customsimilar"}
        )
        
        # Create inferrer with custom keywords
        inferrer = IntentInferrer(keywords=custom_keywords)
        
        # Test count keyword
        count_keyword = custom_count_keywords[0]
        count_query = f"{count_keyword} Toyota cars"
        count_match_result = MatchResult(query=count_query, token_matches=[])
        count_intents = inferrer.infer_intent(count_query, count_match_result)
        
        count_intent_types = {i.intent_type for i in count_intents}
        assert "count" in count_intent_types, \
            f"Custom count keyword '{count_keyword}' should produce count intent"
        
        # Test find keyword
        find_keyword = custom_find_keywords[0]
        find_query = f"{find_keyword} Toyota cars"
        find_match_result = MatchResult(query=find_query, token_matches=[])
        find_intents = inferrer.infer_intent(find_query, find_match_result)
        
        find_intent_types = {i.intent_type for i in find_intents}
        assert "find" in find_intent_types, \
            f"Custom find keyword '{find_keyword}' should produce find intent"


class TestIntentConfidenceBounds:
    """
    Property tests for Intent Confidence Bounds (Property 11).
    
    **Validates: Property 11** - Intent Confidence Bounds
    For any inferred intent, the confidence score SHALL be in the range [0.0, 1.0].
    
    **Validates: Requirement 4.6**
    """
    
    @given(
        keyword=st.sampled_from(COUNT_KEYWORDS + FIND_KEYWORDS + FILTER_KEYWORDS + SIMILAR_KEYWORDS),
        num_keywords=st.integers(min_value=1, max_value=3),
        prefix=st.lists(filler_words, min_size=0, max_size=5),
        suffix=st.lists(filler_words, min_size=0, max_size=5)
    )
    @settings(max_examples=100)
    def test_single_keyword_confidence_bounds(
        self, keyword: str, num_keywords: int, prefix: list, suffix: list
    ):
        """
        Property test: Single keyword queries produce confidence in [0.0, 1.0].
        
        For any query containing a single intent keyword, the inferred intent
        confidence SHALL be in the range [0.0, 1.0].
        
        **Validates: Property 11**
        **Validates: Requirement 4.6**
        """
        # Build query with keyword
        query_parts = prefix + [keyword] + suffix
        query = " ".join(query_parts)
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create empty match result (no schema matches)
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: All inferred intents must have confidence in [0.0, 1.0]
        for intent in intents:
            assert 0.0 <= intent.confidence <= 1.0, \
                f"Intent '{intent.intent_type}' confidence {intent.confidence} " \
                f"should be in [0.0, 1.0] for query '{query}'"
    
    @given(
        count_keyword=st.sampled_from(COUNT_KEYWORDS),
        find_keyword=st.sampled_from(FIND_KEYWORDS),
        filter_keyword=st.sampled_from(FILTER_KEYWORDS),
        similar_keyword=st.sampled_from(SIMILAR_KEYWORDS),
        include_count=st.booleans(),
        include_find=st.booleans(),
        include_filter=st.booleans(),
        include_similar=st.booleans()
    )
    @settings(max_examples=100)
    def test_multiple_keywords_confidence_bounds(
        self, count_keyword: str, find_keyword: str, filter_keyword: str,
        similar_keyword: str, include_count: bool, include_find: bool,
        include_filter: bool, include_similar: bool
    ):
        """
        Property test: Multiple keyword queries produce confidence in [0.0, 1.0].
        
        For any query containing multiple intent keywords from different categories,
        all inferred intent confidences SHALL be in the range [0.0, 1.0].
        
        **Validates: Property 11**
        **Validates: Requirement 4.6**
        """
        # Build query with selected keywords
        keywords = []
        if include_count:
            keywords.append(count_keyword)
        if include_find:
            keywords.append(find_keyword)
        if include_filter:
            keywords.append(filter_keyword)
        if include_similar:
            keywords.append(similar_keyword)
        
        # Skip if no keywords selected
        assume(len(keywords) > 0)
        
        query = " ".join(keywords) + " Toyota cars"
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: All inferred intents must have confidence in [0.0, 1.0]
        for intent in intents:
            assert 0.0 <= intent.confidence <= 1.0, \
                f"Intent '{intent.intent_type}' confidence {intent.confidence} " \
                f"should be in [0.0, 1.0] for query '{query}'"
    
    @given(
        keyword=st.sampled_from(COUNT_KEYWORDS + FIND_KEYWORDS + FILTER_KEYWORDS + SIMILAR_KEYWORDS),
        num_supporting_matches=st.integers(min_value=0, max_value=10),
        similarity_scores=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=0, max_size=10
        )
    )
    @settings(max_examples=100)
    def test_confidence_bounds_with_supporting_matches(
        self, keyword: str, num_supporting_matches: int, similarity_scores: list
    ):
        """
        Property test: Confidence bounds hold with various supporting matches.
        
        For any query with various numbers of supporting matches (0, 1, 5+),
        the inferred intent confidence SHALL be in the range [0.0, 1.0].
        
        **Validates: Property 11**
        **Validates: Requirement 4.6**
        """
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        import numpy as np
        
        query = f"{keyword} Toyota cars"
        
        # Create mock supporting matches with various similarity scores
        supporting_matches = []
        for i in range(min(num_supporting_matches, len(similarity_scores))):
            # Create a mock token
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            
            # Create a mock schema vector with a simple bipolar vector
            mock_vector = np.array([1, -1, 1, -1, 1])
            schema_vector = SchemaVector(
                key=f"value{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"test.role{i}",
                original_value=f"Value{i}"
            )
            
            # Create token match with the generated similarity score
            match = TokenMatch(
                token=token,
                schema_vector=schema_vector,
                similarity=similarity_scores[i],
                match_type="partial"
            )
            supporting_matches.append(match)
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create match result with supporting matches
        match_result = MatchResult(query=query, token_matches=supporting_matches)
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: All inferred intents must have confidence in [0.0, 1.0]
        for intent in intents:
            assert 0.0 <= intent.confidence <= 1.0, \
                f"Intent '{intent.intent_type}' confidence {intent.confidence} " \
                f"should be in [0.0, 1.0] with {len(supporting_matches)} supporting matches"
    
    @given(
        intent_type=st.sampled_from(["count", "find", "filter", "similar"]),
        num_keywords=st.integers(min_value=0, max_value=5),
        num_matches=st.integers(min_value=0, max_value=10),
        avg_similarity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_compute_intent_confidence_bounds(
        self, intent_type: str, num_keywords: int, num_matches: int, avg_similarity: float
    ):
        """
        Property test: compute_intent_confidence always returns value in [0.0, 1.0].
        
        For any combination of intent type, number of keywords, and number of
        supporting matches, compute_intent_confidence SHALL return a value
        in the range [0.0, 1.0].
        
        **Validates: Property 11**
        **Validates: Requirement 4.6**
        """
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        import numpy as np
        
        # Create inferrer
        inferrer = IntentInferrer()
        
        # Generate keywords based on intent type
        keyword_map = {
            "count": COUNT_KEYWORDS,
            "find": FIND_KEYWORDS,
            "filter": FILTER_KEYWORDS,
            "similar": SIMILAR_KEYWORDS
        }
        available_keywords = keyword_map[intent_type]
        keywords = [available_keywords[i % len(available_keywords)] for i in range(num_keywords)]
        
        # Create mock matches with the specified average similarity
        matches = []
        for i in range(num_matches):
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            
            mock_vector = np.array([1, -1, 1, -1, 1])
            schema_vector = SchemaVector(
                key=f"value{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"test.role{i}",
                original_value=f"Value{i}"
            )
            
            match = TokenMatch(
                token=token,
                schema_vector=schema_vector,
                similarity=avg_similarity,
                match_type="partial"
            )
            matches.append(match)
        
        # Compute confidence
        confidence = inferrer.compute_intent_confidence(
            intent_type=intent_type,
            keywords=keywords,
            matches=matches
        )
        
        # Property: Confidence must be in [0.0, 1.0]
        assert 0.0 <= confidence <= 1.0, \
            f"Confidence {confidence} should be in [0.0, 1.0] for " \
            f"intent_type='{intent_type}', {num_keywords} keywords, {num_matches} matches"
    
    @given(
        keyword=st.sampled_from(COUNT_KEYWORDS + FIND_KEYWORDS + FILTER_KEYWORDS + SIMILAR_KEYWORDS),
        num_high_similarity_matches=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=100)
    def test_confidence_bounds_with_high_similarity_matches(
        self, keyword: str, num_high_similarity_matches: int
    ):
        """
        Property test: Confidence bounds hold even with high-similarity matches.
        
        For any query with high-similarity supporting matches (similarity > 0.7),
        the inferred intent confidence SHALL still be in the range [0.0, 1.0].
        
        This tests the high-quality match bonus path in confidence calculation.
        
        **Validates: Property 11**
        **Validates: Requirement 4.6**
        """
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        import numpy as np
        
        query = f"{keyword} Toyota cars"
        
        # Create high-similarity supporting matches
        supporting_matches = []
        for i in range(num_high_similarity_matches):
            token = Token(
                text=f"token{i}",
                original=f"Token{i}",
                position=i,
                is_stop_word=False,
                ngram_size=1
            )
            
            mock_vector = np.array([1, -1, 1, -1, 1])
            schema_vector = SchemaVector(
                key=f"value{i}",
                vector=mock_vector,
                element_type="value",
                role_path=f"test.role{i}",
                original_value=f"Value{i}"
            )
            
            # High similarity score (> 0.7 to trigger high-quality bonus)
            match = TokenMatch(
                token=token,
                schema_vector=schema_vector,
                similarity=0.85,
                match_type="partial"
            )
            supporting_matches.append(match)
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create match result with high-similarity supporting matches
        match_result = MatchResult(query=query, token_matches=supporting_matches)
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: All inferred intents must have confidence in [0.0, 1.0]
        for intent in intents:
            assert 0.0 <= intent.confidence <= 1.0, \
                f"Intent '{intent.intent_type}' confidence {intent.confidence} " \
                f"should be in [0.0, 1.0] with {num_high_similarity_matches} high-similarity matches"
    
    @given(
        num_keywords_per_type=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100)
    def test_confidence_bounds_with_all_intent_types(self, num_keywords_per_type: int):
        """
        Property test: Confidence bounds hold when all intent types are present.
        
        For any query containing keywords from all intent types (count, find,
        filter, similar), all inferred intent confidences SHALL be in [0.0, 1.0].
        
        **Validates: Property 11**
        **Validates: Requirement 4.6**
        """
        # Build query with keywords from all intent types
        keywords = []
        keywords.extend(COUNT_KEYWORDS[:num_keywords_per_type])
        keywords.extend(FIND_KEYWORDS[:num_keywords_per_type])
        keywords.extend(FILTER_KEYWORDS[:num_keywords_per_type])
        keywords.extend(SIMILAR_KEYWORDS[:num_keywords_per_type])
        
        query = " ".join(keywords) + " Toyota cars"
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create empty match result
        match_result = MatchResult(query=query, token_matches=[])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: All inferred intents must have confidence in [0.0, 1.0]
        assert len(intents) >= 4, \
            f"Should have at least 4 intents (one per type), got {len(intents)}"
        
        for intent in intents:
            assert 0.0 <= intent.confidence <= 1.0, \
                f"Intent '{intent.intent_type}' confidence {intent.confidence} " \
                f"should be in [0.0, 1.0] for query with all intent types"
    
    @given(
        keyword=st.sampled_from(COUNT_KEYWORDS + FIND_KEYWORDS + FILTER_KEYWORDS + SIMILAR_KEYWORDS),
        extreme_similarity=st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0])
    )
    @settings(max_examples=100)
    def test_confidence_bounds_with_extreme_similarity_values(
        self, keyword: str, extreme_similarity: float
    ):
        """
        Property test: Confidence bounds hold with extreme similarity values.
        
        For any query with supporting matches having extreme similarity values
        (-1.0, 0.0, 1.0), the inferred intent confidence SHALL be in [0.0, 1.0].
        
        **Validates: Property 11**
        **Validates: Requirement 4.6**
        """
        from glyphh.nl.query_tokenizer import Token
        from glyphh.nl.schema_vectorizer import SchemaVector
        from glyphh.nl.schema_matcher import TokenMatch
        import numpy as np
        
        query = f"{keyword} Toyota cars"
        
        # Create supporting match with extreme similarity
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector = SchemaVector(
            key="make=Toyota",
            vector=mock_vector,
            element_type="value",
            role_path="vehicle.identity.make",
            original_value="Toyota"
        )
        
        match = TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=extreme_similarity,
            match_type="partial"
        )
        
        # Create inferrer with default keywords
        inferrer = IntentInferrer()
        
        # Create match result with extreme similarity match
        match_result = MatchResult(query=query, token_matches=[match])
        
        # Infer intent
        intents = inferrer.infer_intent(query, match_result)
        
        # Property: All inferred intents must have confidence in [0.0, 1.0]
        for intent in intents:
            assert 0.0 <= intent.confidence <= 1.0, \
                f"Intent '{intent.intent_type}' confidence {intent.confidence} " \
                f"should be in [0.0, 1.0] with extreme similarity {extreme_similarity}"
