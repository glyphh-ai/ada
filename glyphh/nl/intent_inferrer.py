from __future__ import annotations

"""
Intent Inferrer module for the Glyphh SDK.

This module provides intent inference functionality for natural language queries,
determining the user's intent (count, find, filter, similar) based on matched
tokens and query structure.

Classes:
    IntentKeywords: Keywords that indicate specific intents
    InferredIntent: An inferred intent with confidence score

The IntentInferrer class (to be implemented in subsequent tasks) will use these
dataclasses to perform the actual intent inference operations.

Validates: Requirement 4 - Intent Inference from Matches
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from glyphh.nl.schema_matcher import TokenMatch, MatchResult


# Valid intent types supported by the system
VALID_INTENT_TYPES = {"count", "find", "filter", "similar"}


@dataclass
class IntentKeywords:
    """
    Keywords that indicate specific intents.
    
    Contains sets of keywords for each supported intent type. When a query
    contains keywords from a particular set, it indicates that the user
    likely has that intent. The keywords are used by the IntentInferrer
    to detect and classify user intent.
    
    Attributes:
        count: Keywords indicating a count/aggregation intent.
            Examples: "how many", "count", "number of"
            These suggest the user wants to count matching items.
        find: Keywords indicating a find/search intent.
            Examples: "find", "show", "get", "list", "search"
            These suggest the user wants to retrieve matching items.
        filter: Keywords indicating a filter/comparison intent.
            Examples: "greater", "less", "between", "more", "fewer"
            These suggest the user wants to filter by comparison operators.
        similar: Keywords indicating a similarity search intent.
            Examples: "similar", "like", "related"
            These suggest the user wants to find similar items.
    
    Example:
        >>> # Default keywords
        >>> keywords = IntentKeywords()
        >>> "how many" in keywords.count
        True
        >>> "find" in keywords.find
        True
        >>> "greater" in keywords.filter
        True
        >>> "similar" in keywords.similar
        True
        
        >>> # Custom keywords
        >>> custom_keywords = IntentKeywords(
        ...     count={"count", "total", "sum"},
        ...     find={"find", "locate", "retrieve"},
        ...     filter={"where", "having"},
        ...     similar={"like", "matching"}
        ... )
        >>> "total" in custom_keywords.count
        True
        >>> "locate" in custom_keywords.find
        True
    
    Notes:
        - Keywords are case-insensitive during matching (queries are normalized)
        - Multi-word keywords like "how many" are supported
        - Keywords can be customized per model for domain-specific terminology
        - Empty keyword sets are allowed but will result in no matches for that intent
    
    Validates: Requirement 4 - Intent Inference from Matches
    Validates: Requirement 4.2 - WHEN a query contains "how many" or "count", THE SDK SHALL infer intent=count
    Validates: Requirement 4.3 - WHEN a query contains "find", "show", "get", THE SDK SHALL infer intent=find
    Validates: Requirement 4.4 - WHEN a query contains comparison operators, THE SDK SHALL infer intent=filter
    Validates: Requirement 4.5 - THE SDK SHALL support configurable intent keywords per model
    """
    count: Set[str] = field(default_factory=lambda: {"how many", "count", "number of"})
    find: Set[str] = field(default_factory=lambda: {"find", "show", "get", "list", "search"})
    filter: Set[str] = field(default_factory=lambda: {"greater", "less", "between", "more", "fewer"})
    similar: Set[str] = field(default_factory=lambda: {"similar", "like", "related"})
    
    def __post_init__(self) -> None:
        """Validate IntentKeywords fields after initialization.
        
        Validates that all keyword fields are sets. If a field is an iterable
        but not a set, it will be converted to a set. All keywords are also
        normalized to lowercase for consistent matching.
        
        Raises:
            TypeError: If any keyword field is not iterable
        """
        # List of all keyword fields to validate
        keyword_fields = [
            ("count", self.count),
            ("find", self.find),
            ("filter", self.filter),
            ("similar", self.similar),
        ]
        
        for field_name, field_value in keyword_fields:
            # Convert to set if it's an iterable but not already a set
            if not isinstance(field_value, set):
                if hasattr(field_value, '__iter__') and not isinstance(field_value, str):
                    # Convert to set and normalize to lowercase
                    setattr(self, field_name, {str(kw).lower() for kw in field_value})
                else:
                    raise TypeError(
                        f"{field_name} must be a set or iterable, "
                        f"got {type(field_value).__name__}"
                    )
            else:
                # Normalize existing set to lowercase
                setattr(self, field_name, {str(kw).lower() for kw in field_value})
    
    def __repr__(self) -> str:
        """Return a string representation of the IntentKeywords."""
        return (
            f"IntentKeywords("
            f"count={len(self.count)} keywords, "
            f"find={len(self.find)} keywords, "
            f"filter={len(self.filter)} keywords, "
            f"similar={len(self.similar)} keywords)"
        )
    
    def get_all_keywords(self) -> Set[str]:
        """Get all keywords across all intent types.
        
        Returns:
            A set containing all keywords from all intent types.
        
        Example:
            >>> keywords = IntentKeywords()
            >>> all_kw = keywords.get_all_keywords()
            >>> "find" in all_kw
            True
            >>> "count" in all_kw
            True
        """
        return self.count | self.find | self.filter | self.similar
    
    def get_intent_for_keyword(self, keyword: str) -> str | None:
        """Get the intent type for a given keyword.
        
        Args:
            keyword: The keyword to look up (case-insensitive)
        
        Returns:
            The intent type ("count", "find", "filter", "similar") if the
            keyword is found, or None if not found.
        
        Example:
            >>> keywords = IntentKeywords()
            >>> keywords.get_intent_for_keyword("find")
            'find'
            >>> keywords.get_intent_for_keyword("how many")
            'count'
            >>> keywords.get_intent_for_keyword("unknown")
            None
        """
        keyword_lower = keyword.lower()
        
        if keyword_lower in self.count:
            return "count"
        if keyword_lower in self.find:
            return "find"
        if keyword_lower in self.filter:
            return "filter"
        if keyword_lower in self.similar:
            return "similar"
        
        return None


@dataclass
class InferredIntent:
    """
    An inferred intent with confidence.
    
    Represents a single inferred intent from a query, including the intent type,
    confidence score, matched keywords that triggered the inference, and any
    supporting schema matches that provide context.
    
    Attributes:
        intent_type: The type of intent inferred. Must be one of:
            - "count": User wants to count matching items
            - "find": User wants to find/retrieve matching items
            - "filter": User wants to filter by comparison operators
            - "similar": User wants to find similar items
        confidence: Confidence score for this intent, in the range [0.0, 1.0].
            Higher values indicate greater confidence in the inference.
            - 1.0: Very high confidence (multiple strong signals)
            - 0.7-0.9: High confidence (clear keyword match)
            - 0.4-0.6: Medium confidence (partial signals)
            - 0.1-0.3: Low confidence (weak signals)
            - 0.0: No confidence (should not typically be returned)
        matched_keywords: List of keywords from the query that matched the
            intent type. For example, if the query contains "find" and "show",
            both would be in this list for a "find" intent.
        supporting_matches: List of TokenMatch objects that provide additional
            context for the intent. These are schema matches (role/value matches)
            that support the inferred intent. For example, if the user asks
            "find Toyota", the match for "Toyota" would be a supporting match.
    
    Example:
        >>> from glyphh.nl.schema_matcher import TokenMatch
        >>> 
        >>> # Create an inferred intent
        >>> intent = InferredIntent(
        ...     intent_type="find",
        ...     confidence=0.85,
        ...     matched_keywords=["find", "show"],
        ...     supporting_matches=[]  # Would contain TokenMatch objects
        ... )
        >>> intent.intent_type
        'find'
        >>> intent.confidence
        0.85
        >>> intent.matched_keywords
        ['find', 'show']
        
        >>> # High confidence count intent
        >>> count_intent = InferredIntent(
        ...     intent_type="count",
        ...     confidence=0.95,
        ...     matched_keywords=["how many"],
        ...     supporting_matches=[]
        ... )
        >>> count_intent.intent_type
        'count'
        
        >>> # Invalid intent type raises ValueError
        >>> InferredIntent(
        ...     intent_type="invalid",
        ...     confidence=0.5,
        ...     matched_keywords=[],
        ...     supporting_matches=[]
        ... )
        ValueError: intent_type must be one of {'count', 'find', 'filter', 'similar'}, got 'invalid'
        
        >>> # Invalid confidence raises ValueError
        >>> InferredIntent(
        ...     intent_type="find",
        ...     confidence=1.5,  # Out of range
        ...     matched_keywords=[],
        ...     supporting_matches=[]
        ... )
        ValueError: confidence must be between 0.0 and 1.0, got 1.5
    
    Notes:
        - The intent_type must be one of the valid types (count, find, filter, similar)
        - Confidence must be in the range [0.0, 1.0]
        - matched_keywords may be empty if intent was inferred from context alone
        - supporting_matches provides context but is not required for inference
        - Multiple InferredIntent objects may be returned for ambiguous queries
    
    Validates: Requirement 4 - Intent Inference from Matches
    Validates: Requirement 4.6 - THE SDK SHALL return confidence scores for inferred intents
    """
    intent_type: str
    confidence: float
    matched_keywords: List[str]
    supporting_matches: List['TokenMatch']
    
    def __post_init__(self) -> None:
        """Validate InferredIntent fields after initialization.
        
        Validates that:
        - intent_type is one of the valid intent types
        - confidence is in the range [0.0, 1.0]
        - matched_keywords is a list
        - supporting_matches is a list
        
        Raises:
            ValueError: If intent_type is not one of the valid types
            ValueError: If confidence is not in range [0.0, 1.0]
            TypeError: If matched_keywords is not a list
            TypeError: If supporting_matches is not a list
            TypeError: If confidence is not a number
        """
        # Validate intent_type is one of the valid types
        if self.intent_type not in VALID_INTENT_TYPES:
            raise ValueError(
                f"intent_type must be one of {VALID_INTENT_TYPES}, "
                f"got '{self.intent_type}'"
            )
        
        # Validate confidence is a number
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"confidence must be a number, "
                f"got {type(self.confidence).__name__}"
            )
        
        # Validate confidence is in valid range [0.0, 1.0]
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, "
                f"got {self.confidence}"
            )
        
        # Validate matched_keywords is a list
        if not isinstance(self.matched_keywords, list):
            raise TypeError(
                f"matched_keywords must be a list, "
                f"got {type(self.matched_keywords).__name__}"
            )
        
        # Validate supporting_matches is a list
        if not isinstance(self.supporting_matches, list):
            raise TypeError(
                f"supporting_matches must be a list, "
                f"got {type(self.supporting_matches).__name__}"
            )
        
        # Note: We don't validate that supporting_matches contains TokenMatch objects
        # here to avoid circular import issues. The IntentInferrer class will handle
        # proper type checking when creating InferredIntent objects.
    
    def __repr__(self) -> str:
        """Return a string representation of the InferredIntent."""
        return (
            f"InferredIntent("
            f"intent_type='{self.intent_type}', "
            f"confidence={self.confidence:.4f}, "
            f"matched_keywords={self.matched_keywords}, "
            f"supporting_matches={len(self.supporting_matches)})"
        )
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if this intent has high confidence.
        
        Args:
            threshold: The confidence threshold for "high confidence".
                Default is 0.7.
        
        Returns:
            True if confidence >= threshold, False otherwise.
        
        Example:
            >>> intent = InferredIntent(
            ...     intent_type="find",
            ...     confidence=0.85,
            ...     matched_keywords=["find"],
            ...     supporting_matches=[]
            ... )
            >>> intent.is_high_confidence()
            True
            >>> intent.is_high_confidence(threshold=0.9)
            False
        """
        return self.confidence >= threshold
    
    def has_keyword_support(self) -> bool:
        """Check if this intent has keyword support.
        
        Returns:
            True if matched_keywords is non-empty, False otherwise.
        
        Example:
            >>> intent = InferredIntent(
            ...     intent_type="find",
            ...     confidence=0.85,
            ...     matched_keywords=["find"],
            ...     supporting_matches=[]
            ... )
            >>> intent.has_keyword_support()
            True
            
            >>> intent_no_keywords = InferredIntent(
            ...     intent_type="find",
            ...     confidence=0.5,
            ...     matched_keywords=[],
            ...     supporting_matches=[]
            ... )
            >>> intent_no_keywords.has_keyword_support()
            False
        """
        return len(self.matched_keywords) > 0


class IntentInferrer:
    """
    Infers intent from matches and query structure.
    
    The IntentInferrer analyzes query tokens and schema matches to determine
    the user's intent. It uses configurable keywords to detect intent types
    (count, find, filter, similar) and computes confidence scores based on
    keyword matches and supporting schema matches.
    
    The inference process works as follows:
    1. Detect intent keywords in the query (e.g., "find", "how many")
    2. Analyze schema matches to understand what the user is asking about
    3. Compute confidence scores for each potential intent
    4. Return ranked list of inferred intents
    
    Attributes:
        keywords: IntentKeywords instance containing keyword sets for each
            intent type. If not provided, default keywords are used:
            - count: "how many", "count", "number of"
            - find: "find", "show", "get", "list", "search"
            - filter: "greater", "less", "between", "more", "fewer"
            - similar: "similar", "like", "related"
    
    Example:
        >>> # Create inferrer with default keywords
        >>> inferrer = IntentInferrer()
        >>> inferrer.keywords.count
        {'how many', 'count', 'number of'}
        >>> inferrer.keywords.find
        {'find', 'show', 'get', 'list', 'search'}
        
        >>> # Create inferrer with custom keywords
        >>> custom_keywords = IntentKeywords(
        ...     count={"count", "total", "sum"},
        ...     find={"find", "locate", "retrieve"},
        ...     filter={"where", "having"},
        ...     similar={"like", "matching"}
        ... )
        >>> inferrer = IntentInferrer(keywords=custom_keywords)
        >>> "total" in inferrer.keywords.count
        True
        >>> "locate" in inferrer.keywords.find
        True
        
        >>> # Verify default keywords are used when None is passed
        >>> inferrer = IntentInferrer(keywords=None)
        >>> "find" in inferrer.keywords.find
        True
        >>> "how many" in inferrer.keywords.count
        True
    
    Notes:
        - The inferrer is stateless - it does not cache results between calls
        - Keywords are case-insensitive during matching
        - Multi-word keywords like "how many" are supported
        - Custom keywords can be provided for domain-specific terminology
        - The inferrer works in conjunction with SchemaMatcher to provide
          context-aware intent inference
    
    Validates: Requirement 4 - Intent Inference from Matches
    Validates: Requirement 4.2 - WHEN a query contains "how many" or "count", THE SDK SHALL infer intent=count
    Validates: Requirement 4.3 - WHEN a query contains "find", "show", "get", THE SDK SHALL infer intent=find
    Validates: Requirement 4.4 - WHEN a query contains comparison operators, THE SDK SHALL infer intent=filter
    Validates: Requirement 4.5 - THE SDK SHALL support configurable intent keywords per model
    """
    
    def __init__(self, keywords: IntentKeywords = None) -> None:
        """
        Initialize the IntentInferrer with optional custom keywords.
        
        Creates a new IntentInferrer instance with the specified keywords for
        intent detection. If no keywords are provided, default keywords are
        used that cover common query patterns:
        
        Default keywords:
            - count: "how many", "count", "number of"
            - find: "find", "show", "get", "list", "search"
            - filter: "greater", "less", "between", "more", "fewer"
            - similar: "similar", "like", "related"
        
        Args:
            keywords: Optional IntentKeywords instance containing custom keyword
                sets for each intent type. If None, default IntentKeywords()
                is used with standard keywords.
        
        Raises:
            TypeError: If keywords is provided but is not an IntentKeywords
                instance or None.
        
        Example:
            >>> # Default keywords
            >>> inferrer = IntentInferrer()
            >>> "find" in inferrer.keywords.find
            True
            >>> "how many" in inferrer.keywords.count
            True
            >>> "greater" in inferrer.keywords.filter
            True
            >>> "similar" in inferrer.keywords.similar
            True
            
            >>> # Custom keywords
            >>> custom = IntentKeywords(
            ...     count={"count", "total"},
            ...     find={"find", "locate"},
            ...     filter={"where"},
            ...     similar={"like"}
            ... )
            >>> inferrer = IntentInferrer(keywords=custom)
            >>> "total" in inferrer.keywords.count
            True
            >>> "locate" in inferrer.keywords.find
            True
            
            >>> # None uses defaults
            >>> inferrer = IntentInferrer(keywords=None)
            >>> "search" in inferrer.keywords.find
            True
            
            >>> # Invalid type raises TypeError
            >>> inferrer = IntentInferrer(keywords="invalid")
            Traceback (most recent call last):
                ...
            TypeError: keywords must be an IntentKeywords instance or None, got str
        
        Validates: Requirements 4.2, 4.3, 4.4, 4.5
        """
        # Validate keywords parameter type
        if keywords is not None and not isinstance(keywords, IntentKeywords):
            raise TypeError(
                f"keywords must be an IntentKeywords instance or None, "
                f"got {type(keywords).__name__}"
            )
        
        # Use provided keywords or create default IntentKeywords
        # Default IntentKeywords contains:
        #   count: {"how many", "count", "number of"}
        #   find: {"find", "show", "get", "list", "search"}
        #   filter: {"greater", "less", "between", "more", "fewer"}
        #   similar: {"similar", "like", "related"}
        self.keywords: IntentKeywords = keywords if keywords is not None else IntentKeywords()
    
    def __repr__(self) -> str:
        """Return a string representation of the IntentInferrer."""
        return f"IntentInferrer(keywords={self.keywords!r})"
    
    def compute_intent_confidence(
        self,
        intent_type: str,
        keywords: List[str],
        matches: List['TokenMatch']
    ) -> float:
        """
        Compute confidence for an intent.
        
        Calculates a confidence score for an inferred intent based on the number
        and quality of keyword matches and supporting schema matches. The confidence
        score is always in the range [0.0, 1.0].
        
        The confidence calculation uses the following formula:
        1. Base confidence from keywords: 0.3 per keyword (max 0.6)
        2. Bonus from schema matches: 0.1 per match (max 0.3)
        3. Bonus from high-quality matches: 0.1 if average similarity > 0.7
        
        The final confidence is clamped to [0.0, 1.0].
        
        Args:
            intent_type: The type of intent being evaluated. Must be one of
                "count", "find", "filter", or "similar".
            keywords: List of matched keywords that triggered this intent.
                More keywords indicate higher confidence.
            matches: List of TokenMatch objects representing supporting schema
                matches. More matches and higher similarity scores increase
                confidence.
        
        Returns:
            A float in the range [0.0, 1.0] representing the confidence score
            for this intent. Higher values indicate greater confidence.
            - 0.0: No confidence (no keywords or matches)
            - 0.3-0.6: Low to medium confidence (keywords only)
            - 0.6-0.9: High confidence (keywords + matches)
            - 1.0: Maximum confidence (multiple keywords + high-quality matches)
        
        Raises:
            ValueError: If intent_type is not one of the valid intent types.
            TypeError: If keywords is not a list.
            TypeError: If matches is not a list.
        
        Example:
            >>> from glyphh.nl.schema_matcher import TokenMatch
            >>> 
            >>> inferrer = IntentInferrer()
            >>> 
            >>> # Low confidence: single keyword, no matches
            >>> confidence = inferrer.compute_intent_confidence(
            ...     intent_type="find",
            ...     keywords=["find"],
            ...     matches=[]
            ... )
            >>> 0.0 <= confidence <= 1.0
            True
            >>> confidence
            0.3
            >>> 
            >>> # Medium confidence: two keywords, no matches
            >>> confidence = inferrer.compute_intent_confidence(
            ...     intent_type="find",
            ...     keywords=["find", "show"],
            ...     matches=[]
            ... )
            >>> confidence
            0.6
            >>> 
            >>> # High confidence: keywords + matches
            >>> # (with mock TokenMatch objects)
            >>> confidence = inferrer.compute_intent_confidence(
            ...     intent_type="find",
            ...     keywords=["find"],
            ...     matches=[mock_match1, mock_match2]
            ... )
            >>> confidence >= 0.3  # At least keyword confidence
            True
            >>> 
            >>> # No keywords or matches: zero confidence
            >>> confidence = inferrer.compute_intent_confidence(
            ...     intent_type="find",
            ...     keywords=[],
            ...     matches=[]
            ... )
            >>> confidence
            0.0
        
        Notes:
            - The confidence formula is designed to reward both keyword matches
              and supporting schema matches
            - Keywords provide the base confidence (up to 0.6)
            - Schema matches provide additional confidence (up to 0.3)
            - High-quality matches (avg similarity > 0.7) provide a bonus (0.1)
            - The result is always clamped to [0.0, 1.0]
            - An empty keywords list with matches still contributes to confidence
              through the match bonus
        
        Validates: Requirement 4.6 - THE SDK SHALL return confidence scores for inferred intents
        """
        # Validate intent_type
        if intent_type not in VALID_INTENT_TYPES:
            raise ValueError(
                f"intent_type must be one of {VALID_INTENT_TYPES}, "
                f"got '{intent_type}'"
            )
        
        # Validate keywords is a list
        if not isinstance(keywords, list):
            raise TypeError(
                f"keywords must be a list, got {type(keywords).__name__}"
            )
        
        # Validate matches is a list
        if not isinstance(matches, list):
            raise TypeError(
                f"matches must be a list, got {type(matches).__name__}"
            )
        
        # Constants for confidence calculation
        KEYWORD_CONFIDENCE = 0.3  # Confidence per keyword
        MAX_KEYWORD_CONFIDENCE = 0.6  # Maximum confidence from keywords
        MATCH_CONFIDENCE = 0.1  # Confidence per schema match
        MAX_MATCH_CONFIDENCE = 0.3  # Maximum confidence from matches
        HIGH_QUALITY_THRESHOLD = 0.7  # Threshold for high-quality match bonus
        HIGH_QUALITY_BONUS = 0.1  # Bonus for high-quality matches
        
        # Calculate base confidence from keywords
        # 0.3 per keyword, capped at 0.6
        keyword_count = len(keywords)
        keyword_confidence = min(keyword_count * KEYWORD_CONFIDENCE, MAX_KEYWORD_CONFIDENCE)
        
        # Calculate bonus from schema matches
        # 0.1 per match, capped at 0.3
        match_count = len(matches)
        match_confidence = min(match_count * MATCH_CONFIDENCE, MAX_MATCH_CONFIDENCE)
        
        # Calculate bonus from high-quality matches
        # 0.1 if average similarity > 0.7
        high_quality_bonus = 0.0
        if matches:
            # Calculate average similarity score
            total_similarity = sum(match.similarity for match in matches)
            avg_similarity = total_similarity / len(matches)
            
            if avg_similarity > HIGH_QUALITY_THRESHOLD:
                high_quality_bonus = HIGH_QUALITY_BONUS
        
        # Calculate total confidence
        total_confidence = keyword_confidence + match_confidence + high_quality_bonus
        
        # Clamp to [0.0, 1.0]
        clamped_confidence = max(0.0, min(1.0, total_confidence))
        
        return clamped_confidence
    
    def detect_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Detect intent keywords in query.
        
        Scans the query (case-insensitive) for keywords from each intent type
        and returns a dictionary mapping intent types to lists of matched keywords.
        Multi-word keywords like "how many" and "number of" are supported.
        
        Args:
            query: The natural language query to scan for intent keywords.
        
        Returns:
            A dictionary mapping intent types ("count", "find", "filter", "similar")
            to lists of matched keywords found in the query. Only intent types with
            at least one matched keyword are included in the result.
        
        Example:
            >>> inferrer = IntentInferrer()
            >>> inferrer.detect_keywords("How many Toyota cars?")
            {'count': ['how many']}
            >>> inferrer.detect_keywords("Find similar Toyota")
            {'find': ['find'], 'similar': ['similar']}
            >>> inferrer.detect_keywords("Show me cars with greater price")
            {'find': ['show'], 'filter': ['greater']}
            >>> inferrer.detect_keywords("Hello world")
            {}
        
        Notes:
            - Matching is case-insensitive
            - Multi-word keywords are matched as phrases
            - A keyword can only match once per query
            - The order of keywords in the result list reflects the order they
              appear in the query
        
        Validates: Requirements 4.2, 4.3, 4.4
        """
        # Normalize query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Result dictionary mapping intent types to matched keywords
        result: Dict[str, List[str]] = {}
        
        # Define intent types and their corresponding keyword sets
        intent_keyword_sets = [
            ("count", self.keywords.count),
            ("find", self.keywords.find),
            ("filter", self.keywords.filter),
            ("similar", self.keywords.similar),
        ]
        
        # For each intent type, find all matching keywords in the query
        for intent_type, keyword_set in intent_keyword_sets:
            matched_keywords: List[str] = []
            
            # Sort keywords by length (descending) to match longer phrases first
            # This ensures "how many" is matched before "how" if both exist
            sorted_keywords = sorted(keyword_set, key=len, reverse=True)
            
            # Track positions that have been matched to avoid overlapping matches
            matched_positions: Set[int] = set()
            
            for keyword in sorted_keywords:
                # Find all occurrences of this keyword in the query
                start_pos = 0
                while True:
                    pos = query_lower.find(keyword, start_pos)
                    if pos == -1:
                        break
                    
                    end_pos = pos + len(keyword)
                    
                    # Check if this position overlaps with already matched positions
                    position_range = set(range(pos, end_pos))
                    if not position_range & matched_positions:
                        # Check word boundaries to avoid partial word matches
                        # e.g., "find" should not match "finding"
                        is_word_start = pos == 0 or not query_lower[pos - 1].isalnum()
                        is_word_end = end_pos == len(query_lower) or not query_lower[end_pos].isalnum()
                        
                        if is_word_start and is_word_end:
                            matched_keywords.append(keyword)
                            matched_positions.update(position_range)
                            # Only match each keyword once per query
                            break
                    
                    start_pos = pos + 1
            
            # Only add to result if there are matched keywords
            if matched_keywords:
                # Sort by position in query for consistent ordering
                matched_keywords_with_pos = [
                    (query_lower.find(kw), kw) for kw in matched_keywords
                ]
                matched_keywords_with_pos.sort(key=lambda x: x[0])
                result[intent_type] = [kw for _, kw in matched_keywords_with_pos]
        
        return result

    def infer_intent(self, query: str, match_result: 'MatchResult') -> List[InferredIntent]:
        """
        Infer intent(s) from query and matches.
        
        Analyzes the query for intent keywords and combines this with schema
        matches to infer the user's intent. Returns a list of possible intents
        sorted by confidence (highest first).
        
        The inference process:
        1. Detect intent keywords in the query using detect_keywords()
        2. For each detected intent type, compute confidence using
           compute_intent_confidence()
        3. Create InferredIntent objects with the intent type, confidence,
           matched keywords, and supporting schema matches
        4. Sort by confidence (descending) and return
        
        Args:
            query: The natural language query to analyze for intent.
            match_result: The MatchResult from schema matching, containing
                token matches that provide context for intent inference.
                The token_matches from this result are used as supporting
                matches for the inferred intents.
        
        Returns:
            A list of InferredIntent objects sorted by confidence (highest
            first). If no intents are detected (no keywords found), returns
            an empty list.
        
        Raises:
            TypeError: If query is not a string.
            TypeError: If match_result is not a MatchResult instance.
        
        Example:
            >>> from glyphh.nl.schema_matcher import MatchResult
            >>> 
            >>> inferrer = IntentInferrer()
            >>> 
            >>> # Query with find intent
            >>> match_result = MatchResult(query="Find Toyota cars", token_matches=[])
            >>> intents = inferrer.infer_intent("Find Toyota cars", match_result)
            >>> len(intents) >= 1
            True
            >>> intents[0].intent_type
            'find'
            >>> intents[0].confidence > 0
            True
            >>> 
            >>> # Query with multiple intents
            >>> match_result = MatchResult(query="Find similar Toyota", token_matches=[])
            >>> intents = inferrer.infer_intent("Find similar Toyota", match_result)
            >>> len(intents) >= 2
            True
            >>> # Sorted by confidence (highest first)
            >>> all(intents[i].confidence >= intents[i+1].confidence 
            ...     for i in range(len(intents)-1))
            True
            >>> 
            >>> # Query with no intent keywords
            >>> match_result = MatchResult(query="Toyota cars", token_matches=[])
            >>> intents = inferrer.infer_intent("Toyota cars", match_result)
            >>> intents
            []
            >>> 
            >>> # Query with count intent
            >>> match_result = MatchResult(query="How many Toyota cars?", token_matches=[])
            >>> intents = inferrer.infer_intent("How many Toyota cars?", match_result)
            >>> intents[0].intent_type
            'count'
            >>> "how many" in intents[0].matched_keywords
            True
        
        Notes:
            - The method uses detect_keywords() to find intent keywords
            - Confidence is computed using compute_intent_confidence()
            - Supporting matches are the token_matches from the MatchResult
            - Results are sorted by confidence in descending order
            - An empty list is returned if no intent keywords are found
            - Multiple intents may be returned for ambiguous queries
        
        Validates: Requirement 4.1 - THE SDK SHALL infer intent based on query keywords and matched roles
        Validates: Requirement 4.5 - THE SDK SHALL support configurable intent keywords per model
        """
        # Import MatchResult here to avoid circular imports
        from glyphh.nl.schema_matcher import MatchResult
        
        # Validate query is a string
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__}"
            )
        
        # Validate match_result is a MatchResult instance
        if not isinstance(match_result, MatchResult):
            raise TypeError(
                f"match_result must be a MatchResult instance, "
                f"got {type(match_result).__name__}"
            )
        
        # Step 1: Detect intent keywords in the query
        detected_keywords = self.detect_keywords(query)
        
        # If no keywords detected, return empty list
        if not detected_keywords:
            return []
        
        # Step 2: For each detected intent type, compute confidence and create InferredIntent
        inferred_intents: List[InferredIntent] = []
        
        for intent_type, keywords in detected_keywords.items():
            # Get supporting matches from the match_result
            # Supporting matches are the token matches that provide context
            supporting_matches = match_result.token_matches
            
            # Compute confidence for this intent
            confidence = self.compute_intent_confidence(
                intent_type=intent_type,
                keywords=keywords,
                matches=supporting_matches
            )
            
            # Create InferredIntent object
            inferred_intent = InferredIntent(
                intent_type=intent_type,
                confidence=confidence,
                matched_keywords=keywords,
                supporting_matches=supporting_matches
            )
            
            inferred_intents.append(inferred_intent)
        
        # Step 3: Sort by confidence (descending)
        inferred_intents.sort(key=lambda x: x.confidence, reverse=True)
        
        return inferred_intents