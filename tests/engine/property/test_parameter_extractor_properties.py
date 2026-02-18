"""
Property-based tests for ParameterExtractor.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for parameter extraction.

**Validates: Property 12** - Parameter Extraction Role Association
For any value match in a query, the extracted parameter SHALL be associated
with the correct role from the schema.

**Validates: Requirements 5.1, 5.2**
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from glyphh.core.config import EncoderConfig
from glyphh.nl.parameter_extractor import ParameterExtractor, ExtractedParameter, ExtractionResult
from glyphh.nl.schema_matcher import MatchResult, TokenMatch
from glyphh.nl.schema_vectorizer import SchemaVector
from glyphh.nl.query_tokenizer import Token


# Generator Strategies

# Role name generator - generates valid role names with letters, numbers, and underscores
role_names = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1, max_size=30
).filter(lambda x: x.strip())

# Value generator - generates arbitrary text values
values = st.text(min_size=1, max_size=50).filter(lambda x: x.strip())

# Role path segment generator - generates valid path segments
path_segments = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1, max_size=20
).filter(lambda x: x.strip())

# Similarity score generator - generates valid similarity scores
similarity_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Match type generator
match_types = st.sampled_from(["exact", "partial", "compound"])


class TestParameterExtractionRoleAssociation:
    """
    Property tests for Parameter Extraction Role Association (Property 12).
    
    **Validates: Property 12** - Parameter Extraction Role Association
    For any value match in a query, the extracted parameter SHALL be associated
    with the correct role from the schema.
    
    **Validates: Requirements 5.1, 5.2**
    """
    
    @given(
        role_name=role_names,
        value=values,
        similarity=similarity_scores,
        match_type=match_types
    )
    @settings(max_examples=100)
    def test_extracted_parameter_role_matches_schema_vector_role_path(
        self, role_name: str, value: str, similarity: float, match_type: str
    ):
        """
        Property test: Extracted parameter role matches the last component of role_path.
        
        For any value match with a schema_vector.role_path, the extracted parameter's
        role field SHALL match the last component of that path.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create a simple role path (just the role name)
        role_path = role_name
        
        # Create a mock token
        token = Token(
            text=value.lower(),
            original=value,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create a mock schema vector with the role path
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector = SchemaVector(
            key=f"{role_name}={value}",
            vector=mock_vector,
            element_type="value",
            role_path=role_path,
            original_value=value
        )
        
        # Create a token match
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=similarity,
            match_type=match_type
        )
        
        # Create a match result with the value match
        match_result = MatchResult(
            query=f"Find {value}",
            token_matches=[token_match],
            value_matches=[token_match]
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: The extracted parameter's role should match the role_path's last component
        expected_role = role_path.split(".")[-1] if "." in role_path else role_path
        
        assert expected_role in result.parameters, \
            f"Expected role '{expected_role}' not found in extracted parameters. " \
            f"Got: {list(result.parameters.keys())}"
        
        extracted_param = result.parameters[expected_role]
        assert extracted_param.role == expected_role, \
            f"Extracted parameter role '{extracted_param.role}' does not match " \
            f"expected role '{expected_role}' from role_path '{role_path}'"
    
    @given(
        layer=path_segments,
        segment=path_segments,
        role=path_segments,
        value=values,
        similarity=similarity_scores,
        match_type=match_types
    )
    @settings(max_examples=100)
    def test_nested_role_path_extracts_correct_role(
        self, layer: str, segment: str, role: str, value: str, 
        similarity: float, match_type: str
    ):
        """
        Property test: Nested role paths extract the correct role (last component).
        
        For any value match with a nested role_path (e.g., "layer.segment.role"),
        the extracted parameter's role field SHALL be the last component of the path.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create a nested role path
        role_path = f"{layer}.{segment}.{role}"
        
        # Create a mock token
        token = Token(
            text=value.lower(),
            original=value,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create a mock schema vector with the nested role path
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector = SchemaVector(
            key=f"{role}={value}",
            vector=mock_vector,
            element_type="value",
            role_path=role_path,
            original_value=value
        )
        
        # Create a token match
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=similarity,
            match_type=match_type
        )
        
        # Create a match result with the value match
        match_result = MatchResult(
            query=f"Find {value}",
            token_matches=[token_match],
            value_matches=[token_match]
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: The extracted parameter's role should be the last component of role_path
        expected_role = role  # Last component of "layer.segment.role"
        
        assert expected_role in result.parameters, \
            f"Expected role '{expected_role}' not found in extracted parameters. " \
            f"Got: {list(result.parameters.keys())}. Role path was: '{role_path}'"
        
        extracted_param = result.parameters[expected_role]
        assert extracted_param.role == expected_role, \
            f"Extracted parameter role '{extracted_param.role}' does not match " \
            f"expected role '{expected_role}' from nested role_path '{role_path}'"
    
    @given(
        num_matches=st.integers(min_value=1, max_value=5),
        base_role=role_names
    )
    @settings(max_examples=100)
    def test_multiple_value_matches_extract_correct_roles(
        self, num_matches: int, base_role: str
    ):
        """
        Property test: Multiple value matches each extract the correct role.
        
        For any set of value matches with different role_paths, each extracted
        parameter SHALL be associated with the correct role from its schema_vector.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create multiple token matches with different roles
        token_matches = []
        value_matches = []
        expected_roles = {}
        
        for i in range(num_matches):
            role_name = f"{base_role}_{i}"
            value = f"value_{i}"
            role_path = f"layer.segment.{role_name}"
            
            # Create a mock token
            token = Token(
                text=value.lower(),
                original=value,
                position=i * 10,
                is_stop_word=False,
                ngram_size=1
            )
            
            # Create a mock schema vector
            mock_vector = np.array([1, -1, 1, -1, 1])
            schema_vector = SchemaVector(
                key=f"{role_name}={value}",
                vector=mock_vector,
                element_type="value",
                role_path=role_path,
                original_value=value
            )
            
            # Create a token match
            token_match = TokenMatch(
                token=token,
                schema_vector=schema_vector,
                similarity=0.8,
                match_type="exact"
            )
            
            token_matches.append(token_match)
            value_matches.append(token_match)
            expected_roles[role_name] = value
        
        # Create a match result with all value matches
        match_result = MatchResult(
            query="Find multiple values",
            token_matches=token_matches,
            value_matches=value_matches
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: Each expected role should be in the extracted parameters
        for expected_role, expected_value in expected_roles.items():
            assert expected_role in result.parameters, \
                f"Expected role '{expected_role}' not found in extracted parameters. " \
                f"Got: {list(result.parameters.keys())}"
            
            extracted_param = result.parameters[expected_role]
            assert extracted_param.role == expected_role, \
                f"Extracted parameter role '{extracted_param.role}' does not match " \
                f"expected role '{expected_role}'"
            assert extracted_param.value == expected_value, \
                f"Extracted parameter value '{extracted_param.value}' does not match " \
                f"expected value '{expected_value}'"
    
    @given(
        role_name=role_names,
        value=values,
        similarity=similarity_scores
    )
    @settings(max_examples=100)
    def test_extracted_value_matches_schema_vector_original_value(
        self, role_name: str, value: str, similarity: float
    ):
        """
        Property test: Extracted parameter value matches schema_vector.original_value.
        
        For any value match, the extracted parameter's value SHALL match the
        original_value from the schema_vector.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create a mock token
        token = Token(
            text=value.lower(),
            original=value,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create a mock schema vector
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector = SchemaVector(
            key=f"{role_name}={value}",
            vector=mock_vector,
            element_type="value",
            role_path=role_name,
            original_value=value
        )
        
        # Create a token match
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=similarity,
            match_type="exact"
        )
        
        # Create a match result with the value match
        match_result = MatchResult(
            query=f"Find {value}",
            token_matches=[token_match],
            value_matches=[token_match]
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: The extracted parameter's value should match original_value
        assert role_name in result.parameters, \
            f"Expected role '{role_name}' not found in extracted parameters"
        
        extracted_param = result.parameters[role_name]
        assert extracted_param.value == value, \
            f"Extracted parameter value '{extracted_param.value}' does not match " \
            f"schema_vector.original_value '{value}'"
    
    @given(
        role_name=role_names,
        value=values,
        similarity=similarity_scores
    )
    @settings(max_examples=100)
    def test_extracted_confidence_matches_similarity_score(
        self, role_name: str, value: str, similarity: float
    ):
        """
        Property test: Extracted parameter confidence matches the similarity score.
        
        For any value match, the extracted parameter's confidence SHALL match
        the similarity score from the token match.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create a mock token
        token = Token(
            text=value.lower(),
            original=value,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create a mock schema vector
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector = SchemaVector(
            key=f"{role_name}={value}",
            vector=mock_vector,
            element_type="value",
            role_path=role_name,
            original_value=value
        )
        
        # Create a token match
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=similarity,
            match_type="exact"
        )
        
        # Create a match result with the value match
        match_result = MatchResult(
            query=f"Find {value}",
            token_matches=[token_match],
            value_matches=[token_match]
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: The extracted parameter's confidence should match similarity
        assert role_name in result.parameters, \
            f"Expected role '{role_name}' not found in extracted parameters"
        
        extracted_param = result.parameters[role_name]
        assert extracted_param.confidence == similarity, \
            f"Extracted parameter confidence {extracted_param.confidence} does not match " \
            f"similarity score {similarity}"
    
    @given(
        role_name=role_names,
        value1=values,
        value2=values,
        similarity1=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
        similarity2=st.floats(min_value=0.51, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_duplicate_roles_keep_highest_confidence(
        self, role_name: str, value1: str, value2: str, 
        similarity1: float, similarity2: float
    ):
        """
        Property test: Duplicate roles keep the match with highest confidence.
        
        For any two value matches with the same role, the extracted parameter
        SHALL be the one with the higher confidence score.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Ensure values are different and similarities are different
        assume(value1 != value2)
        assume(similarity1 != similarity2)
        
        # Create mock tokens
        token1 = Token(
            text=value1.lower(),
            original=value1,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        token2 = Token(
            text=value2.lower(),
            original=value2,
            position=10,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create mock schema vectors with the same role
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector1 = SchemaVector(
            key=f"{role_name}={value1}",
            vector=mock_vector,
            element_type="value",
            role_path=role_name,
            original_value=value1
        )
        schema_vector2 = SchemaVector(
            key=f"{role_name}={value2}",
            vector=mock_vector,
            element_type="value",
            role_path=role_name,
            original_value=value2
        )
        
        # Create token matches with different similarities
        # similarity1 is lower (0.0-0.49), similarity2 is higher (0.51-1.0)
        token_match1 = TokenMatch(
            token=token1,
            schema_vector=schema_vector1,
            similarity=similarity1,
            match_type="exact"
        )
        token_match2 = TokenMatch(
            token=token2,
            schema_vector=schema_vector2,
            similarity=similarity2,
            match_type="exact"
        )
        
        # Create a match result with both value matches
        match_result = MatchResult(
            query=f"Find {value1} or {value2}",
            token_matches=[token_match1, token_match2],
            value_matches=[token_match1, token_match2]
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: The extracted parameter should have the higher confidence
        assert role_name in result.parameters, \
            f"Expected role '{role_name}' not found in extracted parameters"
        
        extracted_param = result.parameters[role_name]
        
        # The higher similarity should win
        expected_value = value2  # value2 has similarity2 which is higher
        expected_confidence = similarity2
        
        assert extracted_param.value == expected_value, \
            f"Extracted parameter value '{extracted_param.value}' should be " \
            f"'{expected_value}' (higher confidence match)"
        assert extracted_param.confidence == expected_confidence, \
            f"Extracted parameter confidence {extracted_param.confidence} should be " \
            f"{expected_confidence} (higher confidence match)"
    
    @given(
        role_name=role_names,
        value=values
    )
    @settings(max_examples=100)
    def test_original_text_preserved_from_token(
        self, role_name: str, value: str
    ):
        """
        Property test: Original text is preserved from the token.
        
        For any value match, the extracted parameter's original_text SHALL
        match the token's original text.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create a mock token with different text and original
        original_text = value
        normalized_text = value.lower()
        
        token = Token(
            text=normalized_text,
            original=original_text,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create a mock schema vector
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector = SchemaVector(
            key=f"{role_name}={value}",
            vector=mock_vector,
            element_type="value",
            role_path=role_name,
            original_value=value
        )
        
        # Create a token match
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=0.9,
            match_type="exact"
        )
        
        # Create a match result with the value match
        match_result = MatchResult(
            query=f"Find {value}",
            token_matches=[token_match],
            value_matches=[token_match]
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: The extracted parameter's original_text should match token.original
        assert role_name in result.parameters, \
            f"Expected role '{role_name}' not found in extracted parameters"
        
        extracted_param = result.parameters[role_name]
        assert extracted_param.original_text == original_text, \
            f"Extracted parameter original_text '{extracted_param.original_text}' " \
            f"does not match token.original '{original_text}'"
    
    @given(
        depth=st.integers(min_value=1, max_value=5),
        role=path_segments,
        value=values
    )
    @settings(max_examples=100)
    def test_variable_depth_role_paths(
        self, depth: int, role: str, value: str
    ):
        """
        Property test: Variable depth role paths extract the correct role.
        
        For any role_path with variable depth (1 to 5 segments), the extracted
        parameter's role SHALL be the last component of the path.
        
        **Validates: Property 12**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create a role path with variable depth
        segments = [f"segment{i}" for i in range(depth - 1)]
        segments.append(role)
        role_path = ".".join(segments)
        
        # Create a mock token
        token = Token(
            text=value.lower(),
            original=value,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create a mock schema vector
        mock_vector = np.array([1, -1, 1, -1, 1])
        schema_vector = SchemaVector(
            key=f"{role}={value}",
            vector=mock_vector,
            element_type="value",
            role_path=role_path,
            original_value=value
        )
        
        # Create a token match
        token_match = TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=0.85,
            match_type="exact"
        )
        
        # Create a match result with the value match
        match_result = MatchResult(
            query=f"Find {value}",
            token_matches=[token_match],
            value_matches=[token_match]
        )
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Property: The extracted parameter's role should be the last component
        expected_role = role
        
        assert expected_role in result.parameters, \
            f"Expected role '{expected_role}' not found in extracted parameters. " \
            f"Role path was: '{role_path}' (depth={depth})"
        
        extracted_param = result.parameters[expected_role]
        assert extracted_param.role == expected_role, \
            f"Extracted parameter role '{extracted_param.role}' does not match " \
            f"expected role '{expected_role}' from role_path '{role_path}'"


class TestMultiValueParameterExtraction:
    """
    Property tests for Multi-Value Parameter Extraction (Property 13).
    
    **Validates: Property 13** - Multi-Value Parameter Extraction
    For any query containing "X or Y" patterns where X and Y are schema values
    for the same role, both values SHALL be extracted as a multi-value parameter
    for that role.
    
    **Validates: Requirements 5.3**
    """
    
    # Generator for valid value strings (non-empty, no "or" keyword)
    value_text = st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
        min_size=1, max_size=20
    ).filter(lambda x: x.strip() and x.lower() != "or")
    
    @given(
        value_x=value_text,
        value_y=value_text
    )
    @settings(max_examples=100)
    def test_x_or_y_pattern_extracts_both_values(
        self, value_x: str, value_y: str
    ):
        """
        Property test: "X or Y" pattern extracts both X and Y values.
        
        For any two values X and Y, creating tokens representing "X or Y"
        and calling handle_multi_value() SHALL return both X and Y.
        
        **Validates: Property 13**
        **Validates: Requirements 5.3**
        """
        # Ensure X and Y are different values
        assume(value_x.lower() != value_y.lower())
        
        # Create tokens for "X or Y" pattern
        tokens = [
            Token(
                text=value_x.lower(),
                original=value_x,
                position=0,
                is_stop_word=False,
                ngram_size=1
            ),
            Token(
                text="or",
                original="or",
                position=len(value_x) + 1,
                is_stop_word=False,
                ngram_size=1
            ),
            Token(
                text=value_y.lower(),
                original=value_y,
                position=len(value_x) + 4,
                is_stop_word=False,
                ngram_size=1
            )
        ]
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract multi-values
        result = extractor.handle_multi_value(tokens)
        
        # Property: Both X and Y should be extracted
        assert len(result) == 2, \
            f"Expected 2 values, got {len(result)}: {result}"
        assert value_x in result, \
            f"Expected '{value_x}' in result, got: {result}"
        assert value_y in result, \
            f"Expected '{value_y}' in result, got: {result}"
    
    @given(
        value_x=value_text,
        value_y=value_text,
        value_z=value_text
    )
    @settings(max_examples=100)
    def test_x_or_y_or_z_pattern_extracts_all_values(
        self, value_x: str, value_y: str, value_z: str
    ):
        """
        Property test: "X or Y or Z" pattern extracts all three values.
        
        For any three values X, Y, and Z, creating tokens representing
        "X or Y or Z" and calling handle_multi_value() SHALL return all three.
        
        **Validates: Property 13**
        **Validates: Requirements 5.3**
        """
        # Ensure all values are different
        assume(value_x.lower() != value_y.lower())
        assume(value_y.lower() != value_z.lower())
        assume(value_x.lower() != value_z.lower())
        
        # Create tokens for "X or Y or Z" pattern
        pos = 0
        tokens = [
            Token(
                text=value_x.lower(),
                original=value_x,
                position=pos,
                is_stop_word=False,
                ngram_size=1
            )
        ]
        pos += len(value_x) + 1
        
        tokens.append(Token(
            text="or",
            original="or",
            position=pos,
            is_stop_word=False,
            ngram_size=1
        ))
        pos += 3
        
        tokens.append(Token(
            text=value_y.lower(),
            original=value_y,
            position=pos,
            is_stop_word=False,
            ngram_size=1
        ))
        pos += len(value_y) + 1
        
        tokens.append(Token(
            text="or",
            original="or",
            position=pos,
            is_stop_word=False,
            ngram_size=1
        ))
        pos += 3
        
        tokens.append(Token(
            text=value_z.lower(),
            original=value_z,
            position=pos,
            is_stop_word=False,
            ngram_size=1
        ))
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract multi-values
        result = extractor.handle_multi_value(tokens)
        
        # Property: All three values should be extracted
        assert len(result) == 3, \
            f"Expected 3 values, got {len(result)}: {result}"
        assert value_x in result, \
            f"Expected '{value_x}' in result, got: {result}"
        assert value_y in result, \
            f"Expected '{value_y}' in result, got: {result}"
        assert value_z in result, \
            f"Expected '{value_z}' in result, got: {result}"
    
    @given(
        value_x=value_text,
        value_y=value_text,
        or_case=st.sampled_from(["or", "Or", "OR", "oR"])
    )
    @settings(max_examples=100)
    def test_case_insensitive_or_detection(
        self, value_x: str, value_y: str, or_case: str
    ):
        """
        Property test: "or" keyword is detected case-insensitively.
        
        For any case variation of "or" (or, Or, OR, oR), the pattern
        SHALL be detected and both values extracted.
        
        **Validates: Property 13**
        **Validates: Requirements 5.3**
        """
        # Ensure X and Y are different values
        assume(value_x.lower() != value_y.lower())
        
        # Create tokens with case-varied "or"
        tokens = [
            Token(
                text=value_x.lower(),
                original=value_x,
                position=0,
                is_stop_word=False,
                ngram_size=1
            ),
            Token(
                text=or_case.lower(),  # normalized text is lowercase
                original=or_case,       # original preserves case
                position=len(value_x) + 1,
                is_stop_word=False,
                ngram_size=1
            ),
            Token(
                text=value_y.lower(),
                original=value_y,
                position=len(value_x) + len(or_case) + 2,
                is_stop_word=False,
                ngram_size=1
            )
        ]
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract multi-values
        result = extractor.handle_multi_value(tokens)
        
        # Property: Both values should be extracted regardless of "or" case
        assert len(result) == 2, \
            f"Expected 2 values with '{or_case}', got {len(result)}: {result}"
        assert value_x in result, \
            f"Expected '{value_x}' in result with '{or_case}', got: {result}"
        assert value_y in result, \
            f"Expected '{value_y}' in result with '{or_case}', got: {result}"
    
    @given(
        value=value_text
    )
    @settings(max_examples=100)
    def test_no_or_pattern_returns_empty(
        self, value: str
    ):
        """
        Property test: Tokens without "or" pattern return empty list.
        
        For any single value without an "or" pattern, handle_multi_value()
        SHALL return an empty list.
        
        **Validates: Property 13**
        **Validates: Requirements 5.3**
        """
        # Create tokens without "or" pattern
        tokens = [
            Token(
                text=value.lower(),
                original=value,
                position=0,
                is_stop_word=False,
                ngram_size=1
            )
        ]
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract multi-values
        result = extractor.handle_multi_value(tokens)
        
        # Property: No "or" pattern means empty result
        assert result == [], \
            f"Expected empty list for single value, got: {result}"
    
    @given(
        num_values=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100)
    def test_multiple_or_patterns_extract_all_values(
        self, num_values: int
    ):
        """
        Property test: Multiple "or" patterns extract all values.
        
        For any number of values (2-5) separated by "or", all values
        SHALL be extracted.
        
        **Validates: Property 13**
        **Validates: Requirements 5.3**
        """
        # Generate unique values
        values = [f"value{i}" for i in range(num_values)]
        
        # Create tokens for "value0 or value1 or value2 ..." pattern
        tokens = []
        pos = 0
        
        for i, value in enumerate(values):
            # Add value token
            tokens.append(Token(
                text=value.lower(),
                original=value,
                position=pos,
                is_stop_word=False,
                ngram_size=1
            ))
            pos += len(value) + 1
            
            # Add "or" token between values (not after the last one)
            if i < len(values) - 1:
                tokens.append(Token(
                    text="or",
                    original="or",
                    position=pos,
                    is_stop_word=False,
                    ngram_size=1
                ))
                pos += 3
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract multi-values
        result = extractor.handle_multi_value(tokens)
        
        # Property: All values should be extracted
        assert len(result) == num_values, \
            f"Expected {num_values} values, got {len(result)}: {result}"
        
        for value in values:
            assert value in result, \
                f"Expected '{value}' in result, got: {result}"
    
    @settings(max_examples=100)
    @given(st.data())
    def test_empty_tokens_returns_empty(self, data):
        """
        Property test: Empty token list returns empty list.
        
        For an empty token list, handle_multi_value() SHALL return
        an empty list.
        
        **Validates: Property 13**
        **Validates: Requirements 5.3**
        """
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract multi-values from empty list
        result = extractor.handle_multi_value([])
        
        # Property: Empty input means empty result
        assert result == [], \
            f"Expected empty list for empty input, got: {result}"
    
    @given(
        value_x=value_text,
        value_y=value_text
    )
    @settings(max_examples=100)
    def test_original_text_preserved_in_extraction(
        self, value_x: str, value_y: str
    ):
        """
        Property test: Original text is preserved in extracted values.
        
        For any "X or Y" pattern, the extracted values SHALL use the
        original text (preserving case) from the tokens.
        
        **Validates: Property 13**
        **Validates: Requirements 5.3**
        """
        # Ensure X and Y are different values
        assume(value_x.lower() != value_y.lower())
        
        # Create tokens with different original vs normalized text
        original_x = value_x
        original_y = value_y
        
        tokens = [
            Token(
                text=value_x.lower(),  # normalized (lowercase)
                original=original_x,    # original (preserves case)
                position=0,
                is_stop_word=False,
                ngram_size=1
            ),
            Token(
                text="or",
                original="or",
                position=len(value_x) + 1,
                is_stop_word=False,
                ngram_size=1
            ),
            Token(
                text=value_y.lower(),  # normalized (lowercase)
                original=original_y,    # original (preserves case)
                position=len(value_x) + 4,
                is_stop_word=False,
                ngram_size=1
            )
        ]
        
        # Create the parameter extractor
        config = EncoderConfig(dimension=1000, seed=42)
        extractor = ParameterExtractor(config)
        
        # Extract multi-values
        result = extractor.handle_multi_value(tokens)
        
        # Property: Original text should be preserved (not normalized)
        assert original_x in result, \
            f"Expected original '{original_x}' in result, got: {result}"
        assert original_y in result, \
            f"Expected original '{original_y}' in result, got: {result}"
