"""
Unit tests for the ParameterExtractor class.

Tests the __init__() method, infer_value_type() method, and 
ExtractedParameter/ExtractionResult dataclasses.

Validates: Requirement 5 - Parameter Extraction
"""

import pytest
from glyphh.nl.parameter_extractor import (
    ParameterExtractor,
    ExtractedParameter,
    ExtractionResult,
    VALID_VALUE_TYPES,
)
from glyphh.core.config import EncoderConfig, Layer, Segment, Role


class TestExtractedParameter:
    """Tests for the ExtractedParameter dataclass."""
    
    def test_extracted_parameter_creation(self):
        """Test basic ExtractedParameter creation."""
        param = ExtractedParameter(
            role="make",
            value="Toyota",
            original_text="Toyota",
            confidence=0.95,
            value_type="string"
        )
        
        assert param.role == "make"
        assert param.value == "Toyota"
        assert param.original_text == "Toyota"
        assert param.confidence == 0.95
        assert param.value_type == "string"
    
    def test_extracted_parameter_valid_types(self):
        """Test that all valid value types are accepted."""
        for value_type in VALID_VALUE_TYPES:
            param = ExtractedParameter(
                role="test",
                value="test",
                original_text="test",
                confidence=0.5,
                value_type=value_type
            )
            assert param.value_type == value_type
    
    def test_extracted_parameter_invalid_type_raises(self):
        """Test that invalid value type raises ValueError."""
        with pytest.raises(ValueError, match="value_type must be one of"):
            ExtractedParameter(
                role="test",
                value="test",
                original_text="test",
                confidence=0.5,
                value_type="invalid"
            )
    
    def test_extracted_parameter_empty_role_raises(self):
        """Test that empty role raises ValueError."""
        with pytest.raises(ValueError, match="role must be a non-empty string"):
            ExtractedParameter(
                role="",
                value="test",
                original_text="test",
                confidence=0.5,
                value_type="string"
            )
    
    def test_extracted_parameter_confidence_bounds(self):
        """Test that confidence must be in [0.0, 1.0].
        
        Validates: Requirement 5.4
        """
        # Valid confidence values
        ExtractedParameter(role="test", value="test", original_text="test", confidence=0.0, value_type="string")
        ExtractedParameter(role="test", value="test", original_text="test", confidence=0.5, value_type="string")
        ExtractedParameter(role="test", value="test", original_text="test", confidence=1.0, value_type="string")
        
        # Invalid confidence values
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            ExtractedParameter(role="test", value="test", original_text="test", confidence=-0.1, value_type="string")
        
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            ExtractedParameter(role="test", value="test", original_text="test", confidence=1.1, value_type="string")
    
    def test_is_high_confidence(self):
        """Test is_high_confidence() method."""
        high_param = ExtractedParameter(
            role="make",
            value="Toyota",
            original_text="Toyota",
            confidence=0.85,
            value_type="string"
        )
        low_param = ExtractedParameter(
            role="make",
            value="Toyota",
            original_text="Toyota",
            confidence=0.5,
            value_type="string"
        )
        
        assert high_param.is_high_confidence() is True
        assert high_param.is_high_confidence(threshold=0.9) is False
        assert low_param.is_high_confidence() is False
        assert low_param.is_high_confidence(threshold=0.4) is True
    
    def test_is_string(self):
        """Test is_string() method."""
        string_param = ExtractedParameter(
            role="make", value="Toyota", original_text="Toyota",
            confidence=0.9, value_type="string"
        )
        number_param = ExtractedParameter(
            role="price", value=100, original_text="100",
            confidence=0.9, value_type="number"
        )
        
        assert string_param.is_string() is True
        assert number_param.is_string() is False
    
    def test_is_number(self):
        """Test is_number() method."""
        number_param = ExtractedParameter(
            role="price", value=100, original_text="100",
            confidence=0.9, value_type="number"
        )
        string_param = ExtractedParameter(
            role="make", value="Toyota", original_text="Toyota",
            confidence=0.9, value_type="string"
        )
        
        assert number_param.is_number() is True
        assert string_param.is_number() is False
    
    def test_is_boolean(self):
        """Test is_boolean() method."""
        bool_param = ExtractedParameter(
            role="available", value=True, original_text="true",
            confidence=0.9, value_type="boolean"
        )
        string_param = ExtractedParameter(
            role="make", value="Toyota", original_text="Toyota",
            confidence=0.9, value_type="string"
        )
        
        assert bool_param.is_boolean() is True
        assert string_param.is_boolean() is False


class TestExtractionResult:
    """Tests for the ExtractionResult dataclass."""
    
    def test_extraction_result_creation(self):
        """Test basic ExtractionResult creation."""
        result = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        assert result.parameters == {}
        assert result.multi_value_params == {}
        assert result.unmatched_tokens == []
    
    def test_has_parameters(self):
        """Test has_parameters() method."""
        empty_result = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        param = ExtractedParameter(
            role="make", value="Toyota", original_text="Toyota",
            confidence=0.9, value_type="string"
        )
        with_params = ExtractionResult(
            parameters={"make": param},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        assert empty_result.has_parameters() is False
        assert with_params.has_parameters() is True
    
    def test_get_all_roles(self):
        """Test get_all_roles() method."""
        param1 = ExtractedParameter(
            role="make", value="Toyota", original_text="Toyota",
            confidence=0.9, value_type="string"
        )
        param2 = ExtractedParameter(
            role="category", value="Sedan", original_text="Sedan",
            confidence=0.9, value_type="string"
        )
        
        result = ExtractionResult(
            parameters={"make": param1},
            multi_value_params={"category": [param2]},
            unmatched_tokens=[]
        )
        
        roles = result.get_all_roles()
        assert "make" in roles
        assert "category" in roles
        assert len(roles) == 2


class TestParameterExtractorInit:
    """Tests for the ParameterExtractor.__init__() method.
    
    Validates: Requirement 5
    """
    
    def test_init_with_valid_config(self):
        """Test ParameterExtractor initialization with valid config."""
        config = EncoderConfig(dimension=10000, seed=42)
        extractor = ParameterExtractor(config)
        
        assert extractor.schema_config is config
        assert extractor.schema_config.dimension == 10000
        assert extractor.schema_config.seed == 42
    
    def test_init_with_full_schema(self):
        """Test ParameterExtractor initialization with full schema."""
        config = EncoderConfig(
            dimension=10000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                                Role(name="year")
                            ]
                        )
                    ]
                )
            ]
        )
        extractor = ParameterExtractor(config)
        
        assert len(extractor.schema_config.layers) == 1
        assert extractor.schema_config.layers[0].name == "vehicle"
    
    def test_init_with_invalid_config_type_raises(self):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="schema_config must be an EncoderConfig instance"):
            ParameterExtractor("not a config")
        
        with pytest.raises(TypeError, match="schema_config must be an EncoderConfig instance"):
            ParameterExtractor(None)
        
        with pytest.raises(TypeError, match="schema_config must be an EncoderConfig instance"):
            ParameterExtractor({"dimension": 10000})
    
    def test_repr(self):
        """Test ParameterExtractor __repr__."""
        config = EncoderConfig(dimension=10000, seed=42)
        extractor = ParameterExtractor(config)
        repr_str = repr(extractor)
        
        assert "ParameterExtractor" in repr_str
        assert "dimension=10000" in repr_str


class TestInferValueType:
    """Tests for the ParameterExtractor.infer_value_type() method.
    
    Validates: Requirement 5.4 - THE SDK SHALL preserve value types 
    (string, number, boolean) from the schema
    """
    
    @pytest.fixture
    def extractor(self):
        """Create a ParameterExtractor for testing."""
        config = EncoderConfig(dimension=10000, seed=42)
        return ParameterExtractor(config)
    
    # Boolean detection tests
    def test_infer_boolean_true_lowercase(self, extractor):
        """Test that 'true' is detected as boolean."""
        assert extractor.infer_value_type("true", "available") == "boolean"
    
    def test_infer_boolean_true_uppercase(self, extractor):
        """Test that 'TRUE' is detected as boolean."""
        assert extractor.infer_value_type("TRUE", "available") == "boolean"
    
    def test_infer_boolean_true_mixed_case(self, extractor):
        """Test that 'True' is detected as boolean."""
        assert extractor.infer_value_type("True", "available") == "boolean"
    
    def test_infer_boolean_false_lowercase(self, extractor):
        """Test that 'false' is detected as boolean."""
        assert extractor.infer_value_type("false", "active") == "boolean"
    
    def test_infer_boolean_false_uppercase(self, extractor):
        """Test that 'FALSE' is detected as boolean."""
        assert extractor.infer_value_type("FALSE", "active") == "boolean"
    
    def test_infer_boolean_yes_lowercase(self, extractor):
        """Test that 'yes' is detected as boolean."""
        assert extractor.infer_value_type("yes", "enabled") == "boolean"
    
    def test_infer_boolean_yes_uppercase(self, extractor):
        """Test that 'YES' is detected as boolean."""
        assert extractor.infer_value_type("YES", "enabled") == "boolean"
    
    def test_infer_boolean_no_lowercase(self, extractor):
        """Test that 'no' is detected as boolean."""
        assert extractor.infer_value_type("no", "visible") == "boolean"
    
    def test_infer_boolean_no_uppercase(self, extractor):
        """Test that 'NO' is detected as boolean."""
        assert extractor.infer_value_type("NO", "visible") == "boolean"
    
    # Number detection tests - integers
    def test_infer_number_positive_integer(self, extractor):
        """Test that positive integers are detected as number."""
        assert extractor.infer_value_type("42", "count") == "number"
        assert extractor.infer_value_type("100", "quantity") == "number"
        assert extractor.infer_value_type("0", "index") == "number"
    
    def test_infer_number_negative_integer(self, extractor):
        """Test that negative integers are detected as number."""
        assert extractor.infer_value_type("-100", "offset") == "number"
        assert extractor.infer_value_type("-1", "delta") == "number"
    
    def test_infer_number_large_integer(self, extractor):
        """Test that large integers are detected as number."""
        assert extractor.infer_value_type("1000000", "population") == "number"
        assert extractor.infer_value_type("999999999", "big_number") == "number"
    
    # Number detection tests - floats
    def test_infer_number_positive_float(self, extractor):
        """Test that positive floats are detected as number."""
        assert extractor.infer_value_type("3.14", "pi") == "number"
        assert extractor.infer_value_type("0.5", "ratio") == "number"
        assert extractor.infer_value_type("1.0", "rate") == "number"
    
    def test_infer_number_negative_float(self, extractor):
        """Test that negative floats are detected as number."""
        assert extractor.infer_value_type("-0.5", "discount") == "number"
        assert extractor.infer_value_type("-3.14", "negative_pi") == "number"
    
    def test_infer_number_scientific_notation(self, extractor):
        """Test that scientific notation is detected as number."""
        assert extractor.infer_value_type("1e10", "large") == "number"
        assert extractor.infer_value_type("1.5e-3", "small") == "number"
        assert extractor.infer_value_type("2.5E+6", "scientific") == "number"
    
    # String detection tests
    def test_infer_string_word(self, extractor):
        """Test that regular words are detected as string."""
        assert extractor.infer_value_type("Toyota", "make") == "string"
        assert extractor.infer_value_type("Honda", "make") == "string"
    
    def test_infer_string_multi_word(self, extractor):
        """Test that multi-word values are detected as string."""
        assert extractor.infer_value_type("Brake Pads", "category") == "string"
        assert extractor.infer_value_type("New York", "city") == "string"
    
    def test_infer_string_empty(self, extractor):
        """Test that empty string defaults to string type."""
        assert extractor.infer_value_type("", "empty") == "string"
    
    def test_infer_string_alphanumeric(self, extractor):
        """Test that alphanumeric values are detected as string."""
        assert extractor.infer_value_type("12abc", "code") == "string"
        assert extractor.infer_value_type("abc123", "id") == "string"
    
    def test_infer_string_version_number(self, extractor):
        """Test that version numbers are detected as string."""
        assert extractor.infer_value_type("1.2.3", "version") == "string"
        assert extractor.infer_value_type("v1.0.0", "version") == "string"
    
    def test_infer_string_with_special_chars(self, extractor):
        """Test that values with special characters are detected as string."""
        assert extractor.infer_value_type("hello@world", "email") == "string"
        assert extractor.infer_value_type("test-value", "slug") == "string"
    
    # Edge cases
    def test_infer_string_whitespace_only(self, extractor):
        """Test that whitespace-only values are detected as string."""
        assert extractor.infer_value_type("   ", "whitespace") == "string"
    
    def test_infer_number_with_leading_trailing_spaces(self, extractor):
        """Test that numbers with leading/trailing spaces are detected as number.
        
        Python's int() and float() handle whitespace-padded numbers, so
        ' 42 ' is correctly identified as a number.
        """
        assert extractor.infer_value_type(" 42 ", "padded") == "number"
        assert extractor.infer_value_type("  3.14  ", "padded_float") == "number"
    
    def test_infer_string_partial_boolean(self, extractor):
        """Test that partial boolean keywords are detected as string."""
        assert extractor.infer_value_type("truthy", "flag") == "string"
        assert extractor.infer_value_type("falsey", "flag") == "string"
        assert extractor.infer_value_type("yesno", "flag") == "string"
    
    def test_infer_role_path_does_not_affect_result(self, extractor):
        """Test that role_path parameter doesn't affect type inference.
        
        Currently, role_path is reserved for future schema type hint support.
        """
        # Same value should return same type regardless of role_path
        assert extractor.infer_value_type("42", "count") == "number"
        assert extractor.infer_value_type("42", "price") == "number"
        assert extractor.infer_value_type("42", "vehicle.identity.year") == "number"
        
        assert extractor.infer_value_type("true", "available") == "boolean"
        assert extractor.infer_value_type("true", "active") == "boolean"
        assert extractor.infer_value_type("true", "vehicle.status.available") == "boolean"


class TestExtractParameters:
    """Tests for the ParameterExtractor.extract_parameters() method.
    
    Validates: Requirement 5.1 - THE SDK SHALL extract matched values as query parameters
    Validates: Requirement 5.2 - WHEN a value is matched, THE SDK SHALL associate it with its role
    Validates: Requirement 5.6 - THE SDK SHALL return extracted parameters in a structured format
    """
    
    @pytest.fixture
    def extractor(self):
        """Create a ParameterExtractor for testing."""
        config = EncoderConfig(dimension=10000, seed=42)
        return ParameterExtractor(config)
    
    @pytest.fixture
    def encoder(self):
        """Create an Encoder for generating test vectors."""
        from glyphh.encoder.base import Encoder
        config = EncoderConfig(dimension=10000, seed=42)
        return Encoder(config)
    
    def _create_token(self, text: str, original: str, position: int) -> 'Token':
        """Helper to create a Token for testing."""
        from glyphh.nl.query_tokenizer import Token
        return Token(
            text=text,
            original=original,
            position=position,
            is_stop_word=False,
            ngram_size=1
        )
    
    def _create_schema_vector(self, encoder, key: str, element_type: str, 
                               role_path: str, original_value: str = None) -> 'SchemaVector':
        """Helper to create a SchemaVector for testing."""
        from glyphh.nl.schema_vectorizer import SchemaVector
        vector = encoder.generate_symbol(key)
        return SchemaVector(
            key=key,
            vector=vector,
            element_type=element_type,
            role_path=role_path,
            original_value=original_value
        )
    
    def _create_token_match(self, token, schema_vector, similarity: float, 
                            match_type: str = "exact") -> 'TokenMatch':
        """Helper to create a TokenMatch for testing."""
        from glyphh.nl.schema_matcher import TokenMatch
        return TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=similarity,
            match_type=match_type
        )
    
    def test_extract_single_value_match(self, extractor, encoder):
        """Test extracting a single value match.
        
        Validates: Requirements 5.1, 5.2
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a token and schema vector
        token = self._create_token("toyota", "Toyota", 5)
        schema_vec = self._create_schema_vector(
            encoder, "make=Toyota", "value", 
            "vehicle.identity.make", "Toyota"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        # Create match result
        match_result = MatchResult(
            query="Find Toyota",
            token_matches=[match],
            value_matches=[match]
        )
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Verify extraction
        assert "make" in result.parameters
        assert result.parameters["make"].role == "make"
        assert result.parameters["make"].value == "Toyota"
        assert result.parameters["make"].original_text == "Toyota"
        assert result.parameters["make"].confidence == 0.85
        assert result.parameters["make"].value_type == "string"
    
    def test_extract_multiple_value_matches(self, extractor, encoder):
        """Test extracting multiple value matches for different roles.
        
        Validates: Requirements 5.1, 5.2, 5.6
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create tokens and schema vectors
        token1 = self._create_token("toyota", "Toyota", 5)
        schema_vec1 = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "vehicle.identity.make", "Toyota"
        )
        match1 = self._create_token_match(token1, schema_vec1, 0.85)
        
        token2 = self._create_token("sedan", "Sedan", 12)
        schema_vec2 = self._create_schema_vector(
            encoder, "category=Sedan", "value",
            "vehicle.type.category", "Sedan"
        )
        match2 = self._create_token_match(token2, schema_vec2, 0.90)
        
        # Create match result
        match_result = MatchResult(
            query="Find Toyota Sedan",
            token_matches=[match1, match2],
            value_matches=[match1, match2]
        )
        
        # Extract parameters
        result = extractor.extract_parameters(match_result)
        
        # Verify extraction
        assert len(result.parameters) == 2
        assert "make" in result.parameters
        assert "category" in result.parameters
        assert result.parameters["make"].value == "Toyota"
        assert result.parameters["category"].value == "Sedan"
    
    def test_extract_role_from_nested_path(self, extractor, encoder):
        """Test that role is extracted from the last component of role_path.
        
        Validates: Requirement 5.2
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a token with deeply nested role path
        token = self._create_token("toyota", "Toyota", 0)
        schema_vec = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "vehicle.identity.manufacturer.make", "Toyota"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        match_result = MatchResult(
            query="Toyota",
            token_matches=[match],
            value_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Role should be "make" (last component), not the full path
        assert "make" in result.parameters
        assert result.parameters["make"].role == "make"
    
    def test_extract_role_from_simple_path(self, extractor, encoder):
        """Test that role is extracted correctly from simple (non-nested) path.
        
        Validates: Requirement 5.2
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a token with simple role path (no dots)
        token = self._create_token("toyota", "Toyota", 0)
        schema_vec = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "make", "Toyota"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        match_result = MatchResult(
            query="Toyota",
            token_matches=[match],
            value_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Role should be "make"
        assert "make" in result.parameters
        assert result.parameters["make"].role == "make"
    
    def test_duplicate_roles_keep_highest_confidence(self, extractor, encoder):
        """Test that duplicate roles keep the highest confidence match.
        
        Validates: Requirements 5.1, 5.2
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create two matches for the same role with different confidence
        token1 = self._create_token("toyota", "Toyota", 5)
        schema_vec1 = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "vehicle.identity.make", "Toyota"
        )
        match1 = self._create_token_match(token1, schema_vec1, 0.70)
        
        token2 = self._create_token("honda", "Honda", 12)
        schema_vec2 = self._create_schema_vector(
            encoder, "make=Honda", "value",
            "vehicle.identity.make", "Honda"
        )
        match2 = self._create_token_match(token2, schema_vec2, 0.90)
        
        # Create match result with both matches
        match_result = MatchResult(
            query="Toyota Honda",
            token_matches=[match1, match2],
            value_matches=[match1, match2]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Should keep Honda (higher confidence)
        assert len(result.parameters) == 1
        assert "make" in result.parameters
        assert result.parameters["make"].value == "Honda"
        assert result.parameters["make"].confidence == 0.90
    
    def test_duplicate_roles_first_wins_if_equal_confidence(self, extractor, encoder):
        """Test that first match wins when confidence is equal.
        
        Validates: Requirements 5.1, 5.2
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create two matches for the same role with equal confidence
        token1 = self._create_token("toyota", "Toyota", 5)
        schema_vec1 = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "vehicle.identity.make", "Toyota"
        )
        match1 = self._create_token_match(token1, schema_vec1, 0.85)
        
        token2 = self._create_token("honda", "Honda", 12)
        schema_vec2 = self._create_schema_vector(
            encoder, "make=Honda", "value",
            "vehicle.identity.make", "Honda"
        )
        match2 = self._create_token_match(token2, schema_vec2, 0.85)
        
        match_result = MatchResult(
            query="Toyota Honda",
            token_matches=[match1, match2],
            value_matches=[match1, match2]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Should keep Toyota (first match, equal confidence)
        assert len(result.parameters) == 1
        assert result.parameters["make"].value == "Toyota"
    
    def test_empty_value_matches(self, extractor):
        """Test extraction with no value matches.
        
        Validates: Requirement 5.6
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        match_result = MatchResult(
            query="xyz123",
            token_matches=[],
            value_matches=[]
        )
        
        result = extractor.extract_parameters(match_result)
        
        assert result.parameters == {}
        assert result.multi_value_params == {}
        assert result.unmatched_tokens == []
    
    def test_unmatched_tokens_identified(self, extractor, encoder):
        """Test that unmatched tokens are correctly identified.
        
        Validates: Requirement 5.6
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a value match
        token1 = self._create_token("toyota", "Toyota", 5)
        schema_vec1 = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "vehicle.identity.make", "Toyota"
        )
        value_match = self._create_token_match(token1, schema_vec1, 0.85)
        
        # Create a role match (not a value match)
        token2 = self._create_token("find", "Find", 0)
        schema_vec2 = self._create_schema_vector(
            encoder, "find", "role",
            "intent.find", None
        )
        role_match = self._create_token_match(token2, schema_vec2, 0.60)
        
        match_result = MatchResult(
            query="Find Toyota",
            token_matches=[role_match, value_match],
            value_matches=[value_match]  # Only value_match is a value match
        )
        
        result = extractor.extract_parameters(match_result)
        
        # "find" token should be unmatched (it's a role match, not value match)
        assert len(result.unmatched_tokens) == 1
        assert result.unmatched_tokens[0].text == "find"
    
    def test_value_type_inference_string(self, extractor, encoder):
        """Test that string values are correctly typed.
        
        Validates: Requirement 5.4
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        token = self._create_token("toyota", "Toyota", 0)
        schema_vec = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "make", "Toyota"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        match_result = MatchResult(
            query="Toyota",
            token_matches=[match],
            value_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        assert result.parameters["make"].value_type == "string"
    
    def test_value_type_inference_number(self, extractor, encoder):
        """Test that numeric values are correctly typed.
        
        Validates: Requirement 5.4
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        token = self._create_token("2020", "2020", 0)
        schema_vec = self._create_schema_vector(
            encoder, "year=2020", "value",
            "year", "2020"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        match_result = MatchResult(
            query="2020",
            token_matches=[match],
            value_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        assert result.parameters["year"].value_type == "number"
    
    def test_value_type_inference_boolean(self, extractor, encoder):
        """Test that boolean values are correctly typed.
        
        Validates: Requirement 5.4
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        token = self._create_token("true", "true", 0)
        schema_vec = self._create_schema_vector(
            encoder, "available=true", "value",
            "available", "true"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        match_result = MatchResult(
            query="true",
            token_matches=[match],
            value_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        assert result.parameters["available"].value_type == "boolean"
    
    def test_invalid_match_result_type_raises(self, extractor):
        """Test that invalid match_result type raises TypeError."""
        with pytest.raises(TypeError, match="match_result must be a MatchResult instance"):
            extractor.extract_parameters("not a match result")
        
        with pytest.raises(TypeError, match="match_result must be a MatchResult instance"):
            extractor.extract_parameters(None)
        
        with pytest.raises(TypeError, match="match_result must be a MatchResult instance"):
            extractor.extract_parameters({"query": "test"})
    
    def test_multi_value_params_empty_by_default(self, extractor, encoder):
        """Test that multi_value_params is empty (handled by task 5.5).
        
        Validates: Requirement 5.6
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        token = self._create_token("toyota", "Toyota", 0)
        schema_vec = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "make", "Toyota"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        match_result = MatchResult(
            query="Toyota",
            token_matches=[match],
            value_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # multi_value_params should be empty (handled by handle_multi_value in task 5.5)
        assert result.multi_value_params == {}
    
    def test_compound_match_extraction(self, extractor, encoder):
        """Test extraction from compound matches.
        
        Validates: Requirements 5.1, 5.2
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a compound match (multi-word value)
        token = self._create_token("brake pads", "Brake Pads", 5)
        token.ngram_size = 2  # Indicate it's a 2-gram
        schema_vec = self._create_schema_vector(
            encoder, "category=Brake Pads", "value",
            "vehicle.parts.category", "Brake Pads"
        )
        match = self._create_token_match(token, schema_vec, 0.90, "compound")
        
        match_result = MatchResult(
            query="Find Brake Pads",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        assert "category" in result.parameters
        assert result.parameters["category"].value == "Brake Pads"
        assert result.parameters["category"].original_text == "Brake Pads"
    
    def test_extraction_result_has_parameters(self, extractor, encoder):
        """Test that ExtractionResult.has_parameters() works correctly.
        
        Validates: Requirement 5.6
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # With parameters
        token = self._create_token("toyota", "Toyota", 0)
        schema_vec = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "make", "Toyota"
        )
        match = self._create_token_match(token, schema_vec, 0.85)
        
        match_result = MatchResult(
            query="Toyota",
            token_matches=[match],
            value_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        assert result.has_parameters() is True
        
        # Without parameters
        empty_result = MatchResult(
            query="xyz",
            token_matches=[],
            value_matches=[]
        )
        
        empty_extraction = extractor.extract_parameters(empty_result)
        assert empty_extraction.has_parameters() is False


class TestHandleMultiValue:
    """Tests for the ParameterExtractor.handle_multi_value() method.
    
    Validates: Requirement 5.3 - THE SDK SHALL handle multiple values for 
    the same role (e.g., "Toyota or Honda")
    """
    
    @pytest.fixture
    def extractor(self):
        """Create a ParameterExtractor for testing."""
        config = EncoderConfig(dimension=10000, seed=42)
        return ParameterExtractor(config)
    
    def _create_token(self, text: str, original: str, position: int) -> 'Token':
        """Helper to create a Token for testing."""
        from glyphh.nl.query_tokenizer import Token
        return Token(
            text=text,
            original=original,
            position=position,
            is_stop_word=False,
            ngram_size=1
        )
    
    def test_simple_or_pattern(self, extractor):
        """Test basic "X or Y" pattern detection.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("toyota", "Toyota", 0),
            self._create_token("or", "or", 7),
            self._create_token("honda", "Honda", 10)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        assert len(values) == 2
        assert "Toyota" in values
        assert "Honda" in values
    
    def test_multiple_or_patterns(self, extractor):
        """Test "X or Y or Z" pattern detection.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("red", "red", 0),
            self._create_token("or", "or", 4),
            self._create_token("blue", "blue", 7),
            self._create_token("or", "or", 12),
            self._create_token("green", "green", 15)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        assert len(values) == 3
        assert "red" in values
        assert "blue" in values
        assert "green" in values
    
    def test_no_or_pattern_returns_empty(self, extractor):
        """Test that no "or" pattern returns empty list.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("toyota", "Toyota", 0),
            self._create_token("camry", "Camry", 7)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        assert values == []
    
    def test_empty_tokens_returns_empty(self, extractor):
        """Test that empty token list returns empty list.
        
        Validates: Requirement 5.3
        """
        values = extractor.handle_multi_value([])
        
        assert values == []
    
    def test_case_insensitive_or_detection(self, extractor):
        """Test that "or" is detected case-insensitively.
        
        Validates: Requirement 5.3
        """
        # Test uppercase "OR"
        tokens_upper = [
            self._create_token("toyota", "Toyota", 0),
            self._create_token("or", "OR", 7),
            self._create_token("honda", "Honda", 10)
        ]
        
        values = extractor.handle_multi_value(tokens_upper)
        assert len(values) == 2
        assert "Toyota" in values
        assert "Honda" in values
        
        # Test mixed case "Or"
        tokens_mixed = [
            self._create_token("toyota", "Toyota", 0),
            self._create_token("or", "Or", 7),
            self._create_token("honda", "Honda", 10)
        ]
        
        values = extractor.handle_multi_value(tokens_mixed)
        assert len(values) == 2
    
    def test_preserves_original_case(self, extractor):
        """Test that original token case is preserved in output.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("toyota", "TOYOTA", 0),
            self._create_token("or", "or", 7),
            self._create_token("honda", "Honda", 10)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        # Should use original text, not normalized text
        assert "TOYOTA" in values
        assert "Honda" in values
    
    def test_or_at_start_skipped(self, extractor):
        """Test that "or" at the start is handled gracefully.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("or", "or", 0),
            self._create_token("toyota", "Toyota", 3),
            self._create_token("or", "or", 10),
            self._create_token("honda", "Honda", 13)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        # Should extract Toyota and Honda
        assert len(values) == 2
        assert "Toyota" in values
        assert "Honda" in values
    
    def test_or_at_end_skipped(self, extractor):
        """Test that "or" at the end is handled gracefully.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("toyota", "Toyota", 0),
            self._create_token("or", "or", 7),
            self._create_token("honda", "Honda", 10),
            self._create_token("or", "or", 16)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        # Should extract Toyota and Honda
        assert len(values) == 2
        assert "Toyota" in values
        assert "Honda" in values
    
    def test_consecutive_or_tokens(self, extractor):
        """Test that consecutive "or" tokens are handled gracefully.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("toyota", "Toyota", 0),
            self._create_token("or", "or", 7),
            self._create_token("or", "or", 10),
            self._create_token("honda", "Honda", 13)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        # Should extract Toyota and Honda, skipping consecutive "or"
        assert len(values) == 2
        assert "Toyota" in values
        assert "Honda" in values
    
    def test_single_token_no_or(self, extractor):
        """Test single token without "or" returns empty.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("toyota", "Toyota", 0)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        assert values == []
    
    def test_only_or_token(self, extractor):
        """Test that only "or" token returns empty.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("or", "or", 0)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        assert values == []
    
    def test_four_values_with_or(self, extractor):
        """Test "A or B or C or D" pattern.
        
        Validates: Requirement 5.3
        """
        tokens = [
            self._create_token("a", "A", 0),
            self._create_token("or", "or", 2),
            self._create_token("b", "B", 5),
            self._create_token("or", "or", 7),
            self._create_token("c", "C", 10),
            self._create_token("or", "or", 12),
            self._create_token("d", "D", 15)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        assert len(values) == 4
        assert "A" in values
        assert "B" in values
        assert "C" in values
        assert "D" in values
    
    def test_values_not_duplicated(self, extractor):
        """Test that values are not duplicated in output.
        
        Validates: Requirement 5.3
        """
        # In "A or B or C", B appears after first "or" and before second "or"
        # It should only appear once in the output
        tokens = [
            self._create_token("a", "A", 0),
            self._create_token("or", "or", 2),
            self._create_token("b", "B", 5),
            self._create_token("or", "or", 7),
            self._create_token("c", "C", 10)
        ]
        
        values = extractor.handle_multi_value(tokens)
        
        # B should appear only once
        assert values.count("B") == 1
        assert len(values) == 3
    
    def test_ngram_tokens_supported(self, extractor):
        """Test that n-gram tokens work as values.
        
        Validates: Requirement 5.3
        """
        from glyphh.nl.query_tokenizer import Token
        
        # Create n-gram tokens
        token1 = Token(
            text="brake pads",
            original="Brake Pads",
            position=0,
            is_stop_word=False,
            ngram_size=2
        )
        token_or = Token(
            text="or",
            original="or",
            position=11,
            is_stop_word=False,
            ngram_size=1
        )
        token2 = Token(
            text="oil filters",
            original="Oil Filters",
            position=14,
            is_stop_word=False,
            ngram_size=2
        )
        
        tokens = [token1, token_or, token2]
        
        values = extractor.handle_multi_value(tokens)
        
        assert len(values) == 2
        assert "Brake Pads" in values
        assert "Oil Filters" in values


class TestCompoundValueExtraction:
    """Tests for compound value extraction from multi-word matches.
    
    Validates: Requirement 5.5 - THE SDK SHALL support extracting values from 
    compound matches (multi-word values)
    """
    
    @pytest.fixture
    def extractor(self):
        """Create a ParameterExtractor for testing."""
        config = EncoderConfig(dimension=10000, seed=42)
        return ParameterExtractor(config)
    
    @pytest.fixture
    def encoder(self):
        """Create an Encoder for generating test vectors."""
        from glyphh.encoder.base import Encoder
        config = EncoderConfig(dimension=10000, seed=42)
        return Encoder(config)
    
    def _create_token(self, text: str, original: str, position: int, ngram_size: int = 1) -> 'Token':
        """Helper to create a Token for testing."""
        from glyphh.nl.query_tokenizer import Token
        return Token(
            text=text,
            original=original,
            position=position,
            is_stop_word=False,
            ngram_size=ngram_size
        )
    
    def _create_schema_vector(self, encoder, key: str, element_type: str, 
                               role_path: str, original_value: str = None) -> 'SchemaVector':
        """Helper to create a SchemaVector for testing."""
        from glyphh.nl.schema_vectorizer import SchemaVector
        vector = encoder.generate_symbol(key)
        return SchemaVector(
            key=key,
            vector=vector,
            element_type=element_type,
            role_path=role_path,
            original_value=original_value
        )
    
    def _create_token_match(self, token, schema_vector, similarity: float, 
                            match_type: str = "exact") -> 'TokenMatch':
        """Helper to create a TokenMatch for testing."""
        from glyphh.nl.schema_matcher import TokenMatch
        return TokenMatch(
            token=token,
            schema_vector=schema_vector,
            similarity=similarity,
            match_type=match_type
        )
    
    def test_two_word_compound_value_extraction(self, extractor, encoder):
        """Test extraction of 2-word compound values like 'Brake Pads'.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a 2-word compound match
        token = self._create_token("brake pads", "Brake Pads", 5, ngram_size=2)
        schema_vec = self._create_schema_vector(
            encoder, "category=Brake Pads", "value",
            "vehicle.parts.category", "Brake Pads"
        )
        match = self._create_token_match(token, schema_vec, 0.92, "compound")
        
        match_result = MatchResult(
            query="Find Brake Pads",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Verify compound value is correctly extracted
        assert "category" in result.parameters
        assert result.parameters["category"].value == "Brake Pads"
        assert result.parameters["category"].original_text == "Brake Pads"
        assert result.parameters["category"].confidence == 0.92
        assert result.parameters["category"].value_type == "string"
    
    def test_three_word_compound_value_extraction(self, extractor, encoder):
        """Test extraction of 3-word compound values like 'New York City'.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a 3-word compound match
        token = self._create_token("new york city", "New York City", 10, ngram_size=3)
        schema_vec = self._create_schema_vector(
            encoder, "location=New York City", "value",
            "address.city", "New York City"
        )
        match = self._create_token_match(token, schema_vec, 0.88, "compound")
        
        match_result = MatchResult(
            query="Find stores in New York City",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Verify 3-word compound value is correctly extracted
        assert "city" in result.parameters
        assert result.parameters["city"].value == "New York City"
        assert result.parameters["city"].original_text == "New York City"
    
    def test_compound_value_with_single_word_match(self, extractor, encoder):
        """Test extraction when both compound and single-word matches exist.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a single-word match
        token1 = self._create_token("toyota", "Toyota", 5, ngram_size=1)
        schema_vec1 = self._create_schema_vector(
            encoder, "make=Toyota", "value",
            "vehicle.identity.make", "Toyota"
        )
        match1 = self._create_token_match(token1, schema_vec1, 0.90, "exact")
        
        # Create a compound match
        token2 = self._create_token("brake pads", "Brake Pads", 12, ngram_size=2)
        schema_vec2 = self._create_schema_vector(
            encoder, "category=Brake Pads", "value",
            "vehicle.parts.category", "Brake Pads"
        )
        match2 = self._create_token_match(token2, schema_vec2, 0.88, "compound")
        
        match_result = MatchResult(
            query="Find Toyota Brake Pads",
            token_matches=[match1, match2],
            value_matches=[match1, match2],
            compound_matches=[match2]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Verify both single-word and compound values are extracted
        assert len(result.parameters) == 2
        assert "make" in result.parameters
        assert "category" in result.parameters
        assert result.parameters["make"].value == "Toyota"
        assert result.parameters["category"].value == "Brake Pads"
    
    def test_compound_value_preserves_original_case(self, extractor, encoder):
        """Test that compound value extraction preserves original case.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a compound match with mixed case original value
        token = self._create_token("los angeles", "Los Angeles", 8, ngram_size=2)
        schema_vec = self._create_schema_vector(
            encoder, "city=Los Angeles", "value",
            "location.city", "Los Angeles"
        )
        match = self._create_token_match(token, schema_vec, 0.91, "compound")
        
        match_result = MatchResult(
            query="Find in Los Angeles",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Verify original case is preserved
        assert result.parameters["city"].value == "Los Angeles"
        assert result.parameters["city"].original_text == "Los Angeles"
    
    def test_compound_value_with_special_characters(self, extractor, encoder):
        """Test compound value extraction with special characters.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a compound match with hyphenated value
        token = self._create_token("anti lock brakes", "Anti-Lock Brakes", 5, ngram_size=3)
        schema_vec = self._create_schema_vector(
            encoder, "feature=Anti-Lock Brakes", "value",
            "vehicle.features.safety", "Anti-Lock Brakes"
        )
        match = self._create_token_match(token, schema_vec, 0.85, "compound")
        
        match_result = MatchResult(
            query="Find Anti-Lock Brakes",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Verify compound value with special characters is extracted
        assert "safety" in result.parameters
        assert result.parameters["safety"].value == "Anti-Lock Brakes"
    
    def test_multiple_compound_values_different_roles(self, extractor, encoder):
        """Test extraction of multiple compound values for different roles.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create first compound match
        token1 = self._create_token("brake pads", "Brake Pads", 5, ngram_size=2)
        schema_vec1 = self._create_schema_vector(
            encoder, "category=Brake Pads", "value",
            "parts.category", "Brake Pads"
        )
        match1 = self._create_token_match(token1, schema_vec1, 0.90, "compound")
        
        # Create second compound match
        token2 = self._create_token("front wheel", "Front Wheel", 17, ngram_size=2)
        schema_vec2 = self._create_schema_vector(
            encoder, "position=Front Wheel", "value",
            "parts.position", "Front Wheel"
        )
        match2 = self._create_token_match(token2, schema_vec2, 0.88, "compound")
        
        match_result = MatchResult(
            query="Find Brake Pads Front Wheel",
            token_matches=[match1, match2],
            value_matches=[match1, match2],
            compound_matches=[match1, match2]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Verify both compound values are extracted
        assert len(result.parameters) == 2
        assert "category" in result.parameters
        assert "position" in result.parameters
        assert result.parameters["category"].value == "Brake Pads"
        assert result.parameters["position"].value == "Front Wheel"
    
    def test_compound_value_higher_confidence_wins(self, extractor, encoder):
        """Test that higher confidence compound match wins over lower confidence.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create two compound matches for the same role with different confidence
        token1 = self._create_token("brake pads", "Brake Pads", 5, ngram_size=2)
        schema_vec1 = self._create_schema_vector(
            encoder, "category=Brake Pads", "value",
            "parts.category", "Brake Pads"
        )
        match1 = self._create_token_match(token1, schema_vec1, 0.75, "compound")
        
        token2 = self._create_token("oil filters", "Oil Filters", 17, ngram_size=2)
        schema_vec2 = self._create_schema_vector(
            encoder, "category=Oil Filters", "value",
            "parts.category", "Oil Filters"
        )
        match2 = self._create_token_match(token2, schema_vec2, 0.92, "compound")
        
        match_result = MatchResult(
            query="Find Brake Pads Oil Filters",
            token_matches=[match1, match2],
            value_matches=[match1, match2],
            compound_matches=[match1, match2]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Higher confidence match should win
        assert len(result.parameters) == 1
        assert result.parameters["category"].value == "Oil Filters"
        assert result.parameters["category"].confidence == 0.92
    
    def test_compound_value_from_nested_role_path(self, extractor, encoder):
        """Test compound value extraction with deeply nested role path.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a compound match with deeply nested role path
        token = self._create_token("power steering", "Power Steering", 5, ngram_size=2)
        schema_vec = self._create_schema_vector(
            encoder, "feature=Power Steering", "value",
            "vehicle.systems.steering.type", "Power Steering"
        )
        match = self._create_token_match(token, schema_vec, 0.89, "compound")
        
        match_result = MatchResult(
            query="Find Power Steering",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Role should be extracted from last component of path
        assert "type" in result.parameters
        assert result.parameters["type"].value == "Power Steering"
    
    def test_compound_value_type_inference(self, extractor, encoder):
        """Test that compound values are correctly typed as strings.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a compound match
        token = self._create_token("brake pads", "Brake Pads", 5, ngram_size=2)
        schema_vec = self._create_schema_vector(
            encoder, "category=Brake Pads", "value",
            "parts.category", "Brake Pads"
        )
        match = self._create_token_match(token, schema_vec, 0.90, "compound")
        
        match_result = MatchResult(
            query="Find Brake Pads",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[match]
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Compound values should be typed as strings
        assert result.parameters["category"].value_type == "string"
    
    def test_empty_compound_matches_list(self, extractor, encoder):
        """Test extraction when compound_matches list is empty but value_matches has compound.
        
        Validates: Requirement 5.5
        """
        from glyphh.nl.schema_matcher import MatchResult
        
        # Create a compound match that's in value_matches but not compound_matches
        # This tests that extraction works based on value_matches, not compound_matches
        token = self._create_token("brake pads", "Brake Pads", 5, ngram_size=2)
        schema_vec = self._create_schema_vector(
            encoder, "category=Brake Pads", "value",
            "parts.category", "Brake Pads"
        )
        match = self._create_token_match(token, schema_vec, 0.90, "compound")
        
        match_result = MatchResult(
            query="Find Brake Pads",
            token_matches=[match],
            value_matches=[match],
            compound_matches=[]  # Empty compound_matches
        )
        
        result = extractor.extract_parameters(match_result)
        
        # Should still extract from value_matches
        assert "category" in result.parameters
        assert result.parameters["category"].value == "Brake Pads"
