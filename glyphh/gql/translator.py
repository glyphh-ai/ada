"""
GQL NL Translator - Translates natural language queries to GQL.

The translator uses HDC-based intent matching to find the best matching
pattern, extracts slot values, and generates GQL queries.

Updated to use PatternMerger for default + custom pattern support (Requirement 4.3).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from glyphh.gql.patterns import (
    GQLPattern,
    SlotDefinition,
    SlotType,
    DEFAULT_GQL_PATTERNS,
)
from glyphh.gql.pattern_merger import PatternMerger
from glyphh.gql.parser import parse
from glyphh.gql.exceptions import SlotValidationError, ParseError
from glyphh.encoder import IntentEncoder, IntentPattern

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """
    Result of translating an NL query to GQL.
    
    Attributes:
        success: Whether translation succeeded
        gql: Generated GQL query (if successful)
        pattern_name: Name of matched pattern
        confidence: Confidence score of the match
        extracted_slots: Dictionary of extracted slot values
        error: Error message (if failed)
    """
    success: bool
    gql: Optional[str] = None
    pattern_name: Optional[str] = None
    confidence: float = 0.0
    extracted_slots: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "gql": self.gql,
            "pattern_name": self.pattern_name,
            "confidence": self.confidence,
            "extracted_slots": self.extracted_slots,
            "error": self.error
        }


class NLTranslator:
    """
    Translates natural language queries to GQL.
    
    The translator uses HDC-based intent matching to find patterns
    and regex-based slot extraction to fill in values.
    
    Example:
        >>> from glyphh.gql import NLTranslator, DEFAULT_GQL_PATTERNS
        >>> 
        >>> translator = NLTranslator(DEFAULT_GQL_PATTERNS)
        >>> result = translator.translate("find similar to red car")
        >>> 
        >>> if result.success:
        ...     print(f"GQL: {result.gql}")
        ...     print(f"Confidence: {result.confidence}")
    """
    
    def __init__(
        self,
        patterns: Optional[List[GQLPattern]] = None,
        config: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.5,
        dimension: int = 10000,
        seed: int = 42
    ):
        """
        Initialize the translator.
        
        Args:
            patterns: List of GQL patterns (if provided, used directly)
            config: Model config dict (used to load patterns via PatternMerger)
            confidence_threshold: Minimum confidence for a match
            dimension: Vector dimension for intent encoder
            seed: Random seed for reproducibility
            
        Note:
            If neither patterns nor config is provided, uses DEFAULT_GQL_PATTERNS.
            If config is provided, uses PatternMerger to merge custom + defaults.
            If patterns is provided directly, uses those patterns as-is.
        """
        self.confidence_threshold = confidence_threshold
        self.dimension = dimension
        self.seed = seed
        
        # Determine patterns to use (Requirement 4.3)
        if patterns is not None:
            # Use provided patterns directly
            self.patterns = patterns
            logger.debug(f"Using {len(patterns)} provided patterns")
        elif config is not None:
            # Use PatternMerger to merge custom + defaults
            self.patterns = PatternMerger.from_config(config, log_source=True)
            logger.info(f"Loaded {len(self.patterns)} patterns from config")
        else:
            # Fall back to defaults only
            self.patterns = PatternMerger.get_defaults()
            logger.info(f"Using {len(self.patterns)} default patterns")
        
        # Build intent encoder from patterns
        self._intent_encoder: Optional[IntentEncoder] = None
        self._pattern_map: Dict[str, GQLPattern] = {}
        self._build_intent_encoder()
    
    def _build_intent_encoder(self) -> None:
        """Build the HDC intent encoder from patterns."""
        from glyphh.core.config import EncoderConfig
        
        # Create a minimal config for the intent encoder
        config = EncoderConfig(
            dimension=self.dimension,
            seed=self.seed
        )
        
        self._intent_encoder = IntentEncoder(config)
        
        for pattern in self.patterns:
            self._pattern_map[pattern.name] = pattern
            
            # Collect all normalized phrases for this pattern
            normalized_phrases = [
                self._normalize_phrase(phrase) 
                for phrase in pattern.phrases
            ]
            
            self._intent_encoder.add_pattern(IntentPattern(
                intent_type=pattern.name,
                example_phrases=normalized_phrases,
                query_template={"operation": pattern.name}
            ))
    
    def _normalize_phrase(self, phrase: str) -> str:
        """Normalize a phrase by replacing slot placeholders with generic text."""
        # Replace {slot_name} with a generic word for intent matching
        return re.sub(r'\{[^}]+\}', 'SLOT', phrase)
    
    def translate(self, query: str) -> TranslationResult:
        """
        Translate a natural language query to GQL.
        
        Args:
            query: Natural language query
        
        Returns:
            TranslationResult with GQL and metadata
        """
        if not self._intent_encoder:
            return TranslationResult(
                success=False,
                error="No patterns configured"
            )
        
        # Normalize query for matching
        normalized_query = self._normalize_phrase(query.lower())
        
        # Match intent using HDC
        match = self._intent_encoder.match(normalized_query)
        
        if not match or match.confidence < self.confidence_threshold:
            return TranslationResult(
                success=False,
                confidence=match.confidence if match else 0.0,
                error=f"No matching pattern found (confidence: {match.confidence if match else 0:.2f})"
            )
        
        # Get the matched pattern
        pattern = self._pattern_map.get(match.intent_type)
        if not pattern:
            return TranslationResult(
                success=False,
                confidence=match.confidence,
                error=f"Pattern '{match.intent_type}' not found"
            )
        
        # Extract slots from the original query
        try:
            extracted_slots = self._extract_slots(query, pattern)
        except SlotValidationError as e:
            return TranslationResult(
                success=False,
                pattern_name=pattern.name,
                confidence=match.confidence,
                error=str(e)
            )
        
        # Fill template with extracted slots
        try:
            gql = self._fill_template(pattern.gql_template, extracted_slots, pattern.slots)
        except Exception as e:
            return TranslationResult(
                success=False,
                pattern_name=pattern.name,
                confidence=match.confidence,
                extracted_slots=extracted_slots,
                error=f"Template fill failed: {e}"
            )
        
        # Validate generated GQL
        try:
            self._validate_gql(gql)
        except ParseError as e:
            return TranslationResult(
                success=False,
                pattern_name=pattern.name,
                confidence=match.confidence,
                extracted_slots=extracted_slots,
                gql=gql,
                error=f"Generated invalid GQL: {e}"
            )
        
        return TranslationResult(
            success=True,
            gql=gql,
            pattern_name=pattern.name,
            confidence=match.confidence,
            extracted_slots=extracted_slots
        )
    
    def _extract_slots(
        self, 
        query: str, 
        pattern: GQLPattern
    ) -> Dict[str, Any]:
        """
        Extract slot values from a query using pattern phrases.
        
        Uses regex matching against pattern phrases to extract values.
        """
        extracted: Dict[str, Any] = {}
        query_lower = query.lower()
        
        # Try each phrase pattern
        for phrase in pattern.phrases:
            regex = self._phrase_to_regex(phrase)
            match = re.search(regex, query_lower, re.IGNORECASE)
            
            if match:
                # Extract named groups
                for slot in pattern.slots:
                    if slot.name in match.groupdict():
                        value = match.group(slot.name)
                        if value:
                            extracted[slot.name] = self._convert_slot_value(
                                value.strip(), slot
                            )
                break
        
        # Apply defaults for missing optional slots
        for slot in pattern.slots:
            if slot.name not in extracted:
                if slot.required:
                    # Try to extract from the whole query as fallback
                    fallback = self._extract_fallback(query, slot, pattern)
                    if fallback is not None:
                        extracted[slot.name] = fallback
                    else:
                        raise SlotValidationError(
                            message="Required slot not found",
                            slot_name=slot.name,
                            slot_type=slot.type.value,
                            pattern_name=pattern.name
                        )
                elif slot.default is not None:
                    extracted[slot.name] = slot.default
        
        return extracted
    
    def _extract_fallback(
        self, 
        query: str, 
        slot: SlotDefinition,
        pattern: GQLPattern
    ) -> Optional[Any]:
        """
        Fallback extraction for slots not matched by phrase patterns.
        
        Uses heuristics based on slot type.
        """
        query_lower = query.lower()
        
        # For STRING type "query" slot, use the main content
        if slot.type == SlotType.STRING and slot.name == "query":
            # Remove common prefixes
            prefixes = ["find", "search", "show", "look for", "similar to", 
                       "like", "get", "list", "what is", "what's"]
            result = query_lower
            for prefix in prefixes:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()
            return result if result else None
        
        # For NUMBER type, find first number
        if slot.type == SlotType.NUMBER:
            match = re.search(r'\b(\d+(?:\.\d+)?)\b', query)
            if match:
                return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
        
        # For DURATION type, find duration pattern
        if slot.type == SlotType.DURATION:
            match = re.search(r'\b(\d+[dhms])\b', query_lower)
            if match:
                return match.group(1)
        
        # For GLYPH_REF, look for quoted strings or identifiers
        if slot.type == SlotType.GLYPH_REF:
            # Try quoted string first
            match = re.search(r'"([^"]+)"', query)
            if match:
                return match.group(1)
            # Try identifier pattern
            match = re.search(r'\b([a-zA-Z][\w-]+)\b', query)
            if match and match.group(1).lower() not in ["find", "search", "compare", "to", "from"]:
                return match.group(1)
        
        return None
    
    def _phrase_to_regex(self, phrase: str) -> str:
        """
        Convert a phrase pattern to a regex.
        
        Converts {slot_name} to named capture groups.
        """
        # Escape regex special characters except {}
        escaped = re.escape(phrase)
        # Unescape the slot placeholders
        escaped = escaped.replace(r'\{', '{').replace(r'\}', '}')
        
        # Replace {slot_name} with named capture groups
        def replace_slot(match):
            slot_name = match.group(1)
            # Capture word characters, spaces, and common punctuation
            return f'(?P<{slot_name}>[\\w\\s\\-\\.]+?)'
        
        regex = re.sub(r'\{(\w+)\}', replace_slot, escaped)
        return regex
    
    def _convert_slot_value(self, value: str, slot: SlotDefinition) -> Any:
        """
        Convert extracted string value to appropriate type.
        """
        if slot.type == SlotType.NUMBER:
            try:
                return float(value) if '.' in value else int(value)
            except ValueError:
                raise SlotValidationError(
                    message="Invalid number format",
                    slot_name=slot.name,
                    slot_type=slot.type.value,
                    actual_value=value
                )
        
        if slot.type == SlotType.DURATION:
            if not re.match(r'^\d+[dhms]$', value):
                raise SlotValidationError(
                    message="Invalid duration format (expected e.g., 30d, 7h)",
                    slot_name=slot.name,
                    slot_type=slot.type.value,
                    actual_value=value
                )
            return value
        
        if slot.type == SlotType.ENUM:
            if slot.enum_values and value.lower() not in [v.lower() for v in slot.enum_values]:
                raise SlotValidationError(
                    message=f"Invalid enum value (expected one of {slot.enum_values})",
                    slot_name=slot.name,
                    slot_type=slot.type.value,
                    actual_value=value
                )
            return value.lower()
        
        # STRING and GLYPH_REF return as-is
        return value
    
    def _fill_template(
        self, 
        template: str, 
        slots: Dict[str, Any],
        slot_defs: List[SlotDefinition]
    ) -> str:
        """
        Fill a GQL template with slot values.
        """
        result = template
        
        for slot_def in slot_defs:
            placeholder = f'{{{slot_def.name}}}'
            if placeholder in result:
                value = slots.get(slot_def.name, slot_def.default)
                if value is not None:
                    result = result.replace(placeholder, str(value))
        
        return result
    
    def _validate_gql(self, gql: str) -> None:
        """
        Validate generated GQL by parsing it.
        
        Raises ParseError if invalid.
        """
        parse(gql)
    
    def add_pattern(self, pattern: GQLPattern) -> None:
        """
        Add a new pattern to the translator.
        
        Args:
            pattern: Pattern to add
        """
        self.patterns.append(pattern)
        self._pattern_map[pattern.name] = pattern
        self._build_intent_encoder()
    
    def remove_pattern(self, name: str) -> bool:
        """
        Remove a pattern by name.
        
        Args:
            name: Pattern name to remove
        
        Returns:
            True if pattern was removed
        """
        self.patterns = [p for p in self.patterns if p.name != name]
        if name in self._pattern_map:
            del self._pattern_map[name]
            self._build_intent_encoder()
            return True
        return False
    
    def get_patterns(self) -> List[GQLPattern]:
        """Get all registered patterns."""
        return self.patterns.copy()
