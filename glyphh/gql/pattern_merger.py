"""
Pattern Merger Utility.

Merges custom NL patterns with default patterns, with custom taking precedence.

Requirements: 4.3, 4.4, 4.5, 4.9
"""

import logging
from typing import Any, Dict, List, Optional

from glyphh.gql.patterns import DEFAULT_GQL_PATTERNS, GQLPattern

logger = logging.getLogger(__name__)


class PatternMerger:
    """
    Merges custom NL patterns with default patterns.
    
    Custom patterns override defaults with the same name.
    Logs which patterns are active for debugging.
    
    Requirements: 4.3, 4.4, 4.5, 4.9
    """
    
    @staticmethod
    def merge(
        custom_patterns: Optional[List[GQLPattern]] = None,
        log_source: bool = True,
    ) -> List[GQLPattern]:
        """
        Merge custom patterns with defaults.
        
        Custom patterns take precedence over defaults with the same name.
        
        Args:
            custom_patterns: Optional list of custom patterns
            log_source: Whether to log pattern sources (Requirement 4.9)
            
        Returns:
            Merged list of patterns
        """
        # Start with defaults
        patterns_by_name: Dict[str, GQLPattern] = {
            p.name: p for p in DEFAULT_GQL_PATTERNS
        }
        
        default_count = len(patterns_by_name)
        custom_count = 0
        overridden_count = 0
        
        # Override with custom patterns
        if custom_patterns:
            for pattern in custom_patterns:
                if pattern.name in patterns_by_name:
                    overridden_count += 1
                    if log_source:
                        logger.debug(f"Custom pattern '{pattern.name}' overrides default")
                else:
                    custom_count += 1
                    if log_source:
                        logger.debug(f"Custom pattern '{pattern.name}' added")
                
                patterns_by_name[pattern.name] = pattern
        
        # Log summary (Requirement 4.9)
        if log_source:
            logger.info(
                f"Pattern merge complete: {default_count} defaults, "
                f"{custom_count} custom added, {overridden_count} overridden"
            )
        
        # Sort by priority (higher first)
        result = sorted(
            patterns_by_name.values(),
            key=lambda p: p.priority,
            reverse=True
        )
        
        return result
    
    @staticmethod
    def from_config(
        config: Optional[Dict[str, Any]] = None,
        log_source: bool = True,
    ) -> List[GQLPattern]:
        """
        Load and merge patterns from model config.
        
        Extracts patterns from nl_encoder_config or gql_patterns in config.
        Falls back to defaults when no config provided (Requirement 4.3).
        
        Args:
            config: Model config dict (may contain nl_encoder_config or gql_patterns)
            log_source: Whether to log pattern sources
            
        Returns:
            Merged list of patterns
        """
        if not config:
            if log_source:
                logger.info("No config provided, using default patterns only")
            return PatternMerger.merge(None, log_source)
        
        # Try to extract patterns from config
        custom_patterns: List[GQLPattern] = []
        
        # Check nl_encoder_config.patterns
        nl_config = config.get("nl_encoder_config")
        if nl_config and isinstance(nl_config, dict):
            patterns_data = nl_config.get("patterns", [])
            for p_data in patterns_data:
                try:
                    pattern = GQLPattern.from_dict(p_data)
                    custom_patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to parse pattern from nl_encoder_config: {e}")
        
        # Check gql_patterns (alternative location)
        gql_patterns = config.get("gql_patterns")
        if gql_patterns and isinstance(gql_patterns, list):
            for p_data in gql_patterns:
                try:
                    pattern = GQLPattern.from_dict(p_data)
                    custom_patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to parse pattern from gql_patterns: {e}")
        
        return PatternMerger.merge(custom_patterns, log_source)
    
    @staticmethod
    def get_defaults() -> List[GQLPattern]:
        """Get default patterns without any custom overrides."""
        return list(DEFAULT_GQL_PATTERNS)
    
    @staticmethod
    def get_pattern_names(patterns: List[GQLPattern]) -> List[str]:
        """Get list of pattern names."""
        return [p.name for p in patterns]
    
    @staticmethod
    def find_pattern(
        patterns: List[GQLPattern],
        name: str,
    ) -> Optional[GQLPattern]:
        """Find a pattern by name."""
        for p in patterns:
            if p.name == name:
                return p
        return None
