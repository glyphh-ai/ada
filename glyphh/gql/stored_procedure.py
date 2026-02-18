"""
StoredProcedure - A saved GQL query with NL matching metadata.

StoredProcedure allows developers to define reusable GQL queries that can be:
1. Bundled with a model in notebook.py and deployed to the runtime
2. Created via the runtime API from the GQL Console

Each procedure has associated lexicons (keywords) that enable natural language
matching via the IntentMatcher.

Example:
    >>> from glyphh.gql import StoredProcedure
    >>> 
    >>> procedure = StoredProcedure(
    ...     name="find_reliable_family_car",
    ...     gql_query='FIND SIMILAR TO "reliable family car" WHERE make = "Toyota" LIMIT 10',
    ...     lexicons=["reliable", "family", "car", "toyota"],
    ...     description="Find reliable family cars from Toyota"
    ... )
    >>> 
    >>> # Serialize for storage
    >>> data = procedure.to_dict()
    >>> 
    >>> # Deserialize from storage
    >>> loaded = StoredProcedure.from_dict(data)
"""

import re
from dataclasses import dataclass, field
from typing import List

from glyphh.gql.parser import parse
from glyphh.gql.exceptions import ParseError


# Name must start with a letter and contain only alphanumeric characters and underscores
NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')


@dataclass
class StoredProcedure:
    """
    A stored GQL query with NL matching metadata.
    
    Can be defined in notebook.py and bundled with the model,
    or created via the runtime API.
    
    Attributes:
        name: Unique identifier for the procedure. Must start with a letter
              and contain only alphanumeric characters and underscores.
        gql_query: A valid GQL query string.
        lexicons: List of keywords/phrases for NL matching. At least one
                  non-empty entry is required.
        description: Human-readable description of what the procedure does.
    
    Example:
        >>> procedure = StoredProcedure(
        ...     name="find_reliable_family_car",
        ...     gql_query='FIND SIMILAR TO "reliable family car" WHERE make = "Toyota" LIMIT 10',
        ...     lexicons=["reliable", "family", "car", "toyota"],
        ...     description="Find reliable family cars from Toyota"
        ... )
    
    Raises:
        ValueError: If name format is invalid, gql_query is not valid GQL,
                    or lexicons is empty.
    """
    name: str
    gql_query: str
    lexicons: List[str]
    description: str = ""
    
    def __post_init__(self) -> None:
        """Validate procedure fields after initialization."""
        self._validate_name()
        self._validate_gql_query()
        self._validate_and_clean_lexicons()
    
    def _validate_name(self) -> None:
        """Validate that name follows identifier rules."""
        if not self.name:
            raise ValueError("Procedure name cannot be empty")
        
        if not NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Invalid procedure name '{self.name}'. "
                "Must start with a letter and contain only alphanumeric characters and underscores."
            )
    
    def _validate_gql_query(self) -> None:
        """Validate that gql_query is valid GQL."""
        if not self.gql_query:
            raise ValueError("GQL query cannot be empty")
        
        try:
            parse(self.gql_query)
        except ParseError as e:
            raise ValueError(f"Invalid GQL query: {e}")
    
    def _validate_and_clean_lexicons(self) -> None:
        """Validate and clean lexicons list."""
        if not self.lexicons:
            raise ValueError("At least one lexicon entry is required")
        
        # Clean lexicons: strip whitespace and lowercase
        cleaned = [lex.strip().lower() for lex in self.lexicons if lex.strip()]
        
        if not cleaned:
            raise ValueError("At least one non-empty lexicon entry is required")
        
        self.lexicons = cleaned
    
    def to_dict(self) -> dict:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation of the procedure.
        
        Example:
            >>> procedure = StoredProcedure(
            ...     name="test_proc",
            ...     gql_query='LIST ALL LIMIT 10',
            ...     lexicons=["test"],
            ... )
            >>> procedure.to_dict()
            {'name': 'test_proc', 'gql_query': 'LIST ALL LIMIT 10', 'lexicons': ['test'], 'description': ''}
        """
        return {
            "name": self.name,
            "gql_query": self.gql_query,
            "lexicons": self.lexicons,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StoredProcedure':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing procedure data with keys:
                  name, gql_query, lexicons, and optionally description.
        
        Returns:
            StoredProcedure instance.
        
        Raises:
            KeyError: If required keys are missing.
            ValueError: If validation fails.
        
        Example:
            >>> data = {
            ...     "name": "test_proc",
            ...     "gql_query": "LIST ALL LIMIT 10",
            ...     "lexicons": ["test"],
            ...     "description": "A test procedure"
            ... }
            >>> procedure = StoredProcedure.from_dict(data)
        """
        return cls(
            name=data["name"],
            gql_query=data["gql_query"],
            lexicons=data["lexicons"],
            description=data.get("description", ""),
        )
