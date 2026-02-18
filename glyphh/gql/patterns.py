"""
GQL NL Pattern and Slot System.

Defines patterns for translating natural language queries to GQL.
Patterns include slot definitions for extracting values from user queries.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SlotType(Enum):
    """
    Types of slots that can be extracted from NL queries.
    
    Each type has specific validation and conversion rules.
    """
    STRING = "string"       # Free-form text
    NUMBER = "number"       # Integer or float
    DURATION = "duration"   # Time duration (e.g., "30d", "7h")
    GLYPH_REF = "glyph_ref" # Reference to a glyph by ID
    ENUM = "enum"           # One of a predefined set of values


@dataclass
class SlotDefinition:
    """
    Definition of a slot in an NL pattern.
    
    Attributes:
        name: Slot name (used in GQL template substitution)
        type: Slot type for validation
        required: Whether the slot must be present
        default: Default value if slot is not extracted
        enum_values: Valid values for ENUM type slots
        description: Human-readable description
    
    Example:
        >>> slot = SlotDefinition(
        ...     name="query",
        ...     type=SlotType.STRING,
        ...     required=True,
        ...     description="The search query text"
        ... )
    """
    name: str
    type: SlotType = SlotType.STRING
    required: bool = True
    default: Optional[Any] = None
    enum_values: Optional[List[str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "required": self.required,
            "default": self.default,
            "enum_values": self.enum_values,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SlotDefinition':
        return cls(
            name=data["name"],
            type=SlotType(data.get("type", "string")),
            required=data.get("required", True),
            default=data.get("default"),
            enum_values=data.get("enum_values"),
            description=data.get("description")
        )


@dataclass
class GQLPattern:
    """
    A pattern mapping NL phrases to GQL operations.
    
    Patterns define:
    - Example phrases that should match
    - Slots to extract from the query
    - GQL template with slot placeholders
    
    Attributes:
        name: Unique pattern name
        phrases: Example phrases that match this pattern
        slots: Slot definitions for value extraction
        gql_template: GQL template with {slot_name} placeholders
        description: Human-readable description
        priority: Higher priority patterns are matched first
    
    Example:
        >>> pattern = GQLPattern(
        ...     name="find_similar",
        ...     phrases=[
        ...         "find similar to {query}",
        ...         "search for {query}",
        ...         "show me things like {query}"
        ...     ],
        ...     slots=[
        ...         SlotDefinition(name="query", type=SlotType.STRING),
        ...         SlotDefinition(name="limit", type=SlotType.NUMBER, 
        ...                        required=False, default=10)
        ...     ],
        ...     gql_template='FIND SIMILAR TO "{query}" LIMIT {limit}'
        ... )
    """
    name: str
    phrases: List[str]
    slots: List[SlotDefinition] = field(default_factory=list)
    gql_template: str = ""
    description: Optional[str] = None
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phrases": self.phrases,
            "slots": [s.to_dict() for s in self.slots],
            "gql_template": self.gql_template,
            "description": self.description,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GQLPattern':
        slots = [SlotDefinition.from_dict(s) for s in data.get("slots", [])]
        return cls(
            name=data["name"],
            phrases=data.get("phrases", []),
            slots=slots,
            gql_template=data.get("gql_template", ""),
            description=data.get("description"),
            priority=data.get("priority", 0)
        )


# =============================================================================
# Default GQL Patterns
# =============================================================================
# Comprehensive patterns covering all GQL operations (Requirement 4.1, 4.2)

DEFAULT_GQL_PATTERNS: List[GQLPattern] = [
    # =========================================================================
    # Similarity Search
    # =========================================================================
    GQLPattern(
        name="find_similar",
        phrases=[
            "find similar to {query}",
            "search for {query}",
            "find {query}",
            "what's like {query}",
            "similar to {query}",
            "show me things like {query}",
            "look for {query}",
        ],
        slots=[
            SlotDefinition(name="query", type=SlotType.STRING, required=True,
                          description="The search query text"),
            SlotDefinition(name="limit", type=SlotType.NUMBER, required=False,
                          default=10, description="Maximum results"),
        ],
        gql_template='FIND SIMILAR TO "{query}" LIMIT {limit} THRESHOLD 0.5',
        description="Find glyphs similar to a text query",
        priority=10
    ),
    
    GQLPattern(
        name="find_similar_by_role",
        phrases=[
            "find similar {role} to {value}",
            "search by {role} for {value}",
            "find where {role} is like {value}",
            "similar {role} to {value}",
            "find {role} like {value}",
        ],
        slots=[
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role/field to search by"),
            SlotDefinition(name="value", type=SlotType.STRING, required=True,
                          description="Value to search for"),
            SlotDefinition(name="limit", type=SlotType.NUMBER, required=False,
                          default=10, description="Maximum results"),
        ],
        gql_template='FIND SIMILAR TO "{value}" IN {role} LIMIT {limit} THRESHOLD 0.5',
        description="Find glyphs with similar role values",
        priority=12
    ),
    
    # =========================================================================
    # Listing & Counting
    # =========================================================================
    GQLPattern(
        name="list_all",
        phrases=[
            "list all",
            "show all",
            "get all",
            "show everything",
            "list everything",
        ],
        slots=[
            SlotDefinition(name="limit", type=SlotType.NUMBER, required=False,
                          default=100, description="Maximum results"),
        ],
        gql_template='LIST ALL LIMIT {limit}',
        description="List all glyphs",
        priority=5
    ),
    
    GQLPattern(
        name="count_all",
        phrases=[
            "count all",
            "how many",
            "total count",
            "count everything",
        ],
        slots=[],
        gql_template='COUNT ALL',
        description="Count all glyphs",
        priority=5
    ),
    
    # =========================================================================
    # Comparison
    # =========================================================================
    GQLPattern(
        name="compare",
        phrases=[
            "compare {glyph1} to {glyph2}",
            "difference between {glyph1} and {glyph2}",
            "how similar is {glyph1} to {glyph2}",
            "{glyph1} vs {glyph2}",
            "compare {glyph1} with {glyph2}",
        ],
        slots=[
            SlotDefinition(name="glyph1", type=SlotType.GLYPH_REF, required=True,
                          description="First glyph to compare"),
            SlotDefinition(name="glyph2", type=SlotType.GLYPH_REF, required=True,
                          description="Second glyph to compare"),
        ],
        gql_template='COMPARE glyph("{glyph1}") TO glyph("{glyph2}")',
        description="Compare two glyphs",
        priority=10
    ),
    
    GQLPattern(
        name="compare_by_role",
        phrases=[
            "compare {glyph1} to {glyph2} by {role}",
            "difference in {role} between {glyph1} and {glyph2}",
            "compare {role} of {glyph1} and {glyph2}",
        ],
        slots=[
            SlotDefinition(name="glyph1", type=SlotType.GLYPH_REF, required=True),
            SlotDefinition(name="glyph2", type=SlotType.GLYPH_REF, required=True),
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to compare"),
        ],
        gql_template='COMPARE glyph("{glyph1}") TO glyph("{glyph2}") AT ROLE {role}',
        description="Compare two glyphs by a specific role",
        priority=12
    ),
    
    # =========================================================================
    # Temporal & Trends
    # =========================================================================
    GQLPattern(
        name="trend",
        phrases=[
            "show trend of {glyph}",
            "trend for {glyph}",
            "{glyph} trend over {window}",
            "how has {glyph} changed over {window}",
            "track {glyph} for {window}",
            "monitor {glyph} over {window}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True,
                          description="Glyph to track"),
            SlotDefinition(name="window", type=SlotType.DURATION, required=False,
                          default="7d", description="Time window"),
        ],
        gql_template='TREND glyph("{glyph}") OVER {window}',
        description="Track trends over time",
        priority=10
    ),
    
    GQLPattern(
        name="trend_by_role",
        phrases=[
            "trend of {role} for {glyph}",
            "how has {role} changed for {glyph}",
            "{glyph} {role} trend over {window}",
            "track {role} of {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True),
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to track"),
            SlotDefinition(name="window", type=SlotType.DURATION, required=False,
                          default="7d", description="Time window"),
        ],
        gql_template='TREND glyph("{glyph}") OVER {window} AT ROLE {role}',
        description="Track role-specific trends",
        priority=12
    ),
    
    GQLPattern(
        name="predict",
        phrases=[
            "predict {glyph}",
            "forecast {glyph}",
            "what will {glyph} be",
            "future of {glyph}",
            "predict future of {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True,
                          description="Glyph to predict"),
            SlotDefinition(name="window", type=SlotType.DURATION, required=False,
                          default="30d", description="Prediction window"),
        ],
        gql_template='PREDICT glyph("{glyph}") WINDOW {window}',
        description="Predict future state of a glyph",
        priority=10
    ),
    
    GQLPattern(
        name="predict_by_role",
        phrases=[
            "predict {role} for {glyph}",
            "forecast {role} of {glyph}",
            "what will {role} be for {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True),
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to predict"),
            SlotDefinition(name="window", type=SlotType.DURATION, required=False,
                          default="30d", description="Prediction window"),
        ],
        gql_template='PREDICT glyph("{glyph}") ROLE {role} WINDOW {window}',
        description="Predict future state of a specific role",
        priority=12
    ),
    
    GQLPattern(
        name="drift",
        phrases=[
            "detect drift in {glyph}",
            "has {glyph} changed",
            "drift between {glyph1} and {glyph2}",
            "what changed in {glyph}",
            "check drift for {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=False,
                          description="Glyph to check"),
            SlotDefinition(name="glyph1", type=SlotType.GLYPH_REF, required=False,
                          description="First glyph for comparison"),
            SlotDefinition(name="glyph2", type=SlotType.GLYPH_REF, required=False,
                          description="Second glyph for comparison"),
            SlotDefinition(name="threshold", type=SlotType.NUMBER, required=False,
                          default=0.15, description="Drift threshold"),
        ],
        gql_template='DETECT DRIFT FROM glyph("{glyph}") THRESHOLD {threshold}',
        description="Detect semantic drift",
        priority=10
    ),
    
    GQLPattern(
        name="drift_by_role",
        phrases=[
            "drift in {role} for {glyph}",
            "has {role} changed for {glyph}",
            "check {role} drift in {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True),
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to check"),
            SlotDefinition(name="threshold", type=SlotType.NUMBER, required=False,
                          default=0.15, description="Drift threshold"),
        ],
        gql_template='DETECT DRIFT FROM glyph("{glyph}") AT ROLE {role} THRESHOLD {threshold}',
        description="Detect drift in a specific role",
        priority=12
    ),
    
    # =========================================================================
    # Introspection
    # =========================================================================
    GQLPattern(
        name="introspect",
        phrases=[
            "break down {glyph}",
            "decompose {glyph}",
            "analyze {glyph}",
            "what makes up {glyph}",
            "explain {glyph}",
            "introspect {glyph}",
            "show structure of {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True,
                          description="Glyph to introspect"),
        ],
        gql_template='INTROSPECT glyph("{glyph}") DECOMPOSE BY layer',
        description="Introspect a glyph's structure",
        priority=10
    ),
    
    GQLPattern(
        name="introspect_layer",
        phrases=[
            "show layers of {glyph}",
            "layer breakdown of {glyph}",
            "layers in {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True),
        ],
        gql_template='INTROSPECT glyph("{glyph}") DECOMPOSE BY layer',
        description="Show layer breakdown of a glyph",
        priority=12
    ),
    
    GQLPattern(
        name="introspect_segment",
        phrases=[
            "show segments of {glyph}",
            "segment breakdown of {glyph}",
            "segments in {glyph}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True),
        ],
        gql_template='INTROSPECT glyph("{glyph}") DECOMPOSE BY segment',
        description="Show segment breakdown of a glyph",
        priority=12
    ),
    
    GQLPattern(
        name="introspect_role",
        phrases=[
            "show {role} of {glyph}",
            "what is the {role} of {glyph}",
            "get {role} for {glyph}",
            "{glyph} {role}",
        ],
        slots=[
            SlotDefinition(name="glyph", type=SlotType.GLYPH_REF, required=True),
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to show"),
        ],
        gql_template='INTROSPECT glyph("{glyph}") DECOMPOSE BY role AT ROLE {role}',
        description="Show a specific role of a glyph",
        priority=12
    ),
    
    # =========================================================================
    # Aggregation
    # =========================================================================
    GQLPattern(
        name="aggregate_count",
        phrases=[
            "count by {field}",
            "group by {field}",
            "breakdown by {field}",
        ],
        slots=[
            SlotDefinition(name="field", type=SlotType.STRING, required=True,
                          description="Field to group by"),
        ],
        gql_template='AGGREGATE COUNT GROUP BY {field}',
        description="Count glyphs grouped by field",
        priority=8
    ),
    
    GQLPattern(
        name="aggregate_count_role",
        phrases=[
            "count by {role}",
            "group by {role}",
            "how many per {role}",
        ],
        slots=[
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to group by"),
        ],
        gql_template='AGGREGATE COUNT GROUP BY {role}',
        description="Count glyphs grouped by role",
        priority=10
    ),
    
    GQLPattern(
        name="aggregate_avg",
        phrases=[
            "average {field}",
            "mean {field}",
            "avg {field}",
        ],
        slots=[
            SlotDefinition(name="field", type=SlotType.STRING, required=True,
                          description="Field to average"),
        ],
        gql_template='AGGREGATE AVG {field}',
        description="Calculate average of a field",
        priority=8
    ),
    
    GQLPattern(
        name="aggregate_avg_role",
        phrases=[
            "average {role}",
            "mean {role} across all",
            "avg {role}",
        ],
        slots=[
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to average"),
        ],
        gql_template='AGGREGATE AVG {role}',
        description="Calculate average of a role",
        priority=10
    ),
    
    GQLPattern(
        name="aggregate_min_max",
        phrases=[
            "min and max {role}",
            "range of {role}",
            "highest and lowest {role}",
        ],
        slots=[
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to find range"),
        ],
        gql_template='AGGREGATE MIN {role}, MAX {role}',
        description="Find min and max of a role",
        priority=10
    ),
    
    GQLPattern(
        name="aggregate_sum",
        phrases=[
            "sum of {role}",
            "total {role}",
            "sum {role}",
        ],
        slots=[
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to sum"),
        ],
        gql_template='AGGREGATE SUM {role}',
        description="Calculate sum of a role",
        priority=10
    ),
    
    # =========================================================================
    # Filtering
    # =========================================================================
    GQLPattern(
        name="list_where",
        phrases=[
            "show all where {condition}",
            "find where {condition}",
            "list where {condition}",
        ],
        slots=[
            SlotDefinition(name="condition", type=SlotType.STRING, required=True,
                          description="Filter condition"),
            SlotDefinition(name="limit", type=SlotType.NUMBER, required=False,
                          default=100, description="Maximum results"),
        ],
        gql_template='LIST ALL WHERE {condition} LIMIT {limit}',
        description="List glyphs matching a condition",
        priority=8
    ),
    
    GQLPattern(
        name="list_where_role",
        phrases=[
            "show all where {role} is {value}",
            "find where {role} equals {value}",
            "list with {role} {value}",
            "get all {role} {value}",
        ],
        slots=[
            SlotDefinition(name="role", type=SlotType.STRING, required=True,
                          description="Role to filter on"),
            SlotDefinition(name="value", type=SlotType.STRING, required=True,
                          description="Value to match"),
            SlotDefinition(name="limit", type=SlotType.NUMBER, required=False,
                          default=100, description="Maximum results"),
        ],
        gql_template='LIST ALL WHERE {role} = "{value}" LIMIT {limit}',
        description="List glyphs with matching role value",
        priority=10
    ),
]
