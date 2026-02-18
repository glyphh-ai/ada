"""
GQL Execution Planner - Transforms AST nodes into execution plans.

The planner validates references, resolves hierarchy levels, and
creates optimized execution plans from parsed AST nodes.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

from glyphh.gql.ast import (
    ASTNode,
    ASTVisitor,
    HierarchyLevel,
    GlyphRef,
    Condition,
    ComparisonCondition,
    LogicalCondition,
    SimilaritySearchNode,
    ListNode,
    CountNode,
    CompareNode,
    PredictNode,
    DriftDetectionNode,
    IntrospectNode,
    TrendNode,
    AggregateNode,
)
from glyphh.gql.plans import (
    ExecutionPlan,
    SimilaritySearchPlan,
    ListPlan,
    CountPlan,
    ComparePlan,
    PredictPlan,
    DriftPlan,
    IntrospectPlan,
    TrendPlan,
    AggregatePlan,
)
from glyphh.gql.exceptions import PlanningError, HierarchyError
from glyphh.gql.printer import pretty_print
from glyphh.gql.storage import GlyphStorageProtocol, InMemoryGlyphStorage


class ExecutionContext:
    """
    Context for query execution with storage abstraction.
    
    Provides access to the model, encoder, similarity calculator,
    and glyph storage. The storage is abstracted via GlyphStorageProtocol,
    allowing different backends (in-memory, database, etc.).
    
    Example:
        >>> # Using glyphs dict (backward compatible)
        >>> context = ExecutionContext(glyphs={"g1": glyph1})
        >>> 
        >>> # Using custom storage
        >>> storage = DatabaseGlyphStorage(org_id, model_id, glyphs, embeddings)
        >>> context = ExecutionContext(storage=storage)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        glyphs: Optional[Dict[str, Any]] = None,
        storage: Optional[GlyphStorageProtocol] = None,
        encoder: Optional[Any] = None,
        similarity_calculator: Optional[Any] = None,
        predictor: Optional[Any] = None
    ):
        """
        Initialize execution context.
        
        Args:
            model: GlyphhModel instance
            glyphs: Dictionary of glyph_id -> Glyph (backward compatible)
            storage: GlyphStorageProtocol implementation (preferred)
            encoder: Encoder instance for encoding text queries
            similarity_calculator: SimilarityCalculator instance
            predictor: BeamSearchPredictor instance for temporal predictions
            
        Raises:
            ValueError: If both glyphs and storage are provided
            
        Note:
            Use either `glyphs` OR `storage`, not both. If `glyphs` is
            provided, it will be wrapped in InMemoryGlyphStorage for
            backward compatibility.
        """
        # Validate mutually exclusive parameters
        if glyphs is not None and storage is not None:
            raise ValueError(
                "Cannot specify both 'glyphs' and 'storage'. "
                "Use 'storage' for custom implementations or 'glyphs' for backward compatibility."
            )
        
        self.model = model
        self.encoder = encoder
        self.predictor = predictor
        self._similarity_calculator = similarity_calculator
        self._history: Dict[str, List[Any]] = {}  # glyph_id -> historical versions
        
        # Create storage from glyphs dict for backward compatibility
        if storage is not None:
            self._storage = storage
        elif glyphs is not None:
            self._storage = InMemoryGlyphStorage(
                glyphs=glyphs,
                similarity_calculator=similarity_calculator
            )
        else:
            self._storage = InMemoryGlyphStorage(
                glyphs={},
                similarity_calculator=similarity_calculator
            )
        
        # Keep reference to similarity_calculator for legacy compatibility
        self.similarity_calculator = similarity_calculator
    
    # =========================================================================
    # Storage delegation methods
    # =========================================================================
    
    def has_glyph(self, glyph_id: str) -> bool:
        """Check if a glyph exists."""
        return self._storage.has_glyph(glyph_id)
    
    def get_glyph(self, glyph_id: str) -> Any:
        """Get a glyph by ID."""
        try:
            return self._storage.get_glyph(glyph_id)
        except KeyError:
            raise PlanningError(
                message="Glyph not found",
                reference=glyph_id,
                available=list(self._storage.list_glyphs().keys())[:10]
            )
    
    def list_glyphs(self) -> Dict[str, Any]:
        """List all glyphs."""
        return self._storage.list_glyphs()
    
    def get_embedding(self, glyph_id: str) -> Optional[Any]:
        """Get the primary embedding for a glyph."""
        return self._storage.get_embedding(glyph_id)
    
    def get_embedding_for_scope(
        self,
        glyph_id: str,
        layer: Optional[str] = None,
        segment: Optional[str] = None
    ) -> Optional[Any]:
        """Get embedding for a specific scope (layer/segment)."""
        return self._storage.get_embedding_for_scope(glyph_id, layer, segment)
    
    def compute_similarity(self, v1: Any, v2: Any) -> float:
        """Compute similarity between two vectors."""
        return self._storage.compute_similarity(v1, v2)
    
    def get_glyph_attribute(self, glyph: Any, field: str) -> Any:
        """
        Get an attribute value from a glyph.
        
        Note: This method accepts a glyph object for backward compatibility
        with existing plan code. For new code, prefer using glyph_id.
        """
        # If glyph is a string (glyph_id), delegate to storage
        if isinstance(glyph, str):
            return self._storage.get_glyph_attribute(glyph, field)
        
        # For glyph objects, try to get the ID and delegate
        glyph_id = getattr(glyph, 'identifier', None) or str(getattr(glyph, 'id', ''))
        if glyph_id and self._storage.has_glyph(glyph_id):
            return self._storage.get_glyph_attribute(glyph_id, field)
        
        # Fallback: direct attribute access on glyph object
        if hasattr(glyph, field):
            return getattr(glyph, field)
        if hasattr(glyph, 'attributes') and isinstance(glyph.attributes, dict):
            return glyph.attributes.get(field)
        if hasattr(glyph, 'metadata') and isinstance(glyph.metadata, dict):
            return glyph.metadata.get(field)
        return None
    
    # =========================================================================
    # Legacy methods for backward compatibility
    # =========================================================================
    
    def get_layer_vector(self, glyph: Any, layer_name: str) -> Optional[Any]:
        """
        Get a layer's cortex vector from a glyph.
        
        Legacy method - delegates to get_embedding_for_scope().
        """
        glyph_id = getattr(glyph, 'identifier', None) or str(getattr(glyph, 'id', ''))
        if glyph_id:
            return self.get_embedding_for_scope(glyph_id, layer=layer_name)
        
        # Fallback for direct glyph object access
        if hasattr(glyph, 'layers'):
            for layer in glyph.layers:
                if hasattr(layer, 'name') and layer.name == layer_name:
                    return layer.cortex if hasattr(layer, 'cortex') else None
        return None
    
    def get_segment_vector(
        self, glyph: Any, layer_name: str, segment_name: Optional[str]
    ) -> Optional[Any]:
        """
        Get a segment's cortex vector from a glyph.
        
        Legacy method - delegates to get_embedding_for_scope().
        """
        glyph_id = getattr(glyph, 'identifier', None) or str(getattr(glyph, 'id', ''))
        if glyph_id:
            return self.get_embedding_for_scope(glyph_id, layer=layer_name, segment=segment_name)
        
        # Fallback for direct glyph object access
        if hasattr(glyph, 'layers'):
            for layer in glyph.layers:
                if hasattr(layer, 'name') and layer.name == layer_name:
                    if segment_name and hasattr(layer, 'segments'):
                        for segment in layer.segments:
                            if hasattr(segment, 'name') and segment.name == segment_name:
                                return segment.cortex if hasattr(segment, 'cortex') else None
                    return layer.cortex if hasattr(layer, 'cortex') else None
        return None
    
    def get_predictor(self) -> Optional[Any]:
        """Get the temporal predictor."""
        return self.predictor
    
    def get_glyph_history(self, glyph_id: str, window_seconds: int) -> List[Any]:
        """Get historical versions of a glyph."""
        return self._history.get(glyph_id, [])
    
    def decompose_glyph(self, glyph: Any, level: str) -> Dict[str, Any]:
        """Decompose a glyph's structure at the specified level."""
        result = {"glyph_id": getattr(glyph, 'identifier', 'unknown')}
        
        if level == "layer" and hasattr(glyph, 'layers'):
            result["layers"] = [
                {"name": getattr(l, 'name', 'unknown')}
                for l in glyph.layers
            ]
        elif level == "segment" and hasattr(glyph, 'layers'):
            result["layers"] = []
            for layer in glyph.layers:
                layer_info = {"name": getattr(layer, 'name', 'unknown'), "segments": []}
                if hasattr(layer, 'segments'):
                    for seg in layer.segments:
                        layer_info["segments"].append({"name": getattr(seg, 'name', 'unknown')})
                result["layers"].append(layer_info)
        elif level == "role" and hasattr(glyph, 'layers'):
            result["layers"] = []
            for layer in glyph.layers:
                layer_info = {"name": getattr(layer, 'name', 'unknown'), "segments": []}
                if hasattr(layer, 'segments'):
                    for seg in layer.segments:
                        seg_info = {"name": getattr(seg, 'name', 'unknown'), "roles": []}
                        if hasattr(seg, 'roles'):
                            for role in seg.roles:
                                seg_info["roles"].append({"name": getattr(role, 'name', 'unknown')})
                        layer_info["segments"].append(seg_info)
                result["layers"].append(layer_info)
        
        return result
    
    def encode_text(self, text: str) -> Any:
        """Encode text into a vector."""
        if self.encoder:
            from glyphh import Concept
            concept = Concept(name=text, attributes={"text": text})
            return self.encoder.encode(concept).cortex
        return None


class ExecutionPlanner(ASTVisitor):
    """
    Transforms AST nodes into execution plans.
    
    The planner validates references, resolves hierarchy levels,
    and creates optimized execution plans.
    
    Example:
        >>> from glyphh.gql import parse, ExecutionPlanner, ExecutionContext
        >>> 
        >>> context = ExecutionContext(glyphs={"g1": glyph1, "g2": glyph2})
        >>> planner = ExecutionPlanner(context)
        >>> 
        >>> ast = parse('COMPARE glyph("g1") TO glyph("g2")')
        >>> plan = planner.plan(ast)
    """
    
    # Mapping from hierarchy level to edge type
    LEVEL_TO_EDGE_TYPE = {
        "cortex": "neural_cortex",
        "layer": "neural_layer",
        "segment": "neural_segment",
        "role": "neural_role"
    }
    
    # Duration parsing pattern
    DURATION_PATTERN = re.compile(r'^(\d+)([dhms])$')
    
    def __init__(self, context: ExecutionContext):
        """
        Initialize the planner.
        
        Args:
            context: Execution context for validation
        """
        self.context = context
    
    def plan(self, node: ASTNode) -> ExecutionPlan:
        """
        Create an execution plan from an AST node.
        
        Args:
            node: The AST node to plan
        
        Returns:
            ExecutionPlan ready for execution
        
        Raises:
            PlanningError: If references are invalid
            HierarchyError: If hierarchy paths are invalid
        """
        return node.accept(self)
    
    def visit_similarity_search(self, node: SimilaritySearchNode) -> SimilaritySearchPlan:
        """Plan a similarity search with optional WHERE predicate."""
        target_vector = None
        target_glyph_id = None
        
        if isinstance(node.target, str):
            # Encode text query
            target_vector = self.context.encode_text(node.target)
        else:
            # Validate glyph reference
            target_glyph_id = node.target.identifier
            if not self.context.has_glyph(target_glyph_id):
                raise PlanningError(
                    message="Target glyph not found",
                    reference=target_glyph_id,
                    available=list(self.context.list_glyphs().keys())[:10]
                )
        
        scope_layer, scope_segment = self._parse_scope(node.scope)
        
        # Build predicate from WHERE clause
        predicate = None
        predicate_desc = None
        if node.where:
            predicate = self._build_predicate(node.where)
            # Extract WHERE clause from pretty-printed output
            printed = pretty_print(node)
            if "WHERE" in printed:
                # Extract text between WHERE and LIMIT/THRESHOLD/IN/end
                where_part = printed.split("WHERE")[1]
                # Remove trailing clauses (LIMIT, THRESHOLD, IN)
                for keyword in [" LIMIT ", " THRESHOLD ", " IN "]:
                    if keyword in where_part:
                        where_part = where_part.split(keyword)[0]
                predicate_desc = where_part.strip()
        
        return SimilaritySearchPlan(
            target_vector=target_vector,
            target_glyph_id=target_glyph_id,
            scope_layer=scope_layer,
            scope_segment=scope_segment,
            limit=node.limit,
            threshold=node.threshold,
            predicate=predicate,
            predicate_desc=predicate_desc,
        )
    
    def visit_list(self, node: ListNode) -> ListPlan:
        """Plan a list query."""
        predicate = None
        predicate_desc = None
        
        if node.where:
            predicate = self._build_predicate(node.where)
            predicate_desc = pretty_print(node).split("WHERE")[1].strip() if "WHERE" in pretty_print(node) else None
        
        return ListPlan(
            predicate=predicate,
            predicate_desc=predicate_desc,
            limit=node.limit
        )
    
    def visit_count(self, node: CountNode) -> CountPlan:
        """Plan a count query."""
        predicate = None
        predicate_desc = None
        
        if node.where:
            predicate = self._build_predicate(node.where)
            predicate_desc = pretty_print(node).split("WHERE")[1].strip() if "WHERE" in pretty_print(node) else None
        
        return CountPlan(
            predicate=predicate,
            predicate_desc=predicate_desc
        )
    
    def visit_compare(self, node: CompareNode) -> ComparePlan:
        """Plan a comparison."""
        # Validate both glyphs exist
        if not self.context.has_glyph(node.glyph1.identifier):
            raise PlanningError(
                message="First glyph not found",
                reference=node.glyph1.identifier,
                available=list(self.context.list_glyphs().keys())[:10]
            )
        if not self.context.has_glyph(node.glyph2.identifier):
            raise PlanningError(
                message="Second glyph not found",
                reference=node.glyph2.identifier,
                available=list(self.context.list_glyphs().keys())[:10]
            )
        
        edge_type = self._level_to_edge_type(node.level)
        
        return ComparePlan(
            glyph1_id=node.glyph1.identifier,
            glyph2_id=node.glyph2.identifier,
            edge_type=edge_type,
            level_path=node.level.path
        )
    
    def visit_predict(self, node: PredictNode) -> PredictPlan:
        """Plan a prediction."""
        if not self.context.has_glyph(node.glyph.identifier):
            raise PlanningError(
                message="Glyph not found",
                reference=node.glyph.identifier,
                available=list(self.context.list_glyphs().keys())[:10]
            )
        
        window_seconds = self._parse_duration(node.window) if node.window else None
        
        return PredictPlan(
            glyph_id=node.glyph.identifier,
            role_path=node.role,
            window_seconds=window_seconds
        )
    
    def visit_drift_detection(self, node: DriftDetectionNode) -> DriftPlan:
        """Plan drift detection."""
        if not self.context.has_glyph(node.from_glyph.identifier):
            raise PlanningError(
                message="From glyph not found",
                reference=node.from_glyph.identifier,
                available=list(self.context.list_glyphs().keys())[:10]
            )
        if not self.context.has_glyph(node.to_glyph.identifier):
            raise PlanningError(
                message="To glyph not found",
                reference=node.to_glyph.identifier,
                available=list(self.context.list_glyphs().keys())[:10]
            )
        
        edge_type = self._level_to_edge_type(node.level)
        
        return DriftPlan(
            from_glyph_id=node.from_glyph.identifier,
            to_glyph_id=node.to_glyph.identifier,
            edge_type=edge_type,
            level_path=node.level.path,
            threshold=node.threshold
        )
    
    def visit_introspect(self, node: IntrospectNode) -> IntrospectPlan:
        """Plan introspection."""
        if not self.context.has_glyph(node.glyph.identifier):
            raise PlanningError(
                message="Glyph not found",
                reference=node.glyph.identifier,
                available=list(self.context.list_glyphs().keys())[:10]
            )
        
        return IntrospectPlan(
            glyph_id=node.glyph.identifier,
            decompose_level=node.decompose_by
        )
    
    def visit_trend(self, node: TrendNode) -> TrendPlan:
        """Plan trend analysis."""
        if not self.context.has_glyph(node.glyph.identifier):
            raise PlanningError(
                message="Glyph not found",
                reference=node.glyph.identifier,
                available=list(self.context.list_glyphs().keys())[:10]
            )
        
        window_seconds = self._parse_duration(node.window)
        edge_type = self._level_to_edge_type(node.level)
        
        # Parse alert threshold from condition string
        alert_threshold = None
        if node.alert_condition:
            # Parse "drift > 0.2" format
            match = re.match(r'drift\s*[><=]+\s*([\d.]+)', node.alert_condition)
            if match:
                alert_threshold = float(match.group(1))
        
        return TrendPlan(
            glyph_id=node.glyph.identifier,
            window_seconds=window_seconds,
            edge_type=edge_type,
            level_path=node.level.path,
            alert_threshold=alert_threshold
        )
    
    def visit_aggregate(self, node: AggregateNode) -> AggregatePlan:
        """Plan aggregation."""
        predicate = None
        predicate_desc = None
        
        if node.where:
            predicate = self._build_predicate(node.where)
            predicate_desc = str(node.where.to_dict())
        
        return AggregatePlan(
            function=node.function,
            field=node.field,
            group_by=node.group_by,
            predicate=predicate,
            predicate_desc=predicate_desc
        )
    
    def visit_comparison(self, node: ComparisonCondition) -> Any:
        """Not used directly - conditions are built via _build_predicate."""
        pass
    
    def visit_logical(self, node: LogicalCondition) -> Any:
        """Not used directly - conditions are built via _build_predicate."""
        pass
    
    def _level_to_edge_type(self, level: HierarchyLevel) -> str:
        """Map hierarchy level to SDK edge type."""
        return self.LEVEL_TO_EDGE_TYPE.get(level.level, "neural_cortex")
    
    def _parse_scope(self, scope: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Parse scope string into layer and segment."""
        if not scope:
            return None, None
        parts = scope.split(".")
        layer = parts[0] if len(parts) > 0 else None
        segment = parts[1] if len(parts) > 1 else None
        return layer, segment
    
    def _parse_duration(self, duration: str) -> int:
        """Parse duration string to seconds."""
        match = self.DURATION_PATTERN.match(duration)
        if not match:
            return 0
        
        value = int(match.group(1))
        unit = match.group(2)
        
        multipliers = {
            'd': 86400,  # days
            'h': 3600,   # hours
            'm': 60,     # minutes
            's': 1       # seconds
        }
        
        return value * multipliers.get(unit, 1)
    
    def _build_predicate(self, condition: Condition) -> Callable[[str, 'ExecutionContext'], bool]:
        """
        Build a predicate function from a condition.
        
        The predicate takes (glyph_id, context) and uses context.get_glyph_attribute()
        for attribute access, supporting the storage abstraction (Requirement 3.4).
        """
        if isinstance(condition, ComparisonCondition):
            return self._build_comparison_predicate(condition)
        elif isinstance(condition, LogicalCondition):
            return self._build_logical_predicate(condition)
        return lambda glyph_id, ctx: True
    
    def _build_comparison_predicate(self, cond: ComparisonCondition) -> Callable[[str, 'ExecutionContext'], bool]:
        """
        Build a comparison predicate using storage abstraction.
        
        Uses context.get_glyph_attribute(glyph_id, field) instead of
        accessing glyph object properties directly (Requirement 3.4).
        """
        field = cond.field
        op = cond.operator
        value = cond.value
        
        def predicate(glyph_id: str, ctx: 'ExecutionContext') -> bool:
            # Use storage abstraction for attribute access
            glyph_value = ctx.get_glyph_attribute(glyph_id, field)
            if glyph_value is None:
                return False
            
            try:
                if op == "=":
                    return glyph_value == value
                elif op == "!=":
                    return glyph_value != value
                elif op == "<":
                    return glyph_value < value
                elif op == ">":
                    return glyph_value > value
                elif op == "<=":
                    return glyph_value <= value
                elif op == ">=":
                    return glyph_value >= value
            except TypeError:
                return False
            
            return False
        
        return predicate
    
    def _build_logical_predicate(self, cond: LogicalCondition) -> Callable[[str, 'ExecutionContext'], bool]:
        """Build a logical predicate using storage abstraction."""
        predicates = [self._build_predicate(op) for op in cond.operands]
        
        if cond.operator == "AND":
            return lambda glyph_id, ctx: all(p(glyph_id, ctx) for p in predicates)
        elif cond.operator == "OR":
            return lambda glyph_id, ctx: any(p(glyph_id, ctx) for p in predicates)
        elif cond.operator == "NOT":
            return lambda glyph_id, ctx: not predicates[0](glyph_id, ctx)
        
        return lambda glyph_id, ctx: True
