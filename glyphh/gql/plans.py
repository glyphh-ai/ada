"""
GQL Execution Plans - Intermediate representation for query execution.

Execution plans are created from AST nodes and contain all information
needed to execute a query against the SDK. Plans can be optimized,
cached, and executed independently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import numpy as np

from glyphh.core import Vector


class ExecutionPlan(ABC):
    """
    Base class for execution plans.
    
    Execution plans are the intermediate representation between
    parsed AST and actual SDK operations. They contain resolved
    references and optimized execution strategies.
    """
    
    @abstractmethod
    def execute(self, context: 'ExecutionContext') -> 'FactTree':
        """
        Execute the plan and return a Fact Tree.
        
        Args:
            context: Execution context with model, encoder, etc.
        
        Returns:
            FactTree with query results
        """
        pass
    
    @abstractmethod
    def get_glyph_refs(self) -> Set[str]:
        """
        Get all glyph IDs referenced by this plan.
        
        Used for cache invalidation.
        
        Returns:
            Set of glyph IDs
        """
        pass


@dataclass
class SimilaritySearchPlan(ExecutionPlan):
    """
    Plan for FIND SIMILAR queries.
    
    Attributes:
        target_vector: Pre-encoded target vector (if text query)
        target_glyph_id: Glyph ID to use as target (if glyph ref)
        scope_layer: Optional layer to search within
        scope_segment: Optional segment to search within
        limit: Maximum results to return
        threshold: Minimum similarity score
        predicate: Optional filter function that takes (glyph_id, context) tuple
        predicate_desc: Human-readable predicate description for debugging
    """
    target_vector: Optional[Vector] = None
    target_glyph_id: Optional[str] = None
    scope_layer: Optional[str] = None
    scope_segment: Optional[str] = None
    limit: int = 10
    threshold: float = 0.5
    predicate: Optional[Callable[[str, 'ExecutionContext'], bool]] = None
    predicate_desc: Optional[str] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute similarity search with optional post-ranking filter."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        
        # Get target vector (use scoped embedding when AT LAYER / AT SEGMENT)
        if self.target_vector is not None:
            search_vector = self.target_vector
        elif self.target_glyph_id:
            if self.scope_layer or self.scope_segment:
                search_vector = context.get_embedding_for_scope(
                    self.target_glyph_id, layer=self.scope_layer, segment=self.scope_segment
                )
            else:
                search_vector = context.get_embedding(self.target_glyph_id)
            if search_vector is None:
                raise ValueError(f"No embedding found for target glyph: {self.target_glyph_id}")
        else:
            raise ValueError("No target specified for similarity search")
        
        # Perform similarity search — try native (pgvector) first,
        # fall back to Python loop for in-memory backends.
        native_results = None
        storage = getattr(context, '_storage', None)
        if storage is not None and hasattr(storage, 'find_similar'):
            native_results = storage.find_similar(
                query_vector=search_vector,
                limit=self.limit * 3 if self.predicate else self.limit,
                threshold=self.threshold,
                scope_layer=self.scope_layer,
                scope_segment=self.scope_segment,
            )

        if native_results is not None:
            results = native_results
        else:
            results = []
            for glyph_id, glyph in context.list_glyphs().items():
                # Get comparison vector based on scope using storage abstraction
                if self.scope_layer or self.scope_segment:
                    compare_vector = context.get_embedding_for_scope(
                        glyph_id, layer=self.scope_layer, segment=self.scope_segment
                    )
                else:
                    compare_vector = context.get_embedding(glyph_id)

                if compare_vector is None:
                    continue

                # Compute similarity
                score = context.compute_similarity(search_vector, compare_vector)

                if score >= self.threshold:
                    results.append({
                        "glyph_id": glyph_id,
                        "score": score,
                        "glyph": glyph
                    })

            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply predicate filter AFTER ranking (Requirement 1.4)
        if self.predicate:
            results = [
                r for r in results
                if self.predicate(r["glyph_id"], context)
            ]
        
        # Apply limit AFTER filtering
        results = results[:self.limit]
        
        # Build fact tree with full glyph metadata
        result_values = []
        for r in results:
            glyph = r.get("glyph")
            metadata = getattr(glyph, "metadata", None) or {}
            result_values.append({
                "id": r["glyph_id"],
                "score": r["score"],
                "concept_text": getattr(glyph, "concept_text", r["glyph_id"]),
                "metadata": metadata,
            })

        fact_tree.add_fact(
            path=["results"],
            description=f"Found {len(results)} similar glyphs",
            value=result_values,
            data_context={
                "threshold": self.threshold,
                "limit": self.limit,
                "scope": f"{self.scope_layer}.{self.scope_segment}" if self.scope_segment else self.scope_layer,
                "filter": self.predicate_desc,
            }
        )
        
        return fact_tree
    
    def get_glyph_refs(self) -> Set[str]:
        refs = set()
        if self.target_glyph_id:
            refs.add(self.target_glyph_id)
        return refs


@dataclass
class ListPlan(ExecutionPlan):
    """
    Plan for LIST queries.
    
    Uses storage abstraction for iteration and predicate evaluation:
    - Iterates via context.list_glyphs()
    - Evaluates predicates via context.get_glyph_attribute(glyph_id, attr)
    
    Attributes:
        predicate: Optional filter function that takes (glyph_id, context) tuple
        predicate_desc: Human-readable predicate description
        limit: Maximum results to return
    """
    predicate: Optional[Callable[[str, 'ExecutionContext'], bool]] = None
    predicate_desc: Optional[str] = None
    limit: int = 100
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute list query using storage abstraction."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        results = []
        
        # Use context.list_glyphs() for iteration (Requirement 3.2)
        for glyph_id, glyph in context.list_glyphs().items():
            # Predicate evaluation uses glyph_id and context for attribute access (Requirement 3.4)
            if self.predicate is None or self.predicate(glyph_id, context):
                results.append({
                    "glyph_id": glyph_id,
                    "glyph": glyph
                })
                if len(results) >= self.limit:
                    break
        
        # Build result values with glyph details
        result_values = []
        for r in results:
            glyph_id = r["glyph_id"]
            result_entry = {"id": glyph_id}
            
            # Add concept_text if available
            concept_text = context.get_glyph_attribute(glyph_id, "concept_text")
            if concept_text is not None:
                result_entry["concept_text"] = concept_text
            
            # Add metadata if available
            metadata = context.get_glyph_attribute(glyph_id, "metadata")
            if metadata is not None:
                result_entry["metadata"] = metadata
            
            result_values.append(result_entry)
        
        fact_tree.add_fact(
            path=["results"],
            description=f"Listed {len(results)} glyphs",
            value=result_values,
            data_context={
                "limit": self.limit,
                "filter": self.predicate_desc
            }
        )
        
        return fact_tree
    
    def get_glyph_refs(self) -> Set[str]:
        return set()


@dataclass
class CountPlan(ExecutionPlan):
    """
    Plan for COUNT queries.
    
    Uses storage abstraction for iteration and predicate evaluation:
    - Iterates via context.list_glyphs()
    - Evaluates predicates via context.get_glyph_attribute(glyph_id, attr)
    
    Attributes:
        predicate: Optional filter function that takes (glyph_id, context) tuple
        predicate_desc: Human-readable predicate description
    """
    predicate: Optional[Callable[[str, 'ExecutionContext'], bool]] = None
    predicate_desc: Optional[str] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute count query using storage abstraction."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        count = 0
        
        # Use context.list_glyphs() for iteration
        for glyph_id, glyph in context.list_glyphs().items():
            # Predicate evaluation uses glyph_id and context for attribute access
            if self.predicate is None or self.predicate(glyph_id, context):
                count += 1
        
        fact_tree.add_fact(
            path=["count"],
            description=f"Counted {count} glyphs",
            value=count,
            data_context={
                "filter": self.predicate_desc
            }
        )
        
        return fact_tree
    
    def get_glyph_refs(self) -> Set[str]:
        return set()


@dataclass
class ComparePlan(ExecutionPlan):
    """
    Plan for COMPARE queries.
    
    Attributes:
        glyph1_id: First glyph ID
        glyph2_id: Second glyph ID
        edge_type: Edge type for comparison (neural_cortex, neural_layer, etc.)
        level_path: Optional path within the level
    """
    glyph1_id: str = ""
    glyph2_id: str = ""
    edge_type: str = "neural_cortex"
    level_path: Optional[str] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute comparison."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        
        # Get vectors based on edge type using storage abstraction
        if self.edge_type == "neural_cortex":
            v1 = context.get_embedding(self.glyph1_id)
            v2 = context.get_embedding(self.glyph2_id)
        elif self.edge_type == "neural_layer" and self.level_path:
            v1 = context.get_embedding_for_scope(self.glyph1_id, layer=self.level_path)
            v2 = context.get_embedding_for_scope(self.glyph2_id, layer=self.level_path)
        elif self.edge_type == "neural_segment" and self.level_path:
            parts = self.level_path.split(".")
            layer, segment = parts[0], parts[1] if len(parts) > 1 else None
            v1 = context.get_embedding_for_scope(self.glyph1_id, layer=layer, segment=segment)
            v2 = context.get_embedding_for_scope(self.glyph2_id, layer=layer, segment=segment)
        else:
            v1 = context.get_embedding(self.glyph1_id)
            v2 = context.get_embedding(self.glyph2_id)
        
        # Compute similarity
        similarity = context.compute_similarity(v1, v2) if v1 and v2 else 0.0
        
        fact_tree.add_fact(
            path=["comparison"],
            description=f"Compared {self.glyph1_id} to {self.glyph2_id}",
            value={
                "glyph1": self.glyph1_id,
                "glyph2": self.glyph2_id,
                "similarity": similarity,
                "level": self.edge_type,
                "path": self.level_path
            }
        )
        
        return fact_tree
    
    def get_glyph_refs(self) -> Set[str]:
        return {self.glyph1_id, self.glyph2_id}


@dataclass
class PredictPlan(ExecutionPlan):
    """
    Plan for PREDICT queries.
    
    Attributes:
        glyph_id: Glyph to predict
        role_path: Optional role path to predict
        window_seconds: Time window in seconds
    """
    glyph_id: str = ""
    role_path: Optional[str] = None
    window_seconds: Optional[int] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute prediction."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        
        # Get temporal predictor from context
        predictor = context.get_predictor()
        if predictor is None:
            fact_tree.add_fact(
                path=["error"],
                description="Prediction not available",
                value="No temporal predictor configured"
            )
            return fact_tree
        
        glyph = context.get_glyph(self.glyph_id)
        
        # Perform prediction
        try:
            prediction = predictor.predict(glyph, role=self.role_path)
            fact_tree.add_fact(
                path=["prediction"],
                description=f"Predicted future state for {self.glyph_id}",
                value={
                    "glyph_id": self.glyph_id,
                    "role": self.role_path,
                    "predictions": prediction
                }
            )
        except Exception as e:
            fact_tree.add_fact(
                path=["error"],
                description="Prediction failed",
                value=str(e)
            )
        
        return fact_tree
    
    def get_glyph_refs(self) -> Set[str]:
        return {self.glyph_id}


@dataclass
class DriftPlan(ExecutionPlan):
    """
    Plan for DETECT DRIFT queries.
    
    Uses storage abstraction for embedding access:
    - Uses context.get_embedding() for cortex-level comparison
    - Uses context.get_embedding_for_scope() for layer/segment comparison
    - Handles missing embeddings gracefully (returns 0.0 similarity)
    
    Attributes:
        from_glyph_id: Baseline glyph ID
        to_glyph_id: Current glyph ID
        edge_type: Edge type for comparison
        level_path: Optional path within the level
        threshold: Optional drift threshold for alerting
    """
    from_glyph_id: str = ""
    to_glyph_id: str = ""
    edge_type: str = "neural_cortex"
    level_path: Optional[str] = None
    threshold: Optional[float] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute drift detection using storage abstraction."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        
        # Get vectors based on edge type using storage abstraction (Requirement 3.4)
        if self.edge_type == "neural_cortex":
            v1 = context.get_embedding(self.from_glyph_id)
            v2 = context.get_embedding(self.to_glyph_id)
        elif self.edge_type == "neural_layer" and self.level_path:
            v1 = context.get_embedding_for_scope(self.from_glyph_id, layer=self.level_path)
            v2 = context.get_embedding_for_scope(self.to_glyph_id, layer=self.level_path)
        elif self.edge_type == "neural_segment" and self.level_path:
            # Parse layer.segment path
            parts = self.level_path.split(".")
            layer = parts[0]
            segment = parts[1] if len(parts) > 1 else None
            v1 = context.get_embedding_for_scope(self.from_glyph_id, layer=layer, segment=segment)
            v2 = context.get_embedding_for_scope(self.to_glyph_id, layer=layer, segment=segment)
        else:
            v1 = context.get_embedding(self.from_glyph_id)
            v2 = context.get_embedding(self.to_glyph_id)
        
        # Handle missing embeddings gracefully (Requirement 3.5)
        # If either embedding is None, similarity is 0.0 (maximum drift)
        if v1 is None or v2 is None:
            similarity = 0.0
            missing_embeddings = True
        else:
            similarity = context.compute_similarity(v1, v2)
            missing_embeddings = False
        
        drift = 1.0 - similarity
        
        # Check threshold
        alert = self.threshold is not None and drift > self.threshold
        
        fact_tree.add_fact(
            path=["drift"],
            description=f"Drift detection from {self.from_glyph_id} to {self.to_glyph_id}",
            value={
                "from_glyph": self.from_glyph_id,
                "to_glyph": self.to_glyph_id,
                "similarity": similarity,
                "drift": drift,
                "threshold": self.threshold,
                "alert": alert,
                "missing_embeddings": missing_embeddings
            }
        )
        
        return fact_tree
    
    def get_glyph_refs(self) -> Set[str]:
        return {self.from_glyph_id, self.to_glyph_id}


@dataclass
class IntrospectPlan(ExecutionPlan):
    """
    Plan for INTROSPECT queries.
    
    Uses storage abstraction for glyph access:
    - Uses context.get_glyph() to retrieve the glyph
    - Uses context.get_glyph_attribute() for accessing glyph properties
    - Handles missing attributes gracefully
    
    Attributes:
        glyph_id: Glyph to introspect
        decompose_level: Level to decompose by (layer, segment, role)
    """
    glyph_id: str = ""
    decompose_level: str = "layer"
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute introspection using storage abstraction."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        
        # Get glyph using storage abstraction (Requirement 3.4)
        glyph = context.get_glyph(self.glyph_id)
        
        # Use context.decompose_glyph which handles the structure internally
        # This method already handles missing layers/segments gracefully (Requirement 3.5)
        structure = context.decompose_glyph(glyph, self.decompose_level)
        
        # Get additional attributes using storage abstraction
        identifier = context.get_glyph_attribute(self.glyph_id, 'identifier')
        if identifier is None:
            # Fallback to 'id' attribute for database glyphs
            identifier = context.get_glyph_attribute(self.glyph_id, 'id')
        if identifier is None:
            identifier = self.glyph_id
        
        fact_tree.add_fact(
            path=["introspection"],
            description=f"Introspected {self.glyph_id} by {self.decompose_level}",
            value={
                "glyph_id": str(identifier),
                "decompose_by": self.decompose_level,
                "structure": structure
            }
        )
        
        return fact_tree
    
    def get_glyph_refs(self) -> Set[str]:
        return {self.glyph_id}


@dataclass
class TrendPlan(ExecutionPlan):
    """
    Plan for TREND queries.
    
    Uses storage abstraction for embedding access:
    - Uses context.get_embedding() for cortex-level comparison
    - Uses context.get_embedding_for_scope() for layer/segment comparison
    - Uses context.get_glyph_attribute() for accessing glyph properties
    - Handles missing embeddings gracefully
    
    Attributes:
        glyph_id: Glyph to track
        window_seconds: Time window in seconds
        edge_type: Edge type for comparison
        level_path: Optional path within the level
        alert_threshold: Optional drift threshold for alerting
    """
    glyph_id: str = ""
    window_seconds: int = 0
    edge_type: str = "neural_cortex"
    level_path: Optional[str] = None
    alert_threshold: Optional[float] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute trend analysis using storage abstraction."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        
        # Get historical data points
        history = context.get_glyph_history(self.glyph_id, self.window_seconds)
        
        if len(history) < 2:
            fact_tree.add_fact(
                path=["trend"],
                description=f"Insufficient data for trend analysis",
                value={
                    "glyph_id": self.glyph_id,
                    "data_points": len(history),
                    "error": "Need at least 2 data points"
                }
            )
            return fact_tree
        
        # Compute trend (simplified - just compare first and last)
        first = history[0]
        last = history[-1]
        
        # Get glyph IDs using storage abstraction (Requirement 3.4)
        # Try 'identifier' first (SDK glyphs), then 'id' (database glyphs)
        first_id = self._get_glyph_id(first, context)
        last_id = self._get_glyph_id(last, context)
        
        # Get embeddings using storage abstraction (Requirement 3.4)
        v1 = None
        v2 = None
        
        if self.edge_type == "neural_cortex":
            if first_id:
                v1 = context.get_embedding(first_id)
            if last_id:
                v2 = context.get_embedding(last_id)
        elif self.edge_type == "neural_layer" and self.level_path:
            if first_id:
                v1 = context.get_embedding_for_scope(first_id, layer=self.level_path)
            if last_id:
                v2 = context.get_embedding_for_scope(last_id, layer=self.level_path)
        elif self.edge_type == "neural_segment" and self.level_path:
            # Parse layer.segment path
            parts = self.level_path.split(".")
            layer = parts[0]
            segment = parts[1] if len(parts) > 1 else None
            if first_id:
                v1 = context.get_embedding_for_scope(first_id, layer=layer, segment=segment)
            if last_id:
                v2 = context.get_embedding_for_scope(last_id, layer=layer, segment=segment)
        else:
            if first_id:
                v1 = context.get_embedding(first_id)
            if last_id:
                v2 = context.get_embedding(last_id)
        
        # Handle missing embeddings gracefully (Requirement 3.5)
        if v1 is None or v2 is None:
            similarity = 0.0
            missing_embeddings = True
        else:
            similarity = context.compute_similarity(v1, v2)
            missing_embeddings = False
        
        drift = 1.0 - similarity
        alert = self.alert_threshold is not None and drift > self.alert_threshold
        
        fact_tree.add_fact(
            path=["trend"],
            description=f"Trend analysis for {self.glyph_id}",
            value={
                "glyph_id": self.glyph_id,
                "window_seconds": self.window_seconds,
                "data_points": len(history),
                "drift": drift,
                "alert_threshold": self.alert_threshold,
                "alert": alert,
                "missing_embeddings": missing_embeddings
            }
        )
        
        return fact_tree
    
    def _get_glyph_id(self, glyph: Any, context: 'ExecutionContext') -> Optional[str]:
        """
        Extract glyph ID from a glyph object using storage abstraction.
        
        Tries 'identifier' (SDK glyphs) first, then 'id' (database glyphs).
        """
        # Try to get identifier attribute
        identifier = getattr(glyph, 'identifier', None)
        if identifier:
            return str(identifier)
        
        # Try to get id attribute
        glyph_id = getattr(glyph, 'id', None)
        if glyph_id:
            return str(glyph_id)
        
        return None
    
    def get_glyph_refs(self) -> Set[str]:
        return {self.glyph_id}


@dataclass
class AggregatePlan(ExecutionPlan):
    """
    Plan for AGGREGATE queries.
    
    Uses storage abstraction for iteration and predicate evaluation:
    - Iterates via context.list_glyphs()
    - Evaluates predicates via context.get_glyph_attribute(glyph_id, attr)
    
    Attributes:
        function: Aggregation function (COUNT, AVG, MIN, MAX, SUM)
        field: Field to aggregate
        group_by: Fields to group by
        predicate: Optional filter function that takes (glyph_id, context) tuple
        predicate_desc: Human-readable predicate description
    """
    function: str = "COUNT"
    field: Optional[str] = None
    group_by: Optional[List[str]] = None
    predicate: Optional[Callable[[str, 'ExecutionContext'], bool]] = None
    predicate_desc: Optional[str] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute aggregation using storage abstraction."""
        from glyphh.fact_tree import FactTree
        
        fact_tree = FactTree()
        
        # Filter glyphs using storage abstraction
        glyphs = []
        for glyph_id, glyph in context.list_glyphs().items():
            # Predicate evaluation uses glyph_id and context for attribute access
            if self.predicate is None or self.predicate(glyph_id, context):
                glyphs.append((glyph_id, glyph))
        
        # Group if needed
        if self.group_by:
            groups: Dict[tuple, List] = {}
            for glyph_id, glyph in glyphs:
                # Use storage abstraction for attribute access
                key = tuple(
                    context.get_glyph_attribute(glyph_id, f) 
                    for f in self.group_by
                )
                if key not in groups:
                    groups[key] = []
                groups[key].append((glyph_id, glyph))
            
            # Aggregate each group
            results = []
            for key, group_glyphs in groups.items():
                agg_value = self._aggregate(group_glyphs, context)
                results.append({
                    "group": dict(zip(self.group_by, key)),
                    "value": agg_value
                })
        else:
            # Single aggregation
            agg_value = self._aggregate(glyphs, context)
            results = [{"value": agg_value}]
        
        fact_tree.add_fact(
            path=["aggregation"],
            description=f"Aggregated {self.function} over {len(glyphs)} glyphs",
            value={
                "function": self.function,
                "field": self.field,
                "group_by": self.group_by,
                "results": results
            }
        )
        
        return fact_tree
    
    def _aggregate(self, glyphs: List, context: 'ExecutionContext') -> Any:
        """Perform aggregation on a list of glyphs using storage abstraction."""
        if self.function == "COUNT":
            return len(glyphs)
        
        if not self.field:
            return None
        
        values = []
        for glyph_id, glyph in glyphs:
            # Use storage abstraction for attribute access
            val = context.get_glyph_attribute(glyph_id, self.field)
            if val is not None and isinstance(val, (int, float)):
                values.append(val)
        
        if not values:
            return None
        
        if self.function == "AVG":
            return sum(values) / len(values)
        elif self.function == "MIN":
            return min(values)
        elif self.function == "MAX":
            return max(values)
        elif self.function == "SUM":
            return sum(values)
        
        return None
    
    def get_glyph_refs(self) -> Set[str]:
        return set()
