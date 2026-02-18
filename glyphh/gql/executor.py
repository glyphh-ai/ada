"""
GQL Executor - Executes GQL queries and returns Fact Trees.

The executor is the main entry point for running GQL queries. It handles:
- Parsing GQL strings
- Planning execution
- Executing plans
- Building Fact Tree responses
- Caching results
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional, Set

from glyphh.gql.parser import parse
from glyphh.gql.printer import pretty_print
from glyphh.gql.encoder import QueryEncoder
from glyphh.gql.cache import QueryCache
from glyphh.gql.planner import ExecutionPlanner, ExecutionContext
from glyphh.gql.plans import ExecutionPlan
from glyphh.gql.ast import ASTNode
from glyphh.gql.exceptions import ExecutionError, GQLError
from glyphh.fact_tree import FactTree


class GQLExecutor:
    """
    Executes GQL queries against a model.
    
    The executor provides a high-level interface for running GQL queries.
    It handles parsing, planning, execution, and caching.
    
    Example:
        >>> from glyphh.gql import GQLExecutor, ExecutionContext
        >>> 
        >>> context = ExecutionContext(glyphs=my_glyphs)
        >>> executor = GQLExecutor(context)
        >>> 
        >>> result = executor.execute('FIND SIMILAR TO "red car" LIMIT 10')
        >>> print(result.to_text())
    """
    
    def __init__(
        self,
        context: ExecutionContext,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_threshold: float = 0.95
    ):
        """
        Initialize the executor.
        
        Args:
            context: Execution context with model and glyphs
            enable_cache: Whether to enable query caching
            cache_size: Maximum cache size
            cache_threshold: Similarity threshold for cache hits
        """
        self.context = context
        self.planner = ExecutionPlanner(context)
        self.encoder = QueryEncoder()
        
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = QueryCache(
                self.encoder,
                max_size=cache_size,
                similarity_threshold=cache_threshold
            )
        else:
            self.cache = None
    
    def execute(self, query: str) -> FactTree:
        """
        Execute a GQL query string.
        
        Args:
            query: GQL query string
        
        Returns:
            FactTree with query results
        
        Raises:
            GQLError: If parsing, planning, or execution fails
        """
        start_time = time.time()
        
        # Create base fact tree
        fact_tree = FactTree()
        fact_tree.add_fact(
            path=["query"],
            description="Original Query",
            value=query
        )
        
        try:
            # Parse query
            ast = parse(query)
            normalized_query = pretty_print(ast)
            
            fact_tree.add_fact(
                path=["normalized_query"],
                description="Normalized Query",
                value=normalized_query
            )
            
            # Check cache
            if self.enable_cache and self.cache:
                query_vector = self.encoder.encode(ast)
                cached = self.cache.get(query_vector)
                
                if cached:
                    cached_result, similarity = cached
                    fact_tree.add_fact(
                        path=["cache"],
                        description="Cache Hit",
                        value={
                            "hit": True,
                            "similarity": similarity
                        }
                    )
                    # Merge cached results
                    self._merge_results(fact_tree, cached_result)
                    self._add_metadata(fact_tree, start_time, cache_hit=True)
                    return fact_tree
            
            # Plan execution
            plan = self.planner.plan(ast)
            
            # Execute plan
            result = plan.execute(self.context)
            
            # Merge results into fact tree
            self._merge_results(fact_tree, result)
            
            # Cache result
            if self.enable_cache and self.cache:
                glyph_refs = plan.get_glyph_refs()
                self.cache.put(query_vector, normalized_query, result, glyph_refs)
                fact_tree.add_fact(
                    path=["cache"],
                    description="Cache Status",
                    value={"hit": False, "cached": True}
                )
            
        except GQLError as e:
            fact_tree.add_fact(
                path=["error"],
                description="Query Error",
                value={
                    "type": type(e).__name__,
                    "message": str(e),
                    "suggestion": getattr(e, 'suggestion', None)
                }
            )
        except Exception as e:
            fact_tree.add_fact(
                path=["error"],
                description="Execution Error",
                value={
                    "type": type(e).__name__,
                    "message": str(e)
                }
            )
        
        self._add_metadata(fact_tree, start_time, cache_hit=False)
        return fact_tree
    
    def execute_ast(self, ast: ASTNode) -> FactTree:
        """
        Execute a pre-parsed AST node.
        
        Args:
            ast: Parsed AST node
        
        Returns:
            FactTree with query results
        """
        query = pretty_print(ast)
        return self.execute(query)
    
    def execute_plan(self, plan: ExecutionPlan, query: str = "") -> FactTree:
        """
        Execute a pre-built execution plan.
        
        Args:
            plan: Execution plan
            query: Original query string (for metadata)
        
        Returns:
            FactTree with query results
        """
        start_time = time.time()
        
        fact_tree = FactTree()
        if query:
            fact_tree.add_fact(
                path=["query"],
                description="Original Query",
                value=query
            )
        
        try:
            result = plan.execute(self.context)
            self._merge_results(fact_tree, result)
        except Exception as e:
            fact_tree.add_fact(
                path=["error"],
                description="Execution Error",
                value={
                    "type": type(e).__name__,
                    "message": str(e)
                }
            )
        
        self._add_metadata(fact_tree, start_time, cache_hit=False)
        return fact_tree
    
    def invalidate_glyph(self, glyph_id: str) -> int:
        """
        Invalidate cache entries for a glyph.
        
        Call this when a glyph is updated or deleted.
        
        Args:
            glyph_id: The glyph ID to invalidate
        
        Returns:
            Number of cache entries invalidated
        """
        if self.cache:
            return self.cache.invalidate_for_glyph(glyph_id)
        return 0
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats().to_dict()
        return None
    
    def _merge_results(self, target: FactTree, source: FactTree) -> None:
        """Merge source fact tree into target."""
        # Merge by copying children from source root to target root
        if hasattr(source, 'root') and source.root and hasattr(source.root, 'children'):
            for child in source.root.children:
                # Add each child as a new fact in target
                target.add_fact(
                    path=[getattr(child, 'description', 'result')],
                    description=getattr(child, 'description', ''),
                    value=getattr(child, 'value', None),
                    data_context=getattr(child, 'data_context', None)
                )
    
    def _add_metadata(
        self, 
        fact_tree: FactTree, 
        start_time: float,
        cache_hit: bool
    ) -> None:
        """Add execution metadata to fact tree."""
        elapsed = time.time() - start_time
        
        fact_tree.add_fact(
            path=["metadata"],
            description="Execution Metadata",
            value=None,
            data_context={
                "execution_time_ms": round(elapsed * 1000, 2),
                "timestamp": datetime.now().isoformat(),
                "cache_hit": cache_hit
            }
        )


def execute(query: str, context: ExecutionContext) -> FactTree:
    """
    Convenience function to execute a GQL query.
    
    Args:
        query: GQL query string
        context: Execution context
    
    Returns:
        FactTree with results
    
    Example:
        >>> from glyphh.gql import execute, ExecutionContext
        >>> context = ExecutionContext(glyphs=my_glyphs)
        >>> result = execute('LIST ALL LIMIT 10', context)
    """
    executor = GQLExecutor(context, enable_cache=False)
    return executor.execute(query)
