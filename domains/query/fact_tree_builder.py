"""
FactTree Builder utility for runtime.

Wraps the SDK's FactTree to provide typed builders for each query type.
Handles conversion of runtime types (GlyphResponse) to SDK types (Citation).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID
import hashlib

from glyphh.fact_tree.builder import FactTree, Citation as SDKCitation

from domains.models.schemas import GlyphResponse, ScoredGlyph, PredictedState


class FactTreeBuilder:
    """
    Builder utility for creating SDK FactTrees from query results.
    
    Provides typed methods for each query type to ensure consistent
    FactTree structure across all operations.
    """
    
    @staticmethod
    def create_citation(
        glyph: GlyphResponse,
        component: str = "cortex",
    ) -> SDKCitation:
        """
        Convert a GlyphResponse to an SDK Citation.
        
        Args:
            glyph: The GlyphResponse to convert
            component: The component within the glyph (default: "cortex")
        
        Returns:
            SDK Citation with proper fields
        """
        # Generate data hash from glyph metadata
        metadata_str = str(glyph.metadata) if glyph.metadata else ""
        data_hash = hashlib.sha256(metadata_str.encode()).hexdigest()[:16]
        
        return SDKCitation(
            glyph_id=f"{glyph.id}@{glyph.created_at.isoformat()}",
            component=component,
            timestamp=glyph.updated_at or datetime.utcnow(),
            version="v1",
            data_hash=data_hash,
        )

    @staticmethod
    def build_similarity_search(
        query: str,
        results: List[ScoredGlyph],
        query_time_ms: float,
        total_count: int,
    ) -> FactTree:
        """
        Build a FactTree for similarity search results.
        
        Args:
            query: The original search query
            results: List of scored glyphs
            query_time_ms: Query execution time in milliseconds
            total_count: Total number of results
        
        Returns:
            FactTree with search results and citations
        """
        fact_tree = FactTree()
        fact_tree.root.description = "Similarity Search"
        
        # Add query info
        fact_tree.add_fact(
            path=["query"],
            description="Search Query",
            value=query,
        )
        
        # Add results
        for i, scored_glyph in enumerate(results):
            glyph = scored_glyph.glyph
            citation = FactTreeBuilder.create_citation(glyph)
            
            fact_tree.add_fact(
                path=["results", f"match_{i+1}"],
                description=f"Match {i+1}",
                value={
                    "glyph_id": str(glyph.id),
                    "concept_text": glyph.concept_text,
                    "similarity_score": scored_glyph.similarity_score,
                    "final_score": scored_glyph.final_score,
                    "metadata": glyph.metadata,
                },
                citations=[citation],
                data_context={
                    "similarity_score": scored_glyph.similarity_score,
                    "security_weight": scored_glyph.security_weight,
                },
            )
        
        # Add metadata
        fact_tree.add_fact(
            path=["metadata"],
            description="Execution Metadata",
            value=None,
            data_context={
                "query_time_ms": query_time_ms,
                "total_count": total_count,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        
        return fact_tree

    @staticmethod
    def build_count(
        count: int,
        query_time_ms: float,
        filter_applied: Optional[str] = None,
    ) -> FactTree:
        """
        Build a FactTree for count query results.
        
        Args:
            count: The count result
            query_time_ms: Query execution time in milliseconds
            filter_applied: Optional filter description
        
        Returns:
            FactTree with count result
        """
        fact_tree = FactTree()
        fact_tree.root.description = "Count Query"
        
        # Add count result
        fact_tree.add_fact(
            path=["result"],
            description="Total Count",
            value=count,
            data_context={
                "filter": filter_applied,
            } if filter_applied else {},
        )
        
        # Add metadata
        fact_tree.add_fact(
            path=["metadata"],
            description="Execution Metadata",
            value=None,
            data_context={
                "query_time_ms": query_time_ms,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        
        return fact_tree

    @staticmethod
    def build_list(
        glyphs: List[GlyphResponse],
        query_time_ms: float,
        total_count: int,
        limit: int,
        offset: int = 0,
    ) -> FactTree:
        """
        Build a FactTree for list query results.
        
        Args:
            glyphs: List of glyphs
            query_time_ms: Query execution time in milliseconds
            total_count: Total number of glyphs
            limit: Limit applied to query
            offset: Offset applied to query
        
        Returns:
            FactTree with listed glyphs and citations
        """
        fact_tree = FactTree()
        fact_tree.root.description = "List Query"
        
        # Add glyphs
        for i, glyph in enumerate(glyphs):
            citation = FactTreeBuilder.create_citation(glyph)
            
            fact_tree.add_fact(
                path=["results", f"glyph_{i+1}"],
                description=f"Glyph {i+1}",
                value={
                    "glyph_id": str(glyph.id),
                    "concept_text": glyph.concept_text,
                    "metadata": glyph.metadata,
                    "created_at": glyph.created_at.isoformat(),
                },
                citations=[citation],
            )
        
        # Add metadata
        fact_tree.add_fact(
            path=["metadata"],
            description="Execution Metadata",
            value=None,
            data_context={
                "query_time_ms": query_time_ms,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        
        return fact_tree

    @staticmethod
    def build_temporal_predict(
        predictions: List[PredictedState],
        prediction_time_ms: float,
        current_state: List[str],
        steps_ahead: int,
    ) -> FactTree:
        """
        Build a FactTree for temporal prediction results.
        
        Args:
            predictions: List of predicted states
            prediction_time_ms: Prediction execution time in milliseconds
            current_state: The current state concepts
            steps_ahead: Number of steps predicted
        
        Returns:
            FactTree with predictions and citations
        """
        fact_tree = FactTree()
        fact_tree.root.description = "Temporal Prediction"
        
        # Add current state
        fact_tree.add_fact(
            path=["input"],
            description="Current State",
            value=current_state,
            data_context={
                "steps_ahead": steps_ahead,
            },
        )
        
        # Add predictions
        for i, prediction in enumerate(predictions):
            fact_tree.add_fact(
                path=["predictions", f"step_{i+1}"],
                description=f"Prediction Step {i+1}",
                value={
                    "state_glyphs": [str(g) for g in prediction.state_glyphs],
                    "confidence": prediction.confidence,
                    "path_score": prediction.path_score,
                },
                data_context={
                    "deltas_count": len(prediction.deltas),
                },
            )
            
            # Add deltas as children
            for j, delta in enumerate(prediction.deltas):
                fact_tree.add_fact(
                    path=["predictions", f"step_{i+1}", f"delta_{j+1}"],
                    description=f"Delta {j+1}",
                    value={
                        "glyph_id": str(delta.glyph_id),
                        "change_type": delta.change_type,
                        "magnitude": delta.magnitude,
                    },
                )
        
        # Add metadata
        fact_tree.add_fact(
            path=["metadata"],
            description="Execution Metadata",
            value=None,
            data_context={
                "prediction_time_ms": prediction_time_ms,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        
        return fact_tree

    @staticmethod
    def build_two_stage_result(
        exemplar_match: Dict[str, Any],
        data_results: FactTree,
        gql_query: str,
        total_query_time_ms: float = 0.0,
        exclude_glyph_ids: Optional[Set[str]] = None,
    ) -> FactTree:
        """
        Build a FactTree for two-stage query results.

        Produces a standard "Similarity Search" FactTree so both CLI and UI
        render it like any other similarity search (% bars, concept text).
        The matched exemplar context is preserved in Execution Metadata.

        Args:
            exemplar_match: {glyph_id, concept_text, score, metadata}
            data_results: FactTree from Stage 2 similarity search
            gql_query: The resolved GQL template (for debug)
            total_query_time_ms: Total time for both stages
        """
        fact_tree = FactTree()
        fact_tree.root.description = "Similarity Search"

        # Only add the query/exemplar header when there are Stage 2 data
        # results to display beneath it.  When data_results is None the
        # exemplar itself IS the result — showing it twice is redundant.
        if data_results is not None:
            fact_tree.add_fact(
                path=["query"],
                description="Search Query",
                value=exemplar_match.get("concept_text", ""),
            )

        # Copy Stage 2 data results into the standard "results" path.
        # The GQL executor merges plan results with description like
        # "Found N similar glyphs" and stores matches as a list in value.
        # Filter out ALL pattern/exemplar glyphs from Stage 2 results.
        # Uses three checks:
        #   1. Explicit exclusion set (glyph IDs from stage 1 — all patterns)
        #   2. The matched exemplar's glyph_id (always exclude)
        #   3. metadata record_type == "pattern" (belt-and-suspenders)
        exclude_id = exemplar_match.get("glyph_id", "")
        _exclude_set = exclude_glyph_ids or set()
        has_data_results = False
        try:
            s2_json = data_results.to_json() if hasattr(data_results, "to_json") else {}
            match_idx = 1
            for child in s2_json.get("children", []):
                desc = child.get("description", "")
                # Match both QueryService ("results") and GQL plan ("Found N ...")
                if desc == "results" or desc.startswith("Found "):
                    # GQL plan stores results as [{id, score, concept_text, metadata}]
                    value_list = child.get("value")
                    if isinstance(value_list, list):
                        for item in value_list:
                            glyph_id = item.get("id", "")
                            meta = item.get("metadata") or {}
                            # Skip patterns: by ID set, by exemplar ID, or by metadata
                            if glyph_id == exclude_id:
                                continue
                            if glyph_id and glyph_id in _exclude_set:
                                continue
                            if meta.get("record_type") == "pattern":
                                continue
                            score = item.get("score", 0.0)
                            concept = item.get("concept_text", glyph_id)
                            fact_tree.add_fact(
                                path=["results", f"match_{match_idx}"],
                                description=f"Match {match_idx}",
                                value={
                                    "glyph_id": glyph_id,
                                    "concept_text": concept,
                                    "similarity_score": score,
                                    "final_score": score,
                                    "metadata": meta,
                                },
                            )
                            match_idx += 1
                            has_data_results = True
                    # Also handle child nodes (from QueryService path)
                    for match_node in child.get("children", []):
                        fact_tree.add_fact(
                            path=["results", f"match_{match_idx}"],
                            description=f"Match {match_idx}",
                            value=match_node.get("value"),
                            data_context=match_node.get("data_context"),
                        )
                        match_idx += 1
                        has_data_results = True
        except Exception:
            pass

        # If no Stage 2 data results, include the matched exemplar itself
        # (pattern-only models without gql_query still return the match).
        if not has_data_results:
            exemplar_meta = exemplar_match.get("metadata") or {}
            fact_tree.add_fact(
                path=["results", "match_1"],
                description="Match 1",
                value={
                    "glyph_id": exemplar_match.get("glyph_id", ""),
                    "concept_text": exemplar_match.get("concept_text", ""),
                    "similarity_score": exemplar_match.get("score", 0.0),
                    "final_score": exemplar_match.get("score", 0.0),
                    "metadata": exemplar_meta,
                },
            )

        # Metadata — includes exemplar context for reference
        exemplar_metadata = exemplar_match.get("metadata") or {}
        fact_tree.add_fact(
            path=["metadata"],
            description="Execution Metadata",
            value=None,
            data_context={
                "query_time_ms": total_query_time_ms,
                "gql_query": gql_query,
                "matched_exemplar": {
                    "glyph_id": exemplar_match.get("glyph_id"),
                    "concept_text": exemplar_match.get("concept_text"),
                    "confidence": exemplar_match.get("score"),
                    **exemplar_metadata,
                },
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        return fact_tree

    @staticmethod
    def build_error(
        error_message: str,
        error_type: str,
        query: Optional[str] = None,
    ) -> FactTree:
        """
        Build a FactTree for error responses.
        
        Args:
            error_message: The error message
            error_type: The type of error
            query: Optional original query
        
        Returns:
            FactTree with error information
        """
        fact_tree = FactTree()
        fact_tree.root.description = "Query Error"
        
        if query:
            fact_tree.add_fact(
                path=["query"],
                description="Original Query",
                value=query,
            )
        
        fact_tree.add_fact(
            path=["error"],
            description="Error Details",
            value=error_message,
            data_context={
                "error_type": error_type,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        
        return fact_tree