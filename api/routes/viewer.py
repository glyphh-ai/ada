"""
Viewer API Routes for Glyphh Runtime.

Endpoints for 3D visualization data.
All routes scoped by /{org_id}/{model_id}/viewer/...
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.storage import GlyphStorage
from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/viewer", tags=["viewer"])


# =============================================================================
# Response Models
# =============================================================================

class ViewerLayerData(BaseModel):
    """Layer data for viewer."""
    index: int = Field(..., description="Layer index")
    cortex: List[bool] = Field(..., description="Cortex bit vector")


class ViewerGlyphData(BaseModel):
    """Glyph data for 3D viewer."""
    name: str = Field(..., description="Glyph concept text (display name)")
    glyph_id: str = Field(..., description="Glyph UUID")
    node_type: str = Field(default="concept", description="Node type")
    layers: List[ViewerLayerData] = Field(default_factory=list, description="Layer cortex data")
    semantic: Dict[str, Any] = Field(default_factory=dict, description="Semantic metadata")


class ViewerEdge(BaseModel):
    """Edge data for viewer."""
    source: str = Field(..., description="Source glyph name")
    target: str = Field(..., description="Target glyph name")
    weight: float = Field(default=1.0, description="Edge weight")


class ViewerEdges(BaseModel):
    """Grouped edges by type."""
    semantic: List[ViewerEdge] = Field(default_factory=list)
    neural: List[ViewerEdge] = Field(default_factory=list)
    hierarchy: List[ViewerEdge] = Field(default_factory=list)
    temporal: List[ViewerEdge] = Field(default_factory=list)


class ViewerDataResponse(BaseModel):
    """Complete viewer data response."""
    glyphs: List[ViewerGlyphData] = Field(..., description="List of glyphs")
    edges: ViewerEdges = Field(default_factory=ViewerEdges, description="Edges by type")
    total: int = Field(..., description="Total glyph count")


class TemporalDataPoint(BaseModel):
    """A single point in temporal history."""
    glyph_id: str = Field(..., description="Glyph UUID")
    name: str = Field(..., description="Glyph concept text")
    label: str = Field(..., description="Readable label")
    temporal_value: Any = Field(..., description="Temporal value (date, year, etc.)")
    temporal_key: str = Field(..., description="Key used for temporal ordering")
    values: Dict[str, Any] = Field(default_factory=dict, description="All semantic values")
    changes: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Changes from previous point: {field: {from, to, delta}}"
    )


class TemporalHistoryResponse(BaseModel):
    """Temporal history for a glyph following temporal edges."""
    glyph_id: str = Field(..., description="Starting glyph UUID")
    history: List[TemporalDataPoint] = Field(..., description="Temporal history points")
    temporal_key: str = Field(..., description="Key used for temporal ordering")
    total_points: int = Field(..., description="Total points in history")


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_storage(db: AsyncSession = Depends(get_db)) -> GlyphStorage:
    return GlyphStorage(db)


# =============================================================================
# Helper Functions
# =============================================================================

def embedding_to_cortex_bits(embedding: List[float], threshold: float = 0.0) -> List[bool]:
    """
    Convert embedding vector to boolean cortex bits.
    
    Uses threshold to determine which dimensions are "active".
    """
    if not embedding:
        return []
    return [v > threshold for v in embedding]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/data", response_model=ViewerDataResponse)
async def get_viewer_data(
    org_id: str,
    model_id: str,
    limit: int = Query(100, ge=1, le=500, description="Max glyphs to return"),
    storage: GlyphStorage = Depends(get_storage),
) -> ViewerDataResponse:
    """
    Get glyph data formatted for 3D viewer visualization.
    
    Returns glyphs with cortex bit vectors. Edges are computed client-side
    based on cortex similarity.
    """
    try:
        # Get glyphs with embeddings
        glyphs, embeddings = await storage.list_glyphs_with_embeddings(
            org_id, model_id, limit=limit
        )
        
        if not glyphs:
            return ViewerDataResponse(glyphs=[], edges=ViewerEdges(), total=0)
        
        # Get hierarchical embeddings for layer data
        glyph_uuids = [g.id for g in glyphs]
        hierarchical = await storage.get_hierarchical_embeddings(
            org_id, model_id, glyph_uuids
        )
        
        # Build viewer glyph data
        viewer_glyphs: List[ViewerGlyphData] = []
        
        for glyph in glyphs:
            glyph_id_str = str(glyph.id)
            
            # Get cortex embedding and convert to bits
            cortex_embedding = embeddings.get(glyph_id_str, [])
            cortex_bits = embedding_to_cortex_bits(cortex_embedding)
            
            # Build layer data from hierarchical embeddings
            layers: List[ViewerLayerData] = []
            if glyph_id_str in hierarchical:
                glyph_hier = hierarchical[glyph_id_str]
                if 'layer' in glyph_hier:
                    for layer_path, layer_embedding in glyph_hier['layer'].items():
                        # Extract layer index from path (e.g., "layer0" -> 0)
                        try:
                            layer_idx = int(layer_path.replace('layer', ''))
                        except ValueError:
                            layer_idx = 0
                        
                        layer_bits = embedding_to_cortex_bits(layer_embedding)
                        layers.append(ViewerLayerData(index=layer_idx, cortex=layer_bits))
            
            # If no layer data, use cortex as layer 0
            if not layers and cortex_bits:
                layers.append(ViewerLayerData(index=0, cortex=cortex_bits))
            
            # Extract semantic metadata from glyph.metadata AND from concept_text (if JSON)
            semantic: Dict[str, Any] = {}
            
            # Helper to recursively flatten nested dicts
            def flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
                items: Dict[str, Any] = {}
                for k, v in d.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.update(flatten_dict(v, new_key))
                    else:
                        items[new_key] = v
                return items
            
            # First try to parse concept_text as JSON to extract semantic fields
            try:
                parsed_concept = json.loads(glyph.concept_text)
                if isinstance(parsed_concept, dict):
                    # Flatten nested structure
                    flattened = flatten_dict(parsed_concept)
                    for key, value in flattened.items():
                        if not key.startswith('_'):
                            semantic[key] = value
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Then add/override with explicit metadata
            if glyph.metadata:
                for key, value in glyph.metadata.items():
                    if not key.startswith('_'):
                        semantic[key] = value
            
            viewer_glyphs.append(ViewerGlyphData(
                name=glyph.concept_text,
                glyph_id=glyph_id_str,
                node_type="concept",
                layers=layers,
                semantic=semantic,
            ))
        
        # Edges are computed client-side based on cortex similarity
        # But we can also compute semantic edges based on shared metadata values
        edges = ViewerEdges()
        
        # Debug: log semantic data for first few glyphs
        for i, vg in enumerate(viewer_glyphs[:3]):
            logger.info(f"Glyph {i} '{vg.name[:50]}...' semantic keys: {list(vg.semantic.keys())}")
        
        # Build semantic edges based on shared metadata values
        # Glyphs that share the same value for a key are semantically related
        semantic_edges_map: Dict[str, set] = {}  # key -> set of glyph names
        
        for vg in viewer_glyphs:
            for key, value in vg.semantic.items():
                if value is not None and not isinstance(value, (dict, list)):
                    edge_key = f"{key}:{value}"
                    if edge_key not in semantic_edges_map:
                        semantic_edges_map[edge_key] = set()
                    semantic_edges_map[edge_key].add(vg.name)
        
        logger.info(f"Semantic edge groups with 2+ glyphs: {sum(1 for v in semantic_edges_map.values() if len(v) >= 2)}")
        
        # Create edges between glyphs that share semantic values
        seen_pairs: set = set()
        for glyph_names in semantic_edges_map.values():
            if len(glyph_names) < 2:
                continue
            names_list = list(glyph_names)
            for i in range(len(names_list)):
                for j in range(i + 1, len(names_list)):
                    pair = tuple(sorted([names_list[i], names_list[j]]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        edges.semantic.append(ViewerEdge(
                            source=names_list[i],
                            target=names_list[j],
                            weight=0.5,
                        ))
        
        # Build hierarchy edges based on parent-child relationships in metadata
        # Look for fields like 'parent', 'parent_id', 'category', 'group', 'type'
        # Also check nested paths like 'vehicle.identity.make'
        hierarchy_keys = ['parent', 'parent_id', 'category', 'group', 'type', 'class', 'make', 'brand', 'manufacturer']
        hierarchy_groups: Dict[str, Dict[str, List[str]]] = {}  # key -> {value -> [glyph_names]}
        
        for vg in viewer_glyphs:
            for sem_key, value in vg.semantic.items():
                # Check if any hierarchy key is in the semantic key (handles nested paths)
                key_parts = sem_key.lower().split('.')
                matching_key = None
                for hk in hierarchy_keys:
                    if hk in key_parts or sem_key.lower().endswith(hk):
                        matching_key = hk
                        break
                
                if matching_key and value is not None and not isinstance(value, (dict, list)):
                    str_value = str(value)
                    if matching_key not in hierarchy_groups:
                        hierarchy_groups[matching_key] = {}
                    if str_value not in hierarchy_groups[matching_key]:
                        hierarchy_groups[matching_key][str_value] = []
                    hierarchy_groups[matching_key][str_value].append(vg.name)
        
        # Create hierarchy edges - connect glyphs in same hierarchy group
        hierarchy_seen: set = set()
        for key, value_groups in hierarchy_groups.items():
            for group_name, glyph_names in value_groups.items():
                if len(glyph_names) < 2:
                    continue
                # Connect sequentially within group (like a chain)
                for i in range(len(glyph_names) - 1):
                    pair = tuple(sorted([glyph_names[i], glyph_names[i + 1]]))
                    if pair not in hierarchy_seen:
                        hierarchy_seen.add(pair)
                        edges.hierarchy.append(ViewerEdge(
                            source=glyph_names[i],
                            target=glyph_names[i + 1],
                            weight=0.7,
                        ))
        
        # Build temporal edges based on time-related fields
        # Look for fields like 'year', 'date', 'timestamp', 'created_at', 'time'
        temporal_keys = ['year', 'date', 'timestamp', 'created_at', 'time', 'period', 'quarter', 'month']
        temporal_glyphs: List[tuple] = []  # [(glyph_name, temporal_value)]
        
        for vg in viewer_glyphs:
            for sem_key, value in vg.semantic.items():
                # Check if any temporal key is in the semantic key
                key_parts = sem_key.lower().split('.')
                is_temporal = any(tk in key_parts or sem_key.lower().endswith(tk) for tk in temporal_keys)
                
                if is_temporal and value is not None:
                    try:
                        # Try to convert to sortable value
                        if isinstance(value, (int, float)):
                            temporal_glyphs.append((vg.name, float(value)))
                        elif isinstance(value, str):
                            # Try parsing as number
                            temporal_glyphs.append((vg.name, float(value)))
                    except (ValueError, TypeError):
                        pass
                    break  # Only use first temporal field found
        
        # Sort by temporal value and create sequential edges
        if len(temporal_glyphs) >= 2:
            temporal_glyphs.sort(key=lambda x: x[1])
            for i in range(len(temporal_glyphs) - 1):
                edges.temporal.append(ViewerEdge(
                    source=temporal_glyphs[i][0],
                    target=temporal_glyphs[i + 1][0],
                    weight=0.6,
                ))
        
        total = await storage.count_glyphs(org_id, model_id)
        
        logger.info(f"Viewer edges: semantic={len(edges.semantic)}, hierarchy={len(edges.hierarchy)}, temporal={len(edges.temporal)}")
        
        # Build neural edges based on cortex similarity (kNN)
        # This computes similarity between glyph embeddings
        if len(viewer_glyphs) >= 2 and embeddings:
            import numpy as np
            
            K = 8  # kNN neighbors
            MIN_SIM = 0.1  # Minimum similarity threshold
            
            # Build embedding matrix
            glyph_names = [vg.name for vg in viewer_glyphs]
            embedding_list = []
            valid_indices = []
            
            for i, vg in enumerate(viewer_glyphs):
                emb = embeddings.get(vg.glyph_id, [])
                if emb:
                    embedding_list.append(emb)
                    valid_indices.append(i)
            
            if len(embedding_list) >= 2:
                # Convert to numpy for efficient computation
                emb_matrix = np.array(embedding_list)
                
                # Compute pairwise Jaccard similarity on binarized embeddings
                binary_matrix = (emb_matrix > 0).astype(float)
                
                neural_edge_map: Dict[str, float] = {}  # "src|tgt" -> similarity
                
                for i in range(len(valid_indices)):
                    a_bits = binary_matrix[i]
                    scores = []
                    
                    for j in range(len(valid_indices)):
                        if i == j:
                            continue
                        b_bits = binary_matrix[j]
                        
                        # Jaccard similarity
                        intersection = np.sum(np.logical_and(a_bits, b_bits))
                        union = np.sum(np.logical_or(a_bits, b_bits))
                        sim = intersection / union if union > 0 else 0
                        
                        if sim >= MIN_SIM:
                            scores.append((j, sim))
                    
                    # Sort by similarity and take top K
                    scores.sort(key=lambda x: x[1], reverse=True)
                    top_k = scores[:K]
                    
                    for j, sim in top_k:
                        src_name = glyph_names[valid_indices[i]]
                        tgt_name = glyph_names[valid_indices[j]]
                        key = f"{min(src_name, tgt_name)}|{max(src_name, tgt_name)}"
                        
                        # Keep highest similarity for each pair
                        if key not in neural_edge_map or sim > neural_edge_map[key]:
                            neural_edge_map[key] = sim
                
                # Convert to edges
                for key, sim in neural_edge_map.items():
                    src, tgt = key.split('|')
                    edges.neural.append(ViewerEdge(
                        source=src,
                        target=tgt,
                        weight=sim,
                    ))
                
                logger.info(f"Neural edges computed: {len(edges.neural)}")
        
        return ViewerDataResponse(
            glyphs=viewer_glyphs,
            edges=edges,
            total=total,
        )
        
    except Exception as e:
        logger.error(f"Failed to get viewer data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def extract_readable_label(concept_text: str) -> str:
    """Extract a readable label from concept_text JSON."""
    if not concept_text:
        return 'Unknown'
    
    try:
        parsed = json.loads(concept_text)
        if isinstance(parsed, dict):
            priority_keys = ['make', 'model', 'year', 'name', 'title', 'id', 'type', 'key', 'code', 'label']
            
            def find_values(obj: Dict[str, Any], keys: List[str]) -> List[str]:
                found = []
                for key in keys:
                    if key in obj and obj[key] is not None:
                        val = obj[key]
                        if isinstance(val, (str, int, float)):
                            found.append(str(val))
                if not found:
                    for value in obj.values():
                        if isinstance(value, dict):
                            nested = find_values(value, keys)
                            if nested:
                                return nested
                return found
            
            values = find_values(parsed, priority_keys)
            if values:
                return ' '.join(values[:3])
    except (json.JSONDecodeError, TypeError):
        pass
    
    if len(concept_text) > 40:
        return concept_text[:37] + '...'
    return concept_text


@router.get("/temporal-history/{glyph_id}", response_model=TemporalHistoryResponse)
async def get_temporal_history(
    org_id: str,
    model_id: str,
    glyph_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max history points"),
    storage: GlyphStorage = Depends(get_storage),
) -> TemporalHistoryResponse:
    """
    Get temporal history for a glyph by following temporal edges.
    
    Returns a sequence of glyphs connected by temporal relationships,
    sorted by temporal value, with change deltas between consecutive points.
    """
    try:
        # Get all glyphs to find temporal relationships
        glyphs, _ = await storage.list_glyphs_with_embeddings(
            org_id, model_id, limit=500  # Get more to find temporal chain
        )
        
        if not glyphs:
            return TemporalHistoryResponse(
                glyph_id=glyph_id,
                history=[],
                temporal_key="",
                total_points=0,
            )
        
        # Find the target glyph
        target_glyph = None
        for g in glyphs:
            if str(g.id) == glyph_id:
                target_glyph = g
                break
        
        if not target_glyph:
            raise HTTPException(status_code=404, detail=f"Glyph {glyph_id} not found")
        
        # Extract semantic data from all glyphs
        temporal_keys = ['year', 'date', 'timestamp', 'created_at', 'time', 'period', 'service_date', 'mileage']
        
        def extract_semantic(glyph) -> Dict[str, Any]:
            semantic = {}
            try:
                parsed = json.loads(glyph.concept_text)
                if isinstance(parsed, dict):
                    def flatten(d: Dict, prefix: str = '') -> Dict:
                        items = {}
                        for k, v in d.items():
                            new_key = f"{prefix}.{k}" if prefix else k
                            if isinstance(v, dict):
                                items.update(flatten(v, new_key))
                            else:
                                items[new_key] = v
                        return items
                    semantic = flatten(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
            if glyph.metadata:
                for k, v in glyph.metadata.items():
                    if not k.startswith('_'):
                        semantic[k] = v
            return semantic
        
        def find_temporal_value(semantic: Dict[str, Any]) -> tuple:
            """Find temporal key and value from semantic data."""
            for sem_key, value in semantic.items():
                key_parts = sem_key.lower().split('.')
                for tk in temporal_keys:
                    if tk in key_parts or sem_key.lower().endswith(tk):
                        if value is not None:
                            try:
                                if isinstance(value, (int, float)):
                                    return sem_key, float(value)
                                elif isinstance(value, str):
                                    return sem_key, float(value)
                            except (ValueError, TypeError):
                                # Try as string for dates
                                return sem_key, value
            return None, None
        
        # Get target glyph's semantic data to find grouping key
        target_semantic = extract_semantic(target_glyph)
        target_temporal_key, _ = find_temporal_value(target_semantic)
        
        # Find a grouping key (like VIN, id, entity_id) to find related glyphs
        grouping_keys = ['vin', 'id', 'entity_id', 'key', 'identifier', 'name']
        grouping_key = None
        grouping_value = None
        
        for sem_key, value in target_semantic.items():
            key_parts = sem_key.lower().split('.')
            for gk in grouping_keys:
                if gk in key_parts or sem_key.lower().endswith(gk):
                    if value is not None and not isinstance(value, (dict, list)):
                        grouping_key = sem_key
                        grouping_value = str(value)
                        break
            if grouping_key:
                break
        
        # Collect all glyphs that share the same grouping value
        related_glyphs = []
        
        for g in glyphs:
            semantic = extract_semantic(g)
            
            # Check if this glyph shares the grouping value
            is_related = False
            if grouping_key and grouping_value:
                for sem_key, value in semantic.items():
                    if sem_key == grouping_key and str(value) == grouping_value:
                        is_related = True
                        break
            else:
                # No grouping key found, include all glyphs with temporal values
                is_related = True
            
            if is_related:
                temporal_key, temporal_value = find_temporal_value(semantic)
                if temporal_key and temporal_value is not None:
                    related_glyphs.append({
                        'glyph': g,
                        'semantic': semantic,
                        'temporal_key': temporal_key,
                        'temporal_value': temporal_value,
                    })
        
        # Sort by temporal value
        try:
            related_glyphs.sort(key=lambda x: float(x['temporal_value']) if isinstance(x['temporal_value'], (int, float, str)) else 0)
        except (ValueError, TypeError):
            # Sort as strings if numeric conversion fails
            related_glyphs.sort(key=lambda x: str(x['temporal_value']))
        
        # Build history with change deltas
        history: List[TemporalDataPoint] = []
        prev_values: Dict[str, Any] = {}
        
        for item in related_glyphs[:limit]:
            g = item['glyph']
            semantic = item['semantic']
            
            # Compute changes from previous point
            changes: Dict[str, Dict[str, Any]] = {}
            for key, value in semantic.items():
                if key in prev_values and prev_values[key] != value:
                    change_info: Dict[str, Any] = {
                        'from': prev_values[key],
                        'to': value,
                    }
                    # Compute numeric delta if possible
                    try:
                        from_val = float(prev_values[key])
                        to_val = float(value)
                        change_info['delta'] = to_val - from_val
                    except (ValueError, TypeError):
                        pass
                    changes[key] = change_info
            
            history.append(TemporalDataPoint(
                glyph_id=str(g.id),
                name=g.concept_text,
                label=extract_readable_label(g.concept_text),
                temporal_value=item['temporal_value'],
                temporal_key=item['temporal_key'],
                values=semantic,
                changes=changes,
            ))
            
            prev_values = semantic.copy()
        
        return TemporalHistoryResponse(
            glyph_id=glyph_id,
            history=history,
            temporal_key=target_temporal_key or '',
            total_points=len(history),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get temporal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
