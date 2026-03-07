"""
Natural Language Query Service.

Provides query translation and execution via stored procedures and similarity search.

Updated to support AutoSchemaMatcher for automatic schema-based NL query matching.
Updated to return SDK FactTree in all responses for unified response format.
Updated with response state pattern: DONE/ASK/BLOCKED/AUTH_REQUIRED/ERROR.

Design Principle: "When your LLM can't afford to be wrong, sidecar it with Glyphh"
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from glyphh.fact_tree.builder import FactTree

from domains.query.service import QueryService
from domains.query.fact_tree_builder import FactTreeBuilder

if TYPE_CHECKING:
    from domains.nl_query.schema_index import SchemaIndex
    from glyphh.nl.auto_schema_matcher import AutoSchemaMatcher, AutoMatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IntentMatch (lightweight result for translate_query)
# ---------------------------------------------------------------------------

@dataclass
class IntentMatch:
    """Result of matching a query against intent patterns."""
    intent: str
    confidence: float
    parameters: Dict[str, str]
    pattern_matched: Optional[str]
    structured_query: Dict[str, Any]
    match_method: str = "default"


# ---------------------------------------------------------------------------
# Response State Model
# ---------------------------------------------------------------------------

class ResponseState(str, Enum):
    """Response state indicating the outcome of an operation."""
    DONE = "DONE"
    ASK = "ASK"
    BLOCKED = "BLOCKED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    ERROR = "ERROR"


@dataclass
class AskPayload:
    """Payload for ASK state - missing slots or disambiguation needed."""
    question: str
    missing_slots: List[str] = field(default_factory=list)
    disambiguation_options: List[Dict[str, Any]] = field(default_factory=list)
    provided_slots: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "missing_slots": self.missing_slots,
            "disambiguation_options": self.disambiguation_options,
            "provided_slots": self.provided_slots,
        }


@dataclass
class BlockedPayload:
    """Payload for BLOCKED state - policy/permission prevented execution."""
    reason: str
    policy_refs: List[str] = field(default_factory=list)
    constraint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "policy_refs": self.policy_refs,
            "constraint": self.constraint,
        }


@dataclass
class AuthRequiredPayload:
    """Payload for AUTH_REQUIRED state - authentication needed."""
    provider: str
    app: str
    auth_hint: str = "Connect the required account to proceed."
    tool_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "app": self.app,
            "auth_hint": self.auth_hint,
            "tool_id": self.tool_id,
        }


@dataclass
class ErrorPayload:
    """Payload for ERROR state - unexpected failure."""
    error_code: str
    message: str
    debug_info: Optional[Dict[str, Any]] = None
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "error_code": self.error_code,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.debug_info:
            result["debug_info"] = self.debug_info
        return result


# ---------------------------------------------------------------------------
# NLQueryResult with Response State Pattern
# ---------------------------------------------------------------------------

@dataclass
class NLQueryResult:
    """
    Result of executing a natural language query.

    Uses the response state pattern:
    - DONE: Completed successfully; has fact_tree + trace_id
    - ASK: Missing slots / needs disambiguation; returns question + slots[]
    - BLOCKED: Policy/permission prevented execution; returns reason + policy_refs
    - AUTH_REQUIRED: Authentication needed; returns provider + app + auth_hint
    - ERROR: Unexpected failure; returns error_code + debug_info (safe)
    """
    state: ResponseState
    fact_tree: Optional[FactTree] = None
    query_type: str = ""
    match_method: str = ""  # "auto", "similarity", or "none"
    confidence: float = 0.0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    translated_query: Optional[Dict[str, Any]] = None
    query_time_ms: float = 0.0
    matched_route: Optional[str] = None

    # State-specific payloads
    ask: Optional[AskPayload] = None
    blocked: Optional[BlockedPayload] = None
    auth_required: Optional[AuthRequiredPayload] = None
    error: Optional[ErrorPayload] = None

    @property
    def disambiguation_needed(self) -> bool:
        """Backward compat: True when state is ASK with disambiguation options."""
        return (
            self.state == ResponseState.ASK
            and self.ask is not None
            and len(self.ask.disambiguation_options) > 0
        )

    @property
    def disambiguation_suggestions(self) -> List[str]:
        """Backward compat: Disambiguation suggestions as strings."""
        if self.ask and self.ask.disambiguation_options:
            return [
                opt.get("suggestion", opt.get("intent", str(opt)))
                for opt in self.ask.disambiguation_options
            ]
        return []

    @property
    def result(self) -> Any:
        """Backward compat: Returns fact_tree JSON for DONE, error dict otherwise."""
        if self.state == ResponseState.DONE and self.fact_tree is not None:
            return self.fact_tree.to_json()
        if self.state == ResponseState.ERROR and self.error is not None:
            return {"error": self.error.to_dict()}
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API response."""
        result = {
            "state": self.state.value,
            "confidence": self.confidence,
            "trace_id": self.trace_id,
            "query_type": self.query_type,
            "match_method": self.match_method,
            "query_time_ms": self.query_time_ms,
        }

        if self.matched_route:
            result["matched_route"] = self.matched_route

        if self.translated_query:
            result["translated_query"] = self.translated_query

        if self.state == ResponseState.DONE and self.fact_tree is not None:
            result["fact_tree"] = self.fact_tree.to_json()
        elif self.state == ResponseState.ASK and self.ask:
            result["ask"] = self.ask.to_dict()
        elif self.state == ResponseState.BLOCKED and self.blocked:
            result["blocked"] = self.blocked.to_dict()
        elif self.state == ResponseState.AUTH_REQUIRED and self.auth_required:
            result["auth_required"] = self.auth_required.to_dict()
        elif self.state == ResponseState.ERROR and self.error:
            result["error"] = self.error.to_dict()

        return result


# ---------------------------------------------------------------------------
# FactTree score helpers
# ---------------------------------------------------------------------------

def _extract_top_scores(fact_tree) -> list[float]:
    """Return similarity final_scores sorted descending from a FactTree."""
    try:
        ft_json = fact_tree.to_json() if hasattr(fact_tree, "to_json") else fact_tree
        for child in ft_json.get("children", []):
            if child.get("description") == "results":
                scores = []
                for match in child.get("children", []):
                    v = match.get("value") or {}
                    s = v.get("final_score")
                    if s is not None:
                        scores.append(float(s))
                return sorted(scores, reverse=True)
    except Exception:
        pass
    return []


def _extract_top_matches(fact_tree, n: int = 3) -> list[dict]:
    """Return top-n {concept_text, score} dicts from a FactTree."""
    try:
        ft_json = fact_tree.to_json() if hasattr(fact_tree, "to_json") else fact_tree
        for child in ft_json.get("children", []):
            if child.get("description") == "results":
                matches = []
                for match in child.get("children", []):
                    v = match.get("value") or {}
                    s = v.get("final_score")
                    if s is not None:
                        matches.append({
                            "concept_text": v.get("concept_text", ""),
                            "score": float(s),
                        })
                matches.sort(key=lambda x: x["score"], reverse=True)
                return matches[:n]
    except Exception:
        pass
    return []


def _extract_top_match_detail(fact_tree) -> Optional[dict]:
    """Return {glyph_id, concept_text, score, metadata} of the top-1 match."""
    try:
        ft_json = fact_tree.to_json() if hasattr(fact_tree, "to_json") else fact_tree
        for child in ft_json.get("children", []):
            if child.get("description") == "results":
                matches = child.get("children", [])
                if matches:
                    v = matches[0].get("value") or {}
                    return {
                        "glyph_id": v.get("glyph_id"),
                        "concept_text": v.get("concept_text"),
                        "score": v.get("final_score"),
                        "metadata": v.get("metadata") or {},
                    }
    except Exception:
        pass
    return None


class NLQueryService:
    """
    Natural Language Query Service.
    
    Routes NL queries to the appropriate operation:
    - AutoSchemaMatcher for schema-based matching (when available)
    - Direct similarity search as the default path
    
    The legacy IntentMatcher has been removed — each model's own encoder
    handles NL→query translation via encode_query_fn.
    
    Validates: Requirements 3, 4, 5, 12.2, 12.5
    """
    
    def __init__(
        self,
        query_service: QueryService,
        confidence_threshold: float = 0.85,
        schema_index: Optional['SchemaIndex'] = None,
        auto_schema_matcher: Optional['AutoSchemaMatcher'] = None,
        assess_query_fn: Optional[Any] = None,
        min_gap: float = 0.03,
        two_stage: bool = False,
        cognitive_loop_enabled: bool = False,
        cognitive_loop_config: Optional[Dict] = None,
    ):
        """
        Initialize the NL Query Service.

        Args:
            query_service: QueryService for executing structured queries
            confidence_threshold: Minimum confidence threshold
            schema_index: Optional SchemaIndex for auto-schema matching
            auto_schema_matcher: Optional AutoSchemaMatcher for auto-schema matching
            assess_query_fn: Optional callable(query: str) -> dict from the model's
                             encoder.py.  Returns {complete, missing, reason, ...}.
                             When provided, incomplete queries return ASK before
                             the similarity search runs.
            min_gap: Minimum score gap between top-1 and top-2 similarity results
                     for a DONE response.  Queries where all top results cluster
                     within this band return ASK for disambiguation.
            two_stage: If True, the model uses two-stage queries (exemplar match →
                       GQL procedure).  Gap analysis is skipped because stage 1 is
                       just routing — stage 2 results are the final answer.
            cognitive_loop_enabled: If True, route queries through CognitiveLoop
                                   after similarity search (adds memory, slot
                                   extraction, confidence blending).
            cognitive_loop_config: Config dict for CognitiveLoop (dimension,
                                  confidence_threshold, etc.).
        """
        self.query_service = query_service
        self.confidence_threshold = confidence_threshold
        self._schema_index = schema_index
        self._auto_schema_matcher = auto_schema_matcher
        self._assess_query_fn = assess_query_fn
        self._min_gap = min_gap
        self._two_stage = two_stage
        self._cognitive_loop_enabled = cognitive_loop_enabled
        self._cognitive_loop_config = cognitive_loop_config or {}
        # Cache for Stage 2 GQL storage per model — avoids reloading all
        # glyph embeddings on every request.  Key: (org_id, model_id).
        self._gql_storage_cache: Dict[tuple, tuple] = {}  # (org, model) → (storage, timestamp)
        import asyncio
        self._gql_cache_lock = asyncio.Lock()
    
    def set_schema_index(self, schema_index: 'SchemaIndex') -> None:
        """
        Set the schema index for auto-schema matching.
        
        Args:
            schema_index: The SchemaIndex to use for matching
        
        Validates: Requirement 7
        """
        self._schema_index = schema_index
        logger.info(f"Schema index set for model '{schema_index.model_id}'")
    
    def set_auto_schema_matcher(self, matcher: 'AutoSchemaMatcher') -> None:
        """
        Set the AutoSchemaMatcher for auto-schema matching.
        
        Args:
            matcher: The AutoSchemaMatcher to use for matching
        
        Validates: Requirements 3, 4, 5
        """
        self._auto_schema_matcher = matcher
        logger.info("AutoSchemaMatcher set for NL query service")
    
    def _resolve_model_paths(self, model_path: str) -> list:
        """Return candidate model directories (model_path + dev model dir)."""
        import os
        from pathlib import Path
        paths = [Path(model_path)]
        # Dev mode: GLYPHH_DEV_MODEL_DIR points to the actual source directory
        # (model_path may be a temp extraction dir from .glyphh package)
        dev_dir = os.environ.get("GLYPHH_DEV_MODEL_DIR", "").strip()
        if dev_dir:
            paths.append(Path(dev_dir))
        return paths

    def _load_gql_procedures(self, model_path: str) -> dict:
        """Load procedure registry from model's gql.json."""
        import json
        for path in self._resolve_model_paths(model_path):
            gql_path = path / "gql.json"
            if gql_path.exists():
                try:
                    return json.loads(gql_path.read_text())
                except Exception:
                    pass
        return {}

    def _resolve_procedure_id(self, proc_id: str, procedures: dict) -> Optional[str]:
        """If proc_id is a key in the procedure registry, return its GQL template."""
        proc = procedures.get(proc_id)
        if proc and isinstance(proc, dict):
            return proc.get("gql")
        return None

    async def _load_source_files_from_db(
        self, org_id: str, model_id: str,
    ) -> Optional[dict]:
        """Load source_files from model_configs DB table (fallback for no-disk)."""
        try:
            from domains.models.db_models import ModelConfig
            from sqlalchemy import select
            mgr = self.query_service._model_manager
            async with mgr._session_factory() as session:
                result = await session.execute(
                    select(ModelConfig.source_files).where(
                        ModelConfig.org_id == org_id,
                        ModelConfig.model_id == model_id,
                    )
                )
                sf = result.scalar_one_or_none()
            return sf if isinstance(sf, dict) else None
        except Exception:
            return None

    async def _resolve_gql_template(
        self,
        org_id: str,
        model_id: str,
        matched_metadata: dict,
    ) -> Optional[str]:
        """Resolve GQL template from exemplar metadata, gql.json, or config.yaml.

        Resolution order:
          1. Per-exemplar inline gql_query
          2. Per-exemplar gql_id → gql.json procedure
          3. config.yaml gql_query_default → gql.json procedure or inline GQL

        Falls back to DB source_files when files aren't on disk (Heroku).
        """
        # 1. Per-exemplar inline gql_query
        gql = matched_metadata.get("gql_query")
        if gql:
            return gql

        # Load model for path access
        try:
            loaded_model = await self.query_service._model_manager.get_model(org_id, model_id)
        except Exception:
            return None

        model_path = getattr(loaded_model, "model_path", None) if loaded_model else None
        procedures = self._load_gql_procedures(model_path) if model_path else {}

        # Fallback: load gql.json from DB source_files
        if not procedures:
            source_files = await self._load_source_files_from_db(org_id, model_id)
            if source_files and "gql.json" in source_files:
                try:
                    import json
                    procedures = json.loads(source_files["gql.json"])
                except Exception:
                    pass

        logger.debug(
            f"GQL resolution for {org_id}/{model_id}: "
            f"model_path={model_path}, procedures={list(procedures.keys())}"
        )

        # 2. Per-exemplar gql_id → gql.json
        gql_id = matched_metadata.get("gql_id")
        if gql_id:
            resolved = self._resolve_procedure_id(gql_id, procedures)
            if resolved:
                return resolved

        # 3. config.yaml gql_query_default → procedure ID or inline GQL
        config_raw = None
        try:
            import yaml
            if model_path:
                for path in self._resolve_model_paths(model_path):
                    config_path = path / "config.yaml"
                    if config_path.exists():
                        with open(config_path) as f:
                            config_raw = yaml.safe_load(f) or {}
                        break
            # Fallback: config.yaml from DB source_files
            if config_raw is None:
                source_files = await self._load_source_files_from_db(org_id, model_id)
                if source_files and "config.yaml" in source_files:
                    config_raw = yaml.safe_load(source_files["config.yaml"]) or {}
            if config_raw:
                default = config_raw.get("gql_query_default")
                if default:
                    resolved = self._resolve_procedure_id(default, procedures)
                    if resolved:
                        return resolved
                    return default
        except Exception:
            pass

        return None

    @staticmethod
    def _fill_slots(
        template: str,
        matched_id: str,
        matched_metadata: dict,
        query_text: str,
    ) -> str:
        """Fill {slot} placeholders in a GQL template.

        Slot sources:
          {matched_id}       → matched exemplar glyph UUID
          {query}            → original NL query text
          {exm.attribute}    → matched exemplar metadata field
        """
        import re

        # Built-in slots
        template = template.replace("{matched_id}", matched_id)
        template = template.replace("{query}", query_text)

        # Exemplar attribute slots: {exm.field_name} → metadata[field_name]
        def _resolve_exm(match):
            attr = match.group(1)
            val = matched_metadata.get(attr)
            if val is not None and isinstance(val, (str, int, float)):
                return str(val)
            return match.group(0)  # leave unresolved if not found

        template = re.sub(r"\{exm\.([a-zA-Z_][a-zA-Z0-9_]*)\}", _resolve_exm, template)

        return template

    async def _get_or_build_gql_storage(
        self, org_id: str, model_id: str,
    ):
        """Get or build a cached GQL storage for Stage 2 queries.

        Caches glyph embeddings per model to avoid reloading 22K+ vectors
        on every request.  Uses an async lock to prevent thundering herd —
        only one request builds the cache, others wait.
        Cache is invalidated after 5 minutes.
        """
        import time as _time
        from domains.gql.storage import DatabaseGlyphStorage
        from domains.models.storage import GlyphStorage
        from shared.similarity_service import SimilarityService

        cache_key = (org_id, model_id)

        # Fast path: check cache without lock
        cached = self._gql_storage_cache.get(cache_key)
        if cached is not None:
            storage, ts = cached
            if _time.time() - ts < 300:  # 5 min TTL
                return storage

        # Slow path: acquire lock, build cache once
        async with self._gql_cache_lock:
            # Re-check after acquiring lock (another request may have built it)
            cached = self._gql_storage_cache.get(cache_key)
            if cached is not None:
                storage, ts = cached
                if _time.time() - ts < 300:
                    return storage

            loaded_model = await self.query_service._model_manager.get_model(org_id, model_id)
            if loaded_model is None:
                raise ValueError(f"Model {org_id}/{model_id} not loaded")

            async with self.query_service._session_factory() as session:
                storage_db = GlyphStorage(session)

                all_glyphs, all_embeddings = await storage_db.list_glyphs_with_embeddings(
                    org_id=org_id,
                    model_id=model_id,
                    limit=50000,
                )

                # Include all glyphs so glyph("uuid") references resolve
                # (Stage 2 GQL may reference the matched exemplar's vector).
                # Pattern glyphs are filtered from results by build_two_stage_result.
                db_glyphs = all_glyphs

                embeddings = {
                    str(g.id): all_embeddings[str(g.id)]
                    for g in db_glyphs
                    if str(g.id) in all_embeddings
                }

                glyph_ids = [g.id for g in db_glyphs]
                hierarchical = await storage_db.get_hierarchical_embeddings(
                    org_id=org_id,
                    model_id=model_id,
                    glyph_ids=glyph_ids,
                )

            similarity_service = SimilarityService(
                similarity_calculator=getattr(loaded_model, 'similarity_calculator', None),
            )

            gql_storage = DatabaseGlyphStorage(
                org_id=org_id,
                model_id=model_id,
                glyphs=db_glyphs,
                embeddings=embeddings,
                similarity_service=similarity_service,
                hierarchical_embeddings=hierarchical,
            )

            self._gql_storage_cache[cache_key] = (gql_storage, _time.time())
            logger.info(f"Built GQL storage cache for {org_id}/{model_id}: "
                         f"{len(db_glyphs)} glyphs, {len(embeddings)} embeddings")
            return gql_storage

    async def _execute_gql(
        self,
        org_id: str,
        model_id: str,
        gql_query: str,
    ) -> FactTree:
        """Execute a GQL query string through the GQL executor engine.

        Uses cached glyph storage to avoid reloading embeddings per request.
        """
        from glyphh.gql import GQLExecutor, ExecutionContext

        loaded_model = await self.query_service._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ValueError(f"Model {org_id}/{model_id} not loaded")

        gql_storage = await self._get_or_build_gql_storage(org_id, model_id)

        context = ExecutionContext(
            model=loaded_model.sdk_model,
            storage=gql_storage,
            encoder=getattr(loaded_model, 'encoder', None),
        )

        executor = GQLExecutor(context=context, enable_cache=False)
        return executor.execute(gql_query)

    async def execute_nl_query(
        self,
        org_id: str,
        model_id: str,
        query: str,
        debug: bool = False,
        stage: str = "auto",
        confirmed: bool = False,
    ) -> NLQueryResult:
        """
        Execute a natural language query.

        Stage modes:
        - "auto" (default): Full two-stage pipeline (NL → exemplar → GQL → data)
        - "patterns": Stage 1 only — search exemplar patterns, skip Stage 2 GQL
        - "data": Direct data search — skip exemplar matching, search data records

        Flow:
        1. Try auto-schema matching if AutoSchemaMatcher is available
        2. Default to similarity search (model's encode_query_fn handles NL)
        2b. If matched exemplar has gql_query — execute Stage 2 reference search
        3. If encoding fails: ERROR
        """
        start_time = time.time()

        logger.info(f"NL query received: '{query}' for org={org_id}, model={model_id}, stage={stage}")

        # Stage "data" — bypass exemplar matching, search data records directly
        if stage == "data":
            return await self._execute_data_stage(org_id, model_id, query, debug, start_time)

        # Step 0: Try auto-schema matching if available
        if self._auto_schema_matcher is not None:
            try:
                auto_result = await self._execute_auto_schema_query(
                    org_id=org_id,
                    model_id=model_id,
                    query=query,
                    debug=debug,
                    start_time=start_time,
                )
                if auto_result is not None:
                    return auto_result
            except Exception as e:
                logger.warning(f"Auto-schema matching failed: {e}, falling back to similarity search")

        # Step 0b: Semantic slot check — fast, runs before HDC encoding.
        # The model's assess_query_fn returns {complete, missing, reason, ...}.
        # An incomplete query (both action and domain unresolved) returns ASK
        # immediately so the user can refine before we spend time on similarity.
        if self._assess_query_fn is not None:
            try:
                assessment = self._assess_query_fn(query)
                if not assessment.get("complete", True):
                    elapsed_ms = (time.time() - start_time) * 1000
                    missing = assessment.get("missing", [])
                    reason  = assessment.get("reason", "Please clarify your request.")
                    logger.info(
                        f"Query incomplete — missing slots {missing}: '{query}'"
                    )
                    return NLQueryResult(
                        state=ResponseState.ASK,
                        query_type="similarity_search",
                        match_method="direct",
                        confidence=0.0,
                        query_time_ms=elapsed_ms,
                        ask=AskPayload(
                            question=reason,
                            missing_slots=missing,
                        ),
                    )
            except Exception as e:
                logger.warning(f"assess_query_fn failed: {e}, continuing")

        # Step 1: Direct similarity search — the model's encode_query_fn
        # handles NL→embedding translation
        try:
            fact_tree = await self._execute_structured_query(
                org_id,
                model_id,
                "similarity_search",
                {"query": query, "top_k": 10},
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Cognitive loop path — if enabled, route through CognitiveLoop
            # instead of the default gap analysis / threshold / GQL pipeline.
            if self._cognitive_loop_enabled:
                top_matches = _extract_top_matches(fact_tree, n=5)
                return self._execute_cognitive_loop(
                    org_id, model_id, query, fact_tree, top_matches,
                    start_time, debug,
                )

            # Step 1b: Gap analysis — if top results cluster within min_gap,
            # the query is ambiguous.  Return ASK with the top candidates.
            # Skip when confirmed=True (user already picked from disambiguation).
            # Skip for two-stage models — stage 1 is routing, stage 2 produces
            # the final list.  Clustered exemplar scores are expected.
            top_scores = _extract_top_scores(fact_tree)
            if not self._two_stage and not confirmed and len(top_scores) >= 2 and (top_scores[0] - top_scores[1]) < self._min_gap:
                top_matches = _extract_top_matches(fact_tree, n=3)
                logger.info(
                    f"Gap too small ({top_scores[0]:.3f} vs {top_scores[1]:.3f}) "
                    f"for query: '{query}'"
                )
                return NLQueryResult(
                    state=ResponseState.ASK,
                    query_type="similarity_search",
                    match_method="direct",
                    confidence=top_scores[0],
                    query_time_ms=elapsed_ms,
                    ask=AskPayload(
                        question="Your query matches multiple options. Did you mean one of these?",
                        disambiguation_options=[
                            {
                                "intent": m["concept_text"],
                                "confidence": m["score"],
                                "suggestion": f"{m['concept_text']} ({m['score']:.0%} match)",
                            }
                            for m in top_matches
                        ],
                    ),
                )

            # Step 1c: Confidence threshold — if the best match is below the
            # model's similarity threshold, no exemplar matched confidently.
            # Return ASK so the user can refine their query.
            # Skip when confirmed=True (user already picked from disambiguation).
            if not confirmed and top_scores and top_scores[0] < self.confidence_threshold:
                top_matches = _extract_top_matches(fact_tree, n=3)
                logger.info(
                    f"Best score {top_scores[0]:.3f} below threshold "
                    f"{self.confidence_threshold:.3f} for query: '{query}'"
                )
                return NLQueryResult(
                    state=ResponseState.ASK,
                    query_type="similarity_search",
                    match_method="direct",
                    confidence=top_scores[0],
                    query_time_ms=elapsed_ms,
                    ask=AskPayload(
                        question="No confident match found. Can you be more specific?",
                        disambiguation_options=[
                            {
                                "intent": m["concept_text"],
                                "confidence": m["score"],
                                "suggestion": f"{m['concept_text']} ({m['score']:.0%} match)",
                            }
                            for m in top_matches
                        ] if top_matches else [],
                    ),
                )

            # Step 2: Two-stage GQL execution (if model defines gql_query)
            # Skip Stage 2 when:
            #   - stage="patterns" (caller only wants exemplar match)
            #   - matched exemplar is a pattern and has no per-exemplar gql_query
            #     (pattern-only models like Pipedream — Stage 2 would just re-find
            #     the same pattern, wasting memory and time)
            match_detail = _extract_top_match_detail(fact_tree)
            result_tree = fact_tree
            match_method = "direct"

            if stage != "patterns" and match_detail and match_detail.get("glyph_id"):
                match_meta = match_detail.get("metadata", {})
                glyph_id = match_detail["glyph_id"]

                # Skip Stage 2 for pattern records that have no per-exemplar
                # GQL query.  Pattern records ARE the final answer — running
                # the config-level gql_query_default against them would load
                # every glyph into memory for no benefit.
                is_pattern = match_meta.get("record_type") == "pattern"
                has_own_gql = bool(
                    match_meta.get("gql_query") or match_meta.get("gql_id")
                )

                gql_template = None
                if not is_pattern or has_own_gql:
                    gql_template = await self._resolve_gql_template(
                        org_id, model_id, match_meta,
                    )

                if gql_template:
                    try:
                        resolved_gql = self._fill_slots(
                            gql_template, glyph_id, match_meta, query,
                        )
                        stage2_tree = await self._execute_gql(
                            org_id, model_id, resolved_gql,
                        )
                        result_tree = FactTreeBuilder.build_two_stage_result(
                            exemplar_match=match_detail,
                            data_results=stage2_tree,
                            gql_query=resolved_gql,
                            total_query_time_ms=(time.time() - start_time) * 1000,
                        )
                        match_method = "two_stage_gql"
                        logger.info(
                            f"Two-stage query: exemplar={match_detail['concept_text']!r}, "
                            f"gql={resolved_gql}"
                        )
                    except Exception as e:
                        logger.warning(f"Stage 2 GQL failed: {e}, returning Stage 1 only")
                else:
                    # No GQL procedure defined — return exemplar match directly
                    result_tree = FactTreeBuilder.build_two_stage_result(
                        exemplar_match=match_detail,
                        data_results=None,
                        gql_query=None,
                        total_query_time_ms=(time.time() - start_time) * 1000,
                    )
                    match_method = "two_stage_gql"

            elapsed_ms = (time.time() - start_time) * 1000

            return NLQueryResult(
                state=ResponseState.DONE,
                fact_tree=result_tree,
                query_type="similarity_search",
                match_method=match_method,
                confidence=top_scores[0] if top_scores else 1.0,
                translated_query={"operation": "similarity_search", "query": query} if debug else None,
                query_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")

        # Step 2: No match -> ERROR
        logger.info(f"No match for query: '{query}'")

        elapsed_ms = (time.time() - start_time) * 1000

        return NLQueryResult(
            state=ResponseState.ERROR,
            query_type="unknown",
            match_method="none",
            confidence=0.0,
            query_time_ms=elapsed_ms,
            error=ErrorPayload(
                error_code="NO_MATCH",
                message="Could not process query",
                debug_info={"query": query},
                recoverable=True,
            ),
        )
    
    # ------------------------------------------------------------------
    # Cognitive Loop integration
    # ------------------------------------------------------------------

    # Module-level cache: persistent across requests for episodic memory
    _cognitive_loop_cache: Dict[tuple, Any] = {}

    @staticmethod
    def confirm_last(
        org_id: str,
        model_id: str,
        was_correct: bool,
        correct_action: str | None = None,
    ) -> dict:
        """Confirm or correct the last cognitive loop step.

        Called from the ``confirm`` MCP tool.  Operates on the cached
        CognitiveLoop for (org_id, model_id).

        - was_correct=True  → Hebbian reinforcement (strengthen recalled idea)
        - was_correct=False → store correction as new idea for future recall
        """
        key = (org_id, model_id)
        loop = NLQueryService._cognitive_loop_cache.get(key)
        if loop is None:
            return {
                "state": "ERROR",
                "error": (
                    f"No CognitiveLoop for {org_id}/{model_id}. "
                    "Send an nl_query first."
                ),
            }

        try:
            correct_outcome = None
            if not was_correct and correct_action:
                correct_outcome = [{correct_action: {}}]

            loop.confirm(was_correct, correct_outcome)

            ideas_count = loop.idea_space.size
            return {
                "state": "DONE",
                "confirmed": was_correct,
                "correct_action": correct_action,
                "ideas_stored": ideas_count,
                "match_method": "cognitive_loop",
                "confidence": 1.0,
            }
        except Exception as e:
            logger.error(f"confirm_last error: {e}", exc_info=True)
            return {"state": "ERROR", "error": str(e)}

    def _execute_cognitive_loop(
        self,
        org_id: str,
        model_id: str,
        query: str,
        similarity_tree: Any,
        top_matches: list[dict],
        start_time: float,
        debug: bool,
    ) -> NLQueryResult:
        """Route query through CognitiveLoop using pre-computed similarity.

        Safety: even when the loop says CALL, apply gap analysis as a
        secondary check.  If the similarity gap is too narrow, override
        the loop's decision and return ASK instead.
        """
        from shared.precomputed_scorer import PrecomputedScorer
        from glyphh.cognitive import CognitiveLoop

        scorer = PrecomputedScorer(top_matches)
        loop = self._get_or_create_loop(org_id, model_id, scorer)

        # Stateless MCP queries: reset per-query context so idea vectors
        # are consistent across calls and temporal decay doesn't kill
        # recall across independent queries.
        loop._recent_actions = []
        loop._turn = 0
        loop.idea_space._turn = 0

        step_result = loop.step(query)

        # Safety gate: if loop says CALL but similarity gap is too narrow,
        # override to ASK — UNLESS episodic memory confirms the top-1.
        # "Confirms" = recalled idea labels the same function as current
        # top-1 (raw cosine > 0.25 already gated by IdeaSpace.recall).
        # Skip for two-stage models — stage 1 is routing, not disambiguation.
        passed_gap_gate = True
        if not self._two_stage and step_result.action == "CALL" and len(top_matches) >= 2:
            gap = top_matches[0]["score"] - top_matches[1]["score"]
            if gap < self._min_gap:
                recall_confirms = False
                recall_signal = step_result.signals.get("recall", [])
                if recall_signal:
                    recalled_label = recall_signal[0][0]
                    current_top = top_matches[0]["concept_text"]
                    recall_confirms = (recalled_label == current_top)
                if not recall_confirms:
                    passed_gap_gate = False
                    logger.info(
                        f"Cognitive loop CALL overridden to ASK: "
                        f"gap {gap:.3f} < min_gap {self._min_gap:.3f}"
                    )
                    step_result = type(step_result)(
                        action="ASK",
                        missing=["Narrow gap between top matches"],
                        confidence=step_result.confidence,
                        signals=step_result.signals,
                    )

        # Auto-store successful CALLs that passed the gap gate as ideas.
        # This enables future recall for similar queries.
        # NOTE: We do NOT auto-confirm here — the LLM should call the
        # `confirm` MCP tool after verifying the result was correct.
        if step_result.action == "CALL" and passed_gap_gate:
            try:
                idea_vec = getattr(loop, '_last_step_idea_vec', None)
                if idea_vec is not None:
                    outcome = step_result.calls if step_result.calls else []
                    label = (
                        list(outcome[0].keys())[0]
                        if outcome else "unknown"
                    )
                    loop.idea_space.store(
                        idea_vec, outcome, strength=1.0, label=label,
                    )
            except Exception:
                pass

        elapsed_ms = (time.time() - start_time) * 1000
        return self._step_result_to_nl_result(
            step_result, similarity_tree, top_matches, elapsed_ms, debug,
        )

    def _get_or_create_loop(
        self,
        org_id: str,
        model_id: str,
        scorer: Any,
    ) -> Any:
        """Get cached CognitiveLoop or create a new one."""
        from glyphh.cognitive import CognitiveLoop

        key = (org_id, model_id)
        if key not in NLQueryService._cognitive_loop_cache:
            cfg = self._cognitive_loop_config
            loop = CognitiveLoop(
                dimension=cfg.get("dimension", 2000),
                confidence_threshold=cfg.get("confidence_threshold", 0.25),
                model_scorer=scorer,
            )
            # Build function definitions from model's stored exemplars
            func_defs = self._build_func_defs_sync(org_id, model_id)
            loop.begin(functions=func_defs)
            NLQueryService._cognitive_loop_cache[key] = loop
            logger.info(
                f"CognitiveLoop created for {org_id}/{model_id} "
                f"with {len(func_defs)} functions"
            )
        else:
            loop = NLQueryService._cognitive_loop_cache[key]
            # Update scorer with fresh pre-computed results for this query
            if hasattr(loop, '_classifier') and loop._classifier is not None:
                loop._classifier._scorer = scorer

        return loop

    def _build_func_defs_sync(self, org_id: str, model_id: str) -> list[dict]:
        """Build function definitions from model manager's loaded model."""
        import asyncio

        async def _load():
            model = await self.query_service._model_manager.get_model(org_id, model_id)
            if not model or not model.model_path:
                return []
            from pathlib import Path
            import json
            exemplars_path = Path(model.model_path) / "data" / "exemplars.jsonl"
            if not exemplars_path.exists():
                return []
            func_defs = []
            seen: set = set()
            with open(exemplars_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    ak = entry.get("action_key", "") or entry.get("name", "")
                    if ak and ak not in seen:
                        seen.add(ak)
                        func_defs.append({
                            "name": ak,
                            "description": entry.get("action_name", ak),
                            "parameters": entry.get("configured_props", {}),
                        })
            return func_defs

        # We're called from a sync context inside an async event loop.
        # Use the running loop to schedule the coroutine.
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _load())
                return future.result(timeout=10)
        except Exception as e:
            logger.warning(f"Failed to load func_defs: {e}")
            return []

    def _step_result_to_nl_result(
        self,
        step: Any,
        similarity_tree: Any,
        top_matches: list[dict],
        elapsed_ms: float,
        debug: bool,
    ) -> NLQueryResult:
        """Convert CognitiveLoop StepResult → NLQueryResult."""
        if step.action == "CALL":
            # Build DONE response
            func_name = list(step.calls[0].keys())[0] if step.calls else None
            match_detail = _extract_top_match_detail(similarity_tree)

            # If cognitive loop resolved to a different function than top-1,
            # try to find it in the similarity tree
            if func_name and match_detail and match_detail.get("concept_text") != func_name:
                for m in top_matches:
                    if m["concept_text"] == func_name:
                        match_detail = {
                            "concept_text": func_name,
                            "score": m["score"],
                            "glyph_id": m.get("glyph_id"),
                            "metadata": m.get("metadata", {}),
                        }
                        break

            result_tree = FactTreeBuilder.build_two_stage_result(
                exemplar_match=match_detail,
                data_results=None,
                gql_query=None,
                total_query_time_ms=elapsed_ms,
            )
            return NLQueryResult(
                state=ResponseState.DONE,
                fact_tree=result_tree,
                query_type="similarity_search",
                match_method="cognitive_loop",
                confidence=step.confidence,
                query_time_ms=elapsed_ms,
                translated_query={"signals": step.signals} if debug else None,
            )
        else:
            # ASK response
            disambiguation = []
            # Use top similarity matches as disambiguation options
            for m in top_matches[:3]:
                disambiguation.append({
                    "intent": m["concept_text"],
                    "confidence": m["score"],
                    "suggestion": f"{m['concept_text']} ({m['score']:.0%} match)",
                })

            return NLQueryResult(
                state=ResponseState.ASK,
                query_type="similarity_search",
                match_method="cognitive_loop",
                confidence=step.confidence,
                query_time_ms=elapsed_ms,
                ask=AskPayload(
                    question="Could not confidently match your request.",
                    missing_slots=step.missing if hasattr(step, 'missing') else [],
                    disambiguation_options=disambiguation,
                ),
                translated_query={"signals": step.signals} if debug else None,
            )

    async def _execute_data_stage(
        self,
        org_id: str,
        model_id: str,
        query: str,
        debug: bool,
        start_time: float,
    ) -> NLQueryResult:
        """Execute a direct data search — skip exemplar matching.

        Searches data records only (excludes record_type=pattern) by routing
        through _execute_gql which already filters out patterns.
        Uses a FIND SIMILAR TO "query" GQL query against data glyphs.
        """
        try:
            # Escape quotes in query for GQL string literal
            safe_query = query.replace('"', '\\"')
            gql_query = f'FIND SIMILAR TO "{safe_query}" LIMIT 10'

            result_tree = await self._execute_gql(org_id, model_id, gql_query)
            elapsed_ms = (time.time() - start_time) * 1000

            return NLQueryResult(
                state=ResponseState.DONE,
                fact_tree=result_tree,
                query_type="similarity_search",
                match_method="data_direct",
                confidence=1.0,
                query_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.warning(f"Data stage search failed: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return NLQueryResult(
                state=ResponseState.ERROR,
                query_type="unknown",
                match_method="none",
                query_time_ms=elapsed_ms,
                error=ErrorPayload(
                    error_code="DATA_SEARCH_FAILED",
                    message=str(e),
                    recoverable=True,
                ),
            )

    async def _execute_auto_schema_query(
        self,
        org_id: str,
        model_id: str,
        query: str,
        debug: bool,
        start_time: float,
    ) -> Optional[NLQueryResult]:
        """
        Execute a query using auto-schema matching.

        Returns NLQueryResult with appropriate state:
        - DONE: Match found and executed
        - ASK: Disambiguation needed
        - None: Confidence too low, fall through to rules
        """
        if self._auto_schema_matcher is None:
            return None

        # Check cache if schema index is available
        cache_key = None
        if self._schema_index is not None:
            cache_key = self._schema_index.compute_query_hash(query)
            cached_match = self._schema_index.get_cached_result(cache_key)
            if cached_match is not None:
                logger.info(f"Cache hit for query: '{query}'")

        # Use AutoSchemaMatcher to match the query
        auto_result = self._auto_schema_matcher.match_query(query)

        # Cache the match result if schema index is available
        if self._schema_index is not None and cache_key is not None:
            self._schema_index.cache_match_result(cache_key, auto_result.match_result)

        # Check if disambiguation is needed -> ASK
        if auto_result.disambiguation_needed:
            logger.info(
                f"Disambiguation needed for query: '{query}', "
                f"options: {[opt.intent_type for opt in auto_result.disambiguation_options]}"
            )

            options = [
                {
                    "intent": opt.intent_type,
                    "confidence": opt.confidence,
                    "suggestion": f"Did you mean '{opt.intent_type}'? (confidence: {opt.confidence:.0%})",
                }
                for opt in auto_result.disambiguation_options
            ]

            elapsed_ms = (time.time() - start_time) * 1000

            return NLQueryResult(
                state=ResponseState.ASK,
                query_type=auto_result.intent.intent_type,
                match_method=auto_result.match_method,
                confidence=auto_result.confidence,
                translated_query=auto_result.to_dict() if debug else None,
                query_time_ms=elapsed_ms,
                ask=AskPayload(
                    question="Your query is ambiguous. Please clarify:",
                    disambiguation_options=options,
                ),
            )

        # Check confidence threshold
        min_acceptable_confidence = 0.3
        if auto_result.confidence < min_acceptable_confidence:
            logger.info(
                f"Auto-schema confidence too low: {auto_result.confidence:.3f}, "
                f"falling back to rules"
            )
            return None

        logger.info(
            f"Auto-schema match: intent={auto_result.intent.intent_type}, "
            f"confidence={auto_result.confidence:.3f}, "
            f"method={auto_result.match_method}"
        )

        # Map auto-schema intent to operation
        operation = self._map_intent_to_operation(auto_result.intent.intent_type)

        # Build structured query from extracted parameters
        structured_query = self._build_structured_query_from_auto_result(
            auto_result, operation, query
        )

        # Execute the query -> DONE
        fact_tree = await self._execute_structured_query(
            org_id,
            model_id,
            operation,
            structured_query,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return NLQueryResult(
            state=ResponseState.DONE,
            fact_tree=fact_tree,
            query_type=auto_result.intent.intent_type,
            match_method=auto_result.match_method,
            confidence=auto_result.confidence,
            translated_query=auto_result.to_dict() if debug else None,
            query_time_ms=elapsed_ms,
        )
    
    def _map_intent_to_operation(self, intent_type: str) -> str:
        """
        Map auto-schema intent type to query operation.
        
        Args:
            intent_type: The intent type from auto-schema matching
        
        Returns:
            The corresponding query operation name
        """
        intent_to_operation = {
            "find": "similarity_search",
            "count": "count",
            "filter": "similarity_search",  # Filter uses similarity with constraints
            "similar": "similarity_search",
            "compare": "compare",
            "predict": "temporal_predict",
            "verify": "fact_tree",
        }
        return intent_to_operation.get(intent_type, "similarity_search")
    
    def _build_structured_query_from_auto_result(
        self,
        auto_result: 'AutoMatchResult',
        operation: str,
        original_query: str,
    ) -> Dict[str, Any]:
        """
        Build a structured query from auto-match result.
        
        Converts the extracted parameters from auto-schema matching
        into a structured query format for execution.
        
        Args:
            auto_result: The AutoMatchResult with extracted parameters
            operation: The query operation to perform
            original_query: The original query string
        
        Returns:
            Structured query dictionary
        
        Validates: Requirement 5
        """
        structured_query = {
            "operation": operation,
            "query": original_query,
        }
        
        # Add extracted parameters
        for role, param in auto_result.parameters.parameters.items():
            structured_query[role] = param.value
        
        # Add multi-value parameters
        for role, params in auto_result.parameters.multi_value_params.items():
            structured_query[role] = [p.value for p in params]
        
        # Add operation-specific defaults
        if operation == "similarity_search":
            structured_query.setdefault("top_k", 10)
        elif operation == "count":
            pass  # No additional params needed
        elif operation == "temporal_predict":
            structured_query.setdefault("steps_ahead", 1)
            structured_query.setdefault("beam_width", 3)
        elif operation == "fact_tree":
            structured_query.setdefault("max_depth", 3)
        
        return structured_query
    
    async def translate_query(
        self,
        query: str
    ) -> Tuple[Optional[IntentMatch], str]:
        """
        Translate a query without executing it.
        
        Useful for debugging and testing query translation.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (IntentMatch or None, match_method)
        """
        # Try auto-schema matching first if available
        if self._auto_schema_matcher is not None:
            try:
                auto_result = self._auto_schema_matcher.match_query(query)
                if auto_result.confidence >= 0.3:
                    return IntentMatch(
                        intent=auto_result.intent.intent_type,
                        confidence=auto_result.confidence,
                        parameters=auto_result.parameters.to_dict().get("parameters", {}),
                        pattern_matched=None,
                        structured_query=auto_result.to_dict(),
                    ), auto_result.match_method
            except Exception as e:
                logger.warning(f"Auto-schema translation failed: {e}")
        
        # Default: similarity search
        return IntentMatch(
            intent="similarity_search",
            confidence=1.0,
            parameters={},
            pattern_matched=None,
            structured_query={"operation": "similarity_search", "query": query},
        ), "direct"
    
    def _normalize_operation(self, operation: str) -> str:
        """
        Normalize operation names from intent matcher to canonical executor names.
        
        This prevents mismatches between intent names (e.g., 'list_all') and
        executor operation names (e.g., 'list').
        
        Args:
            operation: Raw operation name from intent matcher
            
        Returns:
            Canonical operation name for the executor
        """
        # Map intent matcher names to canonical executor names
        operation_aliases = {
            # List operations
            "list_all": "list",
            "list_everything": "list",
            "show_all": "list",
            "enumerate": "list",
            # Search operations
            "find": "similarity_search",
            "search": "similarity_search",
            "similar": "similarity_search",
            "find_similar": "similarity_search",
            # Predict operations
            "predict": "temporal_predict",
            "forecast": "temporal_predict",
            # Verify operations
            "verify": "fact_tree",
            "explain": "fact_tree",
            "prove": "fact_tree",
        }
        
        normalized = operation_aliases.get(operation, operation)
        if normalized != operation:
            logger.debug(f"Normalized operation '{operation}' -> '{normalized}'")
        return normalized
    
    async def _execute_structured_query(
        self,
        org_id: str,
        model_id: str,
        operation: str,
        query: Dict[str, Any],
    ) -> FactTree:
        """
        Execute a structured query against the QueryService.

        All operations return FactTree. On failure, raises so the caller
        can decide whether to return ERROR state or handle differently.
        """
        from domains.models.schemas import (
            SimilaritySearchRequest,
            FactTreeRequest,
            TemporalPredictRequest,
        )

        # Normalize operation name to canonical form
        operation = self._normalize_operation(operation)

        try:
            if operation == "similarity_search":
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=query.get("top_k", 10),
                )
                # similarity_search already returns FactTree
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
            
            elif operation == "fact_tree":
                # fact_tree intent is effectively a similarity search —
                # route through similarity_search which properly encodes
                # and returns scored results.
                request = SimilaritySearchRequest(
                    query=query.get("query", query.get("claim", "")),
                    top_k=query.get("top_k", 10),
                )
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
            
            elif operation == "temporal_predict":
                current_state = query.get("current_state", [query.get("query", "")])
                if isinstance(current_state, str):
                    current_state = [current_state]
                
                request = TemporalPredictRequest(
                    current_state=current_state,
                    steps_ahead=query.get("steps_ahead", 1),
                    beam_width=query.get("beam_width", 3),
                )
                # Use the _as_fact_tree method
                return await self.query_service.predict_temporal_as_fact_tree(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
            
            elif operation == "list":
                # Use dedicated list method that returns FactTree
                limit = query.get("limit", 100)
                return await self.query_service.list_glyphs_as_fact_tree(
                    org_id=org_id,
                    model_id=model_id,
                    limit=limit,
                )
            
            elif operation == "count":
                # Use dedicated count method that returns FactTree
                logger.info(f"Executing count operation for org={org_id}, model={model_id}")
                return await self.query_service.count_glyphs_as_fact_tree(
                    org_id=org_id,
                    model_id=model_id,
                )
            
            elif operation == "compare":
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=2,
                )
                # similarity_search already returns FactTree
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
            
            else:
                logger.warning(f"Unknown operation '{operation}', defaulting to similarity_search")
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=10,
                )
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # Return error FactTree instead of dict
            error_msg = str(e)
            if "no attributes match" in error_msg.lower() or "cannot encode" in error_msg.lower():
                return FactTreeBuilder.build_error(
                    error_message="The query doesn't match the model's schema. Try rephrasing with specific attribute names or values from your data.",
                    error_type="EncodingError",
                    query=query.get("query"),
                )
            return FactTreeBuilder.build_error(
                error_message=str(e),
                error_type=type(e).__name__,
                query=query.get("query"),
            )
    
    def get_intents(self) -> Dict[str, Any]:
        """
        Get available intents and patterns.
        
        Returns:
            Dictionary with intent names and example patterns
        """
        return {"intents": ["similarity_search"], "patterns": {}}
