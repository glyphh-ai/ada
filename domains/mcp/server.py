"""
MCP Server Implementation for Glyphh Runtime.

Implements the Model Context Protocol (MCP) for agent integration.
Exposes two tools: nl_query and gql_query.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from domains.auth.service import AuthService, User
from domains.mcp.progress import MCPProgressHandler
from domains.query.service import QueryService
from shared.exceptions import (
    AuthenticationException,
    AuthorizationException,
    GlyphNotFoundException,
    ModelNotFoundException,
    ValidationException,
)

logger = logging.getLogger(__name__)


# Type alias for notification sender
NotificationSender = Optional[Callable[[dict], Coroutine[Any, Any, None]]]


@dataclass
class MCPToolSchema:
    """Schema definition for an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResponse:
    """Response from an MCP tool invocation."""
    content: List[Dict[str, Any]]
    is_error: bool = False
    error: Optional[str] = None
    result: Optional[Any] = None
    query_type: Optional[str] = None
    match_method: Optional[str] = None
    confidence: float = 0.0
    query_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compatible dict with consistent JSON structure."""
        response = {
            "content": self.content,
            "isError": self.is_error,
            "result": self.result,
            "query_type": self.query_type,
            "match_method": self.match_method,
            "confidence": self.confidence,
            "query_time_ms": self.query_time_ms,
        }
        if self.error is not None:
            response["error"] = self.error
        return response


class MCPServer:
    """
    MCP Server for Glyphh Runtime.

    Core tools (all models): nl_query, gql_query
    Optional tools (config-driven):
      - confirm:  when cognitive_loop.enabled = true
      - execute:  when execute.provider is set
    """

    def __init__(
        self,
        query_service: QueryService,
        auth_service: AuthService,
    ):
        self._query_service = query_service
        self._auth_service = auth_service
        self._tools = self._build_tool_schemas()

    def _build_tool_schemas(self) -> Dict[str, MCPToolSchema]:
        """Build MCP tool schemas. Only nl_query and gql_query are registered."""
        return {
            "nl_query": MCPToolSchema(
                name="nl_query",
                description="Execute a natural language query. Matches stored procedures first, falls back to similarity search.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "org_id": {
                            "type": "string",
                            "description": "Organization ID"
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Model ID"
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural language query"
                        },
                        "debug": {
                            "type": "boolean",
                            "description": "Include translation details in response",
                            "default": False
                        },
                        "stage": {
                            "type": "string",
                            "description": "Query stage mode: 'auto' (full two-stage), 'patterns' (Stage 1 exemplar match only), 'data' (Stage 2 data search only)",
                            "enum": ["auto", "patterns", "data"],
                            "default": "auto"
                        },
                        "selected_glyph_id": {
                            "type": "string",
                            "description": "Glyph ID from ASK disambiguation — skips re-query and uses this glyph directly for Stage 2"
                        }
                    },
                    "required": ["org_id", "model_id", "query"]
                }
            ),
            "gql_query": MCPToolSchema(
                name="gql_query",
                description="Execute a GQL (Glyph Query Language) query directly. Returns results as a Fact Tree.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "org_id": {
                            "type": "string",
                            "description": "Organization ID"
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Model ID"
                        },
                        "query": {
                            "type": "string",
                            "description": "GQL query string (e.g., 'FIND SIMILAR TO \"red car\" LIMIT 10')"
                        },
                        "enable_cache": {
                            "type": "boolean",
                            "description": "Enable semantic query caching",
                            "default": True
                        }
                    },
                    "required": ["org_id", "model_id", "query"]
                }
            ),
            "confirm": MCPToolSchema(
                name="confirm",
                description=(
                    "Confirm or correct the last nl_query result. "
                    "Call after executing a DONE action (was_correct=true) "
                    "or after resolving an ASK (was_correct=false, correct_action=...)."
                    " Enables episodic memory: confirmed patterns route faster over time."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "org_id": {
                            "type": "string",
                            "description": "Organization ID",
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Model ID",
                        },
                        "was_correct": {
                            "type": "boolean",
                            "description": "True if the DONE result was correct, false if not",
                        },
                        "correct_action": {
                            "type": "string",
                            "description": "The correct action key (required when was_correct=false)",
                        },
                    },
                    "required": ["org_id", "model_id", "was_correct"],
                },
            ),
            "execute": MCPToolSchema(
                name="execute",
                description=(
                    "Execute an action via the model's configured provider "
                    "(e.g. Pipedream Connect). Pass the action_key from a "
                    "DONE nl_query result along with the required props."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "org_id": {
                            "type": "string",
                            "description": "Organization ID",
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Model ID",
                        },
                        "action_key": {
                            "type": "string",
                            "description": "Action component key from nl_query result",
                        },
                        "props": {
                            "type": "object",
                            "description": "Configured properties for the action",
                            "default": {},
                        },
                        "external_user_id": {
                            "type": "string",
                            "description": "Your end-user's ID (for provider auth)",
                        },
                    },
                    "required": [
                        "org_id", "model_id", "action_key", "external_user_id",
                    ],
                },
            ),
        }
    
    async def _load_model_config(self, org_id: str, model_id: str) -> dict:
        """Load a model's config.yaml — tries disk first, falls back to DB source_files."""
        import yaml
        try:
            loaded_model = await self._query_service._model_manager.get_model(
                org_id, model_id,
            )
            if not loaded_model:
                return {}
            # Try disk first
            if loaded_model.model_path:
                from pathlib import Path as _Path
                mp = _Path(loaded_model.model_path)
                cfg_path = (mp if mp.is_dir() else mp.parent) / "config.yaml"
                if cfg_path.exists():
                    return yaml.safe_load(cfg_path.read_text()) or {}
            # Fallback: config.yaml stored in DB source_files (Heroku / no-disk)
            mgr = self._query_service._model_manager
            from domains.models.db_models import ModelConfig
            from sqlalchemy import select
            async with mgr._db_session_factory() as session:
                result = await session.execute(
                    select(ModelConfig.source_files).where(
                        ModelConfig.org_id == org_id,
                        ModelConfig.model_id == model_id,
                    )
                )
                source_files = result.scalar_one_or_none()
            if source_files and isinstance(source_files, dict):
                config_text = source_files.get("config.yaml")
                if config_text:
                    return yaml.safe_load(config_text) or {}
        except Exception:
            pass
        return {}

    def get_tool_schemas(self) -> List[MCPToolSchema]:
        """Return MCP tool schemas for all exposed tools."""
        return list(self._tools.values())

    async def get_tools_list(
        self, org_id: str = "", model_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Return tools list in MCP format, filtered by model config.

        - nl_query, gql_query: always included
        - confirm: included when cognitive_loop.enabled = true
        - execute: included when execute.provider is set
        """
        cfg = await self._load_model_config(org_id, model_id) if org_id else {}

        # Determine which optional tools are enabled
        cl = cfg.get("cognitive_loop", False)
        has_cognitive = (
            (isinstance(cl, dict) and cl.get("enabled", False)) or
            (not isinstance(cl, dict) and bool(cl))
        )
        has_execute = bool(cfg.get("execute", {}).get("provider"))

        tools = []
        for name, schema in self._tools.items():
            if name == "confirm" and not has_cognitive:
                continue
            if name == "execute" and not has_execute:
                continue
            tools.append(schema.to_dict())
        return tools

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        auth_token: str,
        progress_token: Optional[str] = None,
        send_notification: NotificationSender = None,
        pre_authenticated_user: Any = None,
    ) -> MCPResponse:
        """
        Handle MCP tool invocation with authentication and optional progress.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments
            auth_token: Authentication token
            progress_token: Optional MCP progress token for long-running ops
            send_notification: Optional async function to send notifications
            pre_authenticated_user: Skip internal auth if already validated by route
        """
        start_time = datetime.utcnow()

        # Create progress handler if token provided
        progress_handler = None
        if progress_token and send_notification:
            progress_handler = MCPProgressHandler(send_notification)

        try:
            # Authenticate (skip if already validated by FastAPI dependency)
            if pre_authenticated_user is not None:
                user = pre_authenticated_user
            else:
                user = await self._auth_service.validate_token(auth_token)
            
            # Validate tool exists
            if tool_name not in self._tools:
                return self._error_response(f"Unknown tool: {tool_name}")
            
            # Validate arguments
            validation_error = self._validate_arguments(tool_name, arguments)
            if validation_error:
                return self._error_response(validation_error)
            
            # Get org_id/model_id and check authorization
            org_id = arguments.get("org_id")
            model_id = arguments.get("model_id")
            if org_id:
                await self._auth_service.check_access(user, org_id, model_id or "", "read")
            
            # Dispatch to handler with progress support
            handler = getattr(self, f"_handle_{tool_name}", None)
            if handler is None:
                return self._error_response(f"Handler not implemented: {tool_name}")
            
            # Pass progress handler to tool handlers that support it
            result = await handler(
                arguments, 
                user, 
                progress_handler=progress_handler,
                progress_token=progress_token,
            )
            
            # Log success
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"MCP tool {tool_name} completed in {elapsed:.2f}ms")
            
            return MCPResponse(
                content=[{"type": "json", "data": result}],
                is_error=False,
                result=result.get("fact_tree") or result.get("result"),
                query_type=result.get("query_type"),
                match_method=result.get("match_method"),
                confidence=result.get("confidence", 0.0),
                query_time_ms=result.get("query_time_ms", elapsed),
            )
            
        except AuthenticationException as e:
            logger.warning(f"MCP authentication failed: {e}")
            return self._error_response(f"Authentication failed: {e.message}")
        except AuthorizationException as e:
            logger.warning(f"MCP authorization failed: {e}")
            return self._error_response(f"Not authorized: {e.message}")
        except ModelNotFoundException as e:
            logger.warning(f"MCP model not found: {e}")
            return self._error_response(f"Model not found: {e.message}. Deploy the model to the runtime first.")
        except ValidationException as e:
            return self._error_response(f"Validation error: {e.message}")
        except GlyphNotFoundException as e:
            return self._error_response(f"Glyph not found: {e.message}")
        except Exception as e:
            logger.error(f"MCP tool error: {e}", exc_info=True)
            return self._error_response(f"Internal error: {str(e)}")
    
    def _validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate tool arguments against schema."""
        schema = self._tools[tool_name].input_schema
        required = schema.get("required", [])
        for field in required:
            if field not in arguments:
                return f"Missing required field: {field}"
        return None
    
    def _error_response(self, message: str) -> MCPResponse:
        """Create an error response."""
        return MCPResponse(
            content=[{"type": "text", "text": message}],
            is_error=True,
            error=message,
        )
    

    # =========================================================================
    # Tool Handlers
    # =========================================================================
    
    async def _handle_nl_query(
        self,
        arguments: Dict[str, Any],
        user: User,
        progress_handler: Optional[MCPProgressHandler] = None,
        progress_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle nl_query tool. Routes: NLQueryService → stored procedure match → QueryService → HDC Engine.

        Requires the model to be loaded in the runtime — no fallbacks.
        Sends progress notifications if progress_token is provided.
        """
        from domains.nl_query.service import NLQueryService

        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        query = arguments["query"]
        debug = arguments.get("debug", False)
        stage = arguments.get("stage", "auto")
        confirmed = arguments.get("confirmed", False)
        selected_glyph_id = arguments.get("selected_glyph_id")

        # Send initial progress if token provided
        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token, 
                progress=10, 
                message="Analyzing query..."
            )

        # Verify model is loaded — fail early with clear error
        loaded_model = await self._query_service._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)

        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token, 
                progress=30, 
                message="Processing query..."
            )

        # Create NL service — deterministic, no LLM fallback.
        # Pass the model's assess_query_fn (if any) for semantic slot checking,
        # and min_gap from config for gap-based disambiguation.
        min_gap = 0.03  # default; overridden by model's disambiguation.min_gap
        similarity_threshold = 0.5  # default; overridden by model's similarity.threshold
        top_k = 10  # default; overridden by model's similarity.top_k
        two_stage = False
        result_field = None  # metadata field to surface as display result
        cognitive_loop_enabled = False
        cognitive_loop_config: dict = {}
        _cfg = await self._load_model_config(org_id, model_id)
        if _cfg:
            min_gap = _cfg.get("disambiguation", {}).get("min_gap", min_gap)
            sim_cfg = _cfg.get("similarity", {})
            similarity_threshold = sim_cfg.get("threshold", similarity_threshold)
            top_k = sim_cfg.get("top_k", top_k)
            result_field = sim_cfg.get("result_field")
            two_stage = bool(_cfg.get("gql_query_default"))
            cl = _cfg.get("cognitive_loop", False)
            if isinstance(cl, dict):
                cognitive_loop_enabled = cl.get("enabled", False)
                cognitive_loop_config = cl
            elif cl:
                cognitive_loop_enabled = True
                cognitive_loop_config = {}

        nl_service = NLQueryService(
            query_service=self._query_service,
            confidence_threshold=similarity_threshold,
            assess_query_fn=getattr(loaded_model, "assess_query_fn", None),
            min_gap=min_gap,
            top_k=top_k,
            two_stage=two_stage,
            result_field=result_field,
            cognitive_loop_enabled=cognitive_loop_enabled,
            cognitive_loop_config=cognitive_loop_config,
        )

        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token, 
                progress=50, 
                message="Executing query..."
            )

        result = await nl_service.execute_nl_query(
            org_id=org_id,
            model_id=model_id,
            query=query,
            debug=debug,
            stage=stage,
            confirmed=confirmed,
            selected_glyph_id=selected_glyph_id,
        )

        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token, 
                progress=100, 
                message="Complete"
            )

        # Build response with consistent output shape: {state, fact_tree, confidence, match_method}
        response = result.to_dict()

        # Include procedure_name if matched via stored procedure
        if result.match_method == "stored_procedure" and hasattr(result, 'procedure_name'):
            response["procedure_name"] = result.procedure_name

        if result.match_method == "none":
            response["error"] = "No intent match found for query"

        return response


    async def _handle_gql_query(
        self,
        arguments: Dict[str, Any],
        user: User,
        progress_handler: Optional[MCPProgressHandler] = None,
        progress_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle gql_query tool. Routes: QueryService → HDC Engine.

        Parses the GQL query, creates an execution plan, and returns
        results with the same output shape as nl_query: {state, fact_tree, confidence, match_method}.
        """
        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        query = arguments["query"]
        enable_cache = arguments.get("enable_cache", True)

        # Send initial progress
        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token, 
                progress=10, 
                message="Parsing query..."
            )

        # Verify model is loaded
        loaded_model = await self._query_service._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)

        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token, 
                progress=30, 
                message="Building execution context..."
            )

        try:
            # Import GQL components
            from glyphh.gql import (
                parse,
                GQLExecutor,
                ExecutionContext,
                GQLError,
            )
            from domains.gql.storage import DatabaseGlyphStorage
            from domains.models.storage import GlyphStorage
            from shared.similarity_service import SimilarityService

            # Fetch cortex + hierarchical embeddings in one session
            async with self._query_service._session_factory() as session:
                db_storage = GlyphStorage(session)

                db_glyphs, embeddings = await db_storage.list_glyphs_with_embeddings(
                    org_id=org_id,
                    model_id=model_id,
                    limit=10000,
                )

                # Fetch layer/segment vectors for AT LAYER queries
                glyph_ids = [g.id for g in db_glyphs]
                hierarchical = await db_storage.get_hierarchical_embeddings(
                    org_id=org_id,
                    model_id=model_id,
                    glyph_ids=glyph_ids,
                )

            logger.info(
                f"Fetched {len(db_glyphs)} glyphs with {len(embeddings)} cortex "
                f"and {len(hierarchical)} hierarchical embeddings from database"
            )

            # Create SimilarityService for similarity computations
            similarity_service = SimilarityService(
                similarity_calculator=getattr(loaded_model, 'similarity_calculator', None)
            )

            # Create DatabaseGlyphStorage with hierarchical vectors
            storage = DatabaseGlyphStorage(
                org_id=org_id,
                model_id=model_id,
                glyphs=db_glyphs,
                embeddings=embeddings,
                similarity_service=similarity_service,
                hierarchical_embeddings=hierarchical,
            )

            logger.info(f"GQL context built with {len(db_glyphs)} glyphs from database using DatabaseGlyphStorage")

            # Create ExecutionContext with storage parameter
            context = ExecutionContext(
                model=loaded_model.sdk_model,
                storage=storage,
                encoder=getattr(loaded_model, 'encoder', None),
            )

            if progress_handler and progress_token:
                await progress_handler.notify(
                    progress_token, 
                    progress=50, 
                    message="Executing query..."
                )

            # Create executor and run query
            executor = GQLExecutor(
                context=context,
                enable_cache=enable_cache,
            )

            fact_tree = executor.execute(query)

            if progress_handler and progress_token:
                await progress_handler.notify(
                    progress_token, 
                    progress=100, 
                    message="Complete"
                )

            # Return consistent output shape: {state, fact_tree, confidence, match_method}
            fact_tree_json = fact_tree.to_json() if hasattr(fact_tree, 'to_json') else {}

            return {
                "state": "DONE",
                "fact_tree": fact_tree_json,
                "confidence": 1.0,
                "match_method": "direct",
                "query_type": "gql",
                "query_time_ms": 0.0,
                "cache_stats": executor.get_cache_stats() if enable_cache else None,
            }

        except Exception as e:
            logger.error(f"GQL query error: {e}", exc_info=True)
            return {
                "state": "ERROR",
                "fact_tree": None,
                "confidence": 0.0,
                "match_method": "direct",
                "query_type": "gql",
                "error": str(e),
            }

    async def _handle_confirm(
        self,
        arguments: Dict[str, Any],
        user: User,
        progress_handler: Optional[MCPProgressHandler] = None,
        progress_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Confirm or correct the last nl_query result for episodic memory.

        When was_correct=True:  strengthens the recalled idea (Hebbian reinforcement).
        When was_correct=False: stores a correction so future similar queries route correctly.
        """
        from domains.nl_query.service import NLQueryService

        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        was_correct = arguments["was_correct"]
        correct_action = arguments.get("correct_action")

        if not was_correct and not correct_action:
            return {
                "state": "ERROR",
                "error": "correct_action is required when was_correct=false",
            }

        result = NLQueryService.confirm_last(
            org_id, model_id, was_correct, correct_action,
        )
        return result

    async def _handle_execute(
        self,
        arguments: Dict[str, Any],
        user: User,
        progress_handler: Optional[MCPProgressHandler] = None,
        progress_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an action via the model's configured provider.

        Loads the provider from model config (execute.provider), then
        calls provider.execute(action_key, props, external_user_id).
        """
        from shared.providers import get_provider

        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        action_key = arguments["action_key"]
        props = arguments.get("props", {})
        external_user_id = arguments["external_user_id"]

        # Load model config to get execute section
        cfg = await self._load_model_config(org_id, model_id)
        execute_config = cfg.get("execute", {})
        if not execute_config.get("provider"):
            return {
                "state": "ERROR",
                "error": f"Model {org_id}/{model_id} has no execute provider configured",
            }

        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token,
                progress=30,
                message=f"Executing {action_key}...",
            )

        try:
            provider = get_provider(org_id, model_id, execute_config)
            result = await provider.execute(
                action_key=action_key,
                props=props,
                external_user_id=external_user_id,
            )
        except ValueError as e:
            return {"state": "ERROR", "error": str(e)}
        except Exception as e:
            logger.error(f"Execute failed: {e}", exc_info=True)
            return {"state": "ERROR", "error": f"Execution failed: {e}"}

        if progress_handler and progress_token:
            await progress_handler.notify(
                progress_token, progress=100, message="Complete",
            )

        return {
            "state": "DONE" if result.get("success") else "ERROR",
            "provider": execute_config["provider"],
            "action_key": action_key,
            "result": result,
            "confidence": 1.0,
            "match_method": "execute",
            "query_type": "execute",
        }


