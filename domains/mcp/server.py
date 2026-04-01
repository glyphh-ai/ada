"""
MCP Server Implementation for Glyphh Runtime.

Proper MCP protocol implementation using the official mcp Python SDK.
Exposes core tools (nl_query, gql_query) plus model-specific tools
defined in each model's encoder.py via MCP_TOOLS / handle_mcp_tool.

Uses the low-level Server class for dynamic tool registration since
tools vary per org/model (model-specific tools from encoder.py).
"""

import json
import logging
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.server.lowlevel.server import Server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from domains.auth.service import AuthService, User, Permission
from domains.query.service import QueryService
from shared.exceptions import (
    AuthenticationException,
    AuthorizationException,
    GlyphNotFoundException,
    ModelNotFoundException,
    ValidationException,
)

logger = logging.getLogger(__name__)

# Context variables set by the ASGI middleware before each request.
# The MCP SDK's handler callbacks read these to know which org/model
# the current request is scoped to.
current_org_id: ContextVar[str] = ContextVar("current_org_id", default="")
current_model_id: ContextVar[str] = ContextVar("current_model_id", default="")


def _streamline_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a fact_tree response into a compact format when _detail=minimal.

    Models signal minimal mode by setting result["_detail"] = "minimal".
    When set, the verbose fact_tree hierarchy is replaced with a flat
    matches list — cutting the JSON payload from ~60 lines to ~10.

    Full-detail responses pass through unchanged.
    """
    if not isinstance(result, dict):
        return result
    detail = result.pop("_detail", None)
    if detail != "minimal":
        return result

    fact_tree = result.get("fact_tree")
    if not isinstance(fact_tree, dict):
        return result

    children = fact_tree.get("children", [])
    matches = []
    for child in children:
        ds = child.get("data_sample", {})
        if ds:
            matches.append(ds)
        else:
            # Fallback: use description + value
            matches.append({
                "file": child.get("description", ""),
                "confidence": child.get("value", 0),
            })

    result["matches"] = matches
    result.pop("fact_tree", None)
    return result


def create_mcp_server(
    query_service: QueryService,
    auth_service: AuthService,
) -> Server:
    """
    Create an MCP Server using the official SDK.

    Returns a low-level Server instance with on_list_tools and on_call_tool
    handlers registered. The handlers delegate to the same tool logic that
    was previously in the custom MCPServer class.
    """
    handler = ToolHandler(query_service, auth_service)

    app = Server("glyphh-runtime")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        org_id = current_org_id.get()
        model_id = current_model_id.get()
        return await handler.get_tools(org_id, model_id)

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        org_id = current_org_id.get()
        model_id = current_model_id.get()
        return await handler.call_tool(
            tool_name=name,
            arguments=dict(arguments or {}),
            org_id=org_id,
            model_id=model_id,
        )

    return app


class ToolHandler:
    """
    Tool dispatch logic for the Glyphh runtime.

    Handles core tools (nl_query, gql_query, confirm, execute) plus
    model-specific tools defined in encoder.py MCP_TOOLS.
    """

    def __init__(
        self,
        query_service: QueryService,
        auth_service: AuthService,
    ):
        self._query_service = query_service
        self._auth_service = auth_service

    # ------------------------------------------------------------------
    # Tool listing
    # ------------------------------------------------------------------

    async def get_tools(self, org_id: str, model_id: str) -> List[Tool]:
        """Return MCP Tool objects for the given org/model scope."""
        cfg = await self._load_model_config(org_id, model_id) if org_id else {}

        # Determine which optional tools are enabled
        cl = cfg.get("cognitive_loop", False)
        has_cognitive = (
            (isinstance(cl, dict) and cl.get("enabled", False))
            or (not isinstance(cl, dict) and bool(cl))
        )
        has_execute = bool(cfg.get("execute", {}).get("provider"))

        tools: List[Tool] = []

        # Core tools — always present
        tools.append(Tool(
            name="nl_query",
            description=(
                "Execute a natural language query. Matches stored procedures "
                "first, falls back to similarity search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "debug": {
                        "type": "boolean",
                        "description": "Include translation details in response",
                        "default": False,
                    },
                    "stage": {
                        "type": "string",
                        "description": (
                            "Query stage mode: 'auto' (full two-stage), "
                            "'patterns' (Stage 1 exemplar match only), "
                            "'data' (Stage 2 data search only)"
                        ),
                        "enum": ["auto", "patterns", "data"],
                        "default": "auto",
                    },
                    "selected_glyph_id": {
                        "type": "string",
                        "description": (
                            "Glyph ID from ASK disambiguation — skips re-query "
                            "and uses this glyph directly for Stage 2"
                        ),
                    },
                },
                "required": ["query"],
            },
        ))

        tools.append(Tool(
            name="gql_query",
            description=(
                "Execute a GQL (Glyph Query Language) query directly. "
                "Returns results as a Fact Tree."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            'GQL query string (e.g., \'FIND SIMILAR TO "red car" LIMIT 10\')'
                        ),
                    },
                    "enable_cache": {
                        "type": "boolean",
                        "description": "Enable semantic query caching",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        ))

        # Optional: confirm (episodic memory)
        if has_cognitive:
            tools.append(Tool(
                name="confirm",
                description=(
                    "Confirm or correct the last nl_query result. "
                    "Call after executing a DONE action (was_correct=true) "
                    "or after resolving an ASK (was_correct=false, correct_action=...). "
                    "Enables episodic memory: confirmed patterns route faster over time."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "was_correct": {
                            "type": "boolean",
                            "description": "True if the DONE result was correct, false if not",
                        },
                        "correct_action": {
                            "type": "string",
                            "description": "The correct action key (required when was_correct=false)",
                        },
                    },
                    "required": ["was_correct"],
                },
            ))

        # Optional: execute (provider actions)
        if has_execute:
            tools.append(Tool(
                name="execute",
                description=(
                    "Execute an action via the model's configured provider "
                    "(e.g. Pipedream Connect). Pass the action_key from a "
                    "DONE nl_query result along with the required props."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
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
                    "required": ["action_key", "external_user_id"],
                },
            ))

        # Model-specific tools from encoder.py MCP_TOOLS
        if org_id and model_id:
            loaded_model = await self._query_service._model_manager.get_model(
                org_id, model_id,
            )
            if loaded_model and getattr(loaded_model, "mcp_tools", None):
                for tool_def in loaded_model.mcp_tools:
                    tools.append(Tool(
                        name=tool_def["name"],
                        description=tool_def.get("description", ""),
                        inputSchema=tool_def.get("input_schema", {}),
                    ))

        return tools

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        org_id: str,
        model_id: str,
    ) -> CallToolResult:
        """Dispatch a tool call and return an MCP-spec CallToolResult."""
        # Inject org/model into arguments for handler compatibility
        arguments["org_id"] = org_id
        arguments["model_id"] = model_id

        start_time = datetime.utcnow()

        try:
            # Core tool dispatch
            if tool_name == "nl_query":
                result = await self._handle_nl_query(arguments)
            elif tool_name == "gql_query":
                result = await self._handle_gql_query(arguments)
            elif tool_name == "confirm":
                result = await self._handle_confirm(arguments)
            elif tool_name == "execute":
                result = await self._handle_execute(arguments)
            else:
                # Model-specific tool
                result = await self._handle_model_tool(
                    tool_name, arguments, org_id, model_id,
                )

            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"MCP tool {tool_name} completed in {elapsed:.2f}ms")

            # Streamline fact_tree responses when detail=minimal.
            # The model sets _detail="minimal" to signal the runtime should
            # flatten the fact_tree into a compact matches list.
            result = _streamline_response(result)

            # Return as MCP-spec CallToolResult with JSON text content
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, default=str),
                )],
                isError=False,
            )

        except (ModelNotFoundException, GlyphNotFoundException, ValidationException) as e:
            logger.warning(f"MCP tool {tool_name} error: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=str(e))],
                isError=True,
            )
        except Exception as e:
            logger.error(f"MCP tool {tool_name} error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Internal error: {e}")],
                isError=True,
            )

    # ------------------------------------------------------------------
    # Model-specific tool handler
    # ------------------------------------------------------------------

    async def _handle_model_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        org_id: str,
        model_id: str,
    ) -> Dict[str, Any]:
        """Dispatch to a model-specific MCP tool handler."""
        loaded_model = await self._query_service._model_manager.get_model(
            org_id, model_id,
        )
        if not loaded_model:
            raise ModelNotFoundException(org_id, model_id)

        model_tool_names = {
            t["name"] for t in getattr(loaded_model, "mcp_tools", []) or []
        }
        if tool_name not in model_tool_names:
            raise ValidationException("tool_name", f"Unknown tool: {tool_name}")
        if not getattr(loaded_model, "handle_mcp_tool_fn", None):
            raise ValidationException(
                "handle_mcp_tool", f"Model defines tool '{tool_name}' but has no handle_mcp_tool handler"
            )

        context = {
            "org_id": org_id,
            "model_id": model_id,
            "encoder": loaded_model.encoder,
            "encode_query_fn": loaded_model.encode_query_fn,
            "similarity_calculator": loaded_model.similarity_calculator,
            "model_manager": self._query_service._model_manager,
            "session_factory": self._query_service._session_factory,
        }

        import asyncio as _aio

        handler_fn = loaded_model.handle_mcp_tool_fn
        if _aio.iscoroutinefunction(handler_fn):
            result = await handler_fn(tool_name, arguments, context)
        else:
            result = handler_fn(tool_name, arguments, context)

        if not isinstance(result, dict):
            result = {"result": result}
        return result

    # ------------------------------------------------------------------
    # Core tool handlers (logic preserved from original implementation)
    # ------------------------------------------------------------------

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

    async def _handle_nl_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle nl_query tool."""
        from domains.nl_query.service import NLQueryService

        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        query = arguments["query"]
        debug = arguments.get("debug", False)
        stage = arguments.get("stage", "auto")
        confirmed = arguments.get("confirmed", False)
        selected_glyph_id = arguments.get("selected_glyph_id")

        loaded_model = await self._query_service._model_manager.get_model(
            org_id, model_id,
        )
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)

        # Load config for NL service parameters
        min_gap = 0.03
        similarity_threshold = 0.5
        top_k = 10
        two_stage = False
        result_field = None
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

        result = await nl_service.execute_nl_query(
            org_id=org_id,
            model_id=model_id,
            query=query,
            debug=debug,
            stage=stage,
            confirmed=confirmed,
            selected_glyph_id=selected_glyph_id,
        )

        response = result.to_dict()

        if result.match_method == "stored_procedure" and hasattr(result, "procedure_name"):
            response["procedure_name"] = result.procedure_name
        if result.match_method == "none":
            response["error"] = "No intent match found for query"

        return response

    async def _handle_gql_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gql_query tool."""
        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        query = arguments["query"]
        enable_cache = arguments.get("enable_cache", True)

        loaded_model = await self._query_service._model_manager.get_model(
            org_id, model_id,
        )
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)

        try:
            from glyphh.gql import GQLExecutor, ExecutionContext
            from domains.gql.storage import DatabaseGlyphStorage
            from domains.models.storage import GlyphStorage
            from shared.similarity_service import SimilarityService

            async with self._query_service._session_factory() as session:
                db_storage = GlyphStorage(session)
                db_glyphs, embeddings = await db_storage.list_glyphs_with_embeddings(
                    org_id=org_id, model_id=model_id, limit=10000,
                )
                glyph_ids = [g.id for g in db_glyphs]
                hierarchical = await db_storage.get_hierarchical_embeddings(
                    org_id=org_id, model_id=model_id, glyph_ids=glyph_ids,
                )

            similarity_service = SimilarityService(
                similarity_calculator=getattr(loaded_model, "similarity_calculator", None)
            )
            storage = DatabaseGlyphStorage(
                org_id=org_id,
                model_id=model_id,
                glyphs=db_glyphs,
                embeddings=embeddings,
                similarity_service=similarity_service,
                hierarchical_embeddings=hierarchical,
            )
            context = ExecutionContext(
                model=loaded_model.sdk_model,
                storage=storage,
                encoder=getattr(loaded_model, "encoder", None),
            )
            executor = GQLExecutor(context=context, enable_cache=enable_cache)
            fact_tree = executor.execute(query)
            fact_tree_json = fact_tree.to_json() if hasattr(fact_tree, "to_json") else {}

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

    async def _handle_confirm(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle confirm tool (episodic memory)."""
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

        return NLQueryService.confirm_last(org_id, model_id, was_correct, correct_action)

    async def _handle_execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execute tool (provider actions)."""
        from shared.providers import get_provider

        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        action_key = arguments["action_key"]
        props = arguments.get("props", {})
        external_user_id = arguments["external_user_id"]

        cfg = await self._load_model_config(org_id, model_id)
        execute_config = cfg.get("execute", {})
        if not execute_config.get("provider"):
            return {
                "state": "ERROR",
                "error": f"Model {org_id}/{model_id} has no execute provider configured",
            }

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

        return {
            "state": "DONE" if result.get("success") else "ERROR",
            "provider": execute_config["provider"],
            "action_key": action_key,
            "result": result,
            "confidence": 1.0,
            "match_method": "execute",
            "query_type": "execute",
        }
