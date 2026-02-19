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
    
    Exposes two tools: nl_query and gql_query.
    - nl_query: NL text → NLQueryService → stored procedure match → QueryService → HDC Engine
    - gql_query: raw GQL → QueryService → HDC Engine
    Both tools return the same output shape: {state, fact_tree, confidence, match_method}
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
        }
    
    def get_tool_schemas(self) -> List[MCPToolSchema]:
        """Return MCP tool schemas for all exposed tools."""
        return list(self._tools.values())
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Return tools list in MCP format."""
        return [tool.to_dict() for tool in self._tools.values()]

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        auth_token: str,
        progress_token: Optional[str] = None,
        send_notification: NotificationSender = None,
    ) -> MCPResponse:
        """
        Handle MCP tool invocation with authentication and optional progress.
        
        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments
            auth_token: Authentication token
            progress_token: Optional MCP progress token for long-running ops
            send_notification: Optional async function to send notifications
        """
        start_time = datetime.utcnow()
        
        # Create progress handler if token provided
        progress_handler = None
        if progress_token and send_notification:
            progress_handler = MCPProgressHandler(send_notification)
        
        try:
            # Authenticate
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

        # Create NL service — deterministic, no LLM fallback
        nl_service = NLQueryService(
            query_service=self._query_service,
            confidence_threshold=0.85,
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
            from shared.similarity_service import SimilarityService

            # Fetch glyphs with embeddings from database
            db_glyphs, embeddings = await self._query_service.list_glyphs_with_embeddings(
                org_id=org_id,
                model_id=model_id,
                limit=10000,
            )

            logger.info(f"Fetched {len(db_glyphs)} glyphs with {len(embeddings)} embeddings from database")

            # Create SimilarityService for similarity computations
            similarity_service = SimilarityService(
                similarity_calculator=getattr(loaded_model, 'similarity_calculator', None)
            )

            # Create DatabaseGlyphStorage
            storage = DatabaseGlyphStorage(
                org_id=org_id,
                model_id=model_id,
                glyphs=db_glyphs,
                embeddings=embeddings,
                similarity_service=similarity_service,
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
    
    
