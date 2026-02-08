"""
MCP Server Implementation for Glyphh Runtime.

Implements the Model Context Protocol (MCP) for agent integration.
Exposes the nl_query tool through the MCP interface.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from domains.auth.service import AuthService, User
from domains.query.service import QueryService
from shared.exceptions import (
    AuthenticationException,
    AuthorizationException,
    GlyphNotFoundException,
    ModelNotFoundException,
    ValidationException,
)

logger = logging.getLogger(__name__)


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
    
    Exposes one tool: nl_query.
    Delegates all NL query logic to NLQueryService which handles:
    - Rules-based intent matching (via IntentMatcher/SDK)
    - LLM fallback when rules fail
    - Executing the matched query
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
        """Build MCP tool schemas."""
        return {
            "nl_query": MCPToolSchema(
                name="nl_query",
                description="Execute a natural language query. Uses rules-based intent matching first, falls back to LLM if needed.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Model namespace to query"
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
                    "required": ["namespace", "query"]
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
    ) -> MCPResponse:
        """Handle MCP tool invocation with authentication."""
        start_time = datetime.utcnow()
        
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
            
            # Get namespace and check authorization
            namespace = arguments.get("namespace")
            if namespace:
                await self._auth_service.check_namespace_access(user, namespace, "read")
            
            # Dispatch to handler
            handler = getattr(self, f"_handle_{tool_name}", None)
            if handler is None:
                return self._error_response(f"Handler not implemented: {tool_name}")
            
            result = await handler(arguments, user)
            
            # Log success
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"MCP tool {tool_name} completed in {elapsed:.2f}ms")
            
            return MCPResponse(
                content=[{"type": "json", "data": result}],
                is_error=False,
                result=result.get("result"),
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
    ) -> Dict[str, Any]:
        """
        Handle nl_query tool. Delegates entirely to NLQueryService
        which handles rules matching, LLM fallback, and query execution.
        
        Requires the model to be loaded in the runtime — no fallbacks.
        """
        from domains.nl_query.service import NLQueryService
        from domains.nl_query.intent_matcher import IntentMatcher
        from domains.models.manager import ModelManager
        from infrastructure.config import get_settings
        
        settings = get_settings()
        namespace = arguments["namespace"]
        query = arguments["query"]
        debug = arguments.get("debug", False)
        
        # Verify model is loaded — fail early with clear error
        loaded_model = await self._query_service._model_manager.get_model(namespace)
        if loaded_model is None:
            raise ModelNotFoundException(namespace)
        
        # Create intent matcher
        intent_matcher = IntentMatcher(confidence_threshold=0.85)
        
        # Create LLM fallback if available
        llm_fallback = None
        try:
            from domains.nl_query.llm_fallback import LLMFallback
            llm_fallback = LLMFallback(model_name=settings.nl_model)
            if not llm_fallback.is_available():
                llm_fallback = None
        except ImportError:
            pass
        
        # Use NLQueryService - it handles the full flow
        nl_service = NLQueryService(
            query_service=self._query_service,
            intent_matcher=intent_matcher,
            llm_fallback=llm_fallback,
            confidence_threshold=0.85,
        )
        
        result = await nl_service.execute_nl_query(
            namespace=namespace,
            query=query,
            debug=debug,
        )
        
        response = {
            "result": result.result,
            "query_type": result.query_type,
            "match_method": result.match_method,
            "confidence": result.confidence,
            "query_time_ms": result.query_time_ms,
            "translated_query": result.translated_query if debug else None,
        }
        
        # Include error info when no match was found
        if result.match_method == "none":
            response["error"] = "No intent match found for query"
        
        return response
