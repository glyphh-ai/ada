"""
MCP Server Implementation for Glyphh Runtime.

Implements the Model Context Protocol (MCP) for agent integration.
Exposes Glyphh query tools through the MCP interface.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from domains.auth.service import AuthService, User
from domains.query.service import QueryService
from shared.exceptions import (
    AuthenticationException,
    AuthorizationException,
    GlyphNotFoundException,
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
    # Additional fields for consistent JSON response
    result: Optional[Any] = None
    query_type: Optional[str] = None
    match_method: Optional[str] = None
    confidence: float = 0.0
    query_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compatible dict with consistent JSON structure."""
        return {
            "content": self.content,
            "isError": self.is_error,
            "result": self.result,
            "query_type": self.query_type,
            "match_method": self.match_method,
            "confidence": self.confidence,
            "query_time_ms": self.query_time_ms,
        }


class MCPServer:
    """
    MCP Server for Glyphh Runtime.
    
    Implements the Model Context Protocol to expose Glyphh query tools
    to AI agents. Supports:
    - glyph_similarity_search: Find similar glyphs
    - glyph_fact_tree: Generate verification reports
    - glyph_temporal_predict: Predict future states
    - glyph_create: Create new glyphs
    - glyph_get: Retrieve glyph by ID
    - glyph_list: List glyphs with pagination
    """
    
    def __init__(
        self,
        query_service: QueryService,
        auth_service: AuthService,
    ):
        """
        Initialize MCP Server.
        
        Args:
            query_service: Query service for executing queries
            auth_service: Auth service for token validation
        """
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
        """
        Handle MCP tool invocation with authentication.
        
        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments
            auth_token: Authentication token
            
        Returns:
            MCPResponse with result or error
        """
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
                operation = "write" if tool_name == "glyph_create" else "read"
                await self._auth_service.check_namespace_access(user, namespace, operation)
            
            # Dispatch to handler
            handler = getattr(self, f"_handle_{tool_name}", None)
            if handler is None:
                return self._error_response(f"Handler not implemented: {tool_name}")
            
            result = await handler(arguments, user)
            
            # Log success
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"MCP tool {tool_name} completed in {elapsed:.2f}ms")
            
            # Return consistent JSON response
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
            return self._error_response(f"Authentication failed: {e.message}", is_auth_error=True)
        
        except AuthorizationException as e:
            logger.warning(f"MCP authorization failed: {e}")
            return self._error_response(f"Not authorized: {e.message}", is_auth_error=True)
        
        except ValidationException as e:
            return self._error_response(f"Validation error: {e.message}")
        
        except GlyphNotFoundException as e:
            return self._error_response(f"Glyph not found: {e.message}")
        
        except Exception as e:
            logger.error(f"MCP tool error: {e}", exc_info=True)
            return self._error_response(f"Internal error: {str(e)}")
    
    def _validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate tool arguments against schema.
        
        Args:
            tool_name: Tool name
            arguments: Arguments to validate
            
        Returns:
            Error message if invalid, None if valid
        """
        schema = self._tools[tool_name].input_schema
        required = schema.get("required", [])
        
        # Check required fields
        for field in required:
            if field not in arguments:
                return f"Missing required field: {field}"
        
        # Validate types
        properties = schema.get("properties", {})
        for field, value in arguments.items():
            if field not in properties:
                continue
            
            prop_schema = properties[field]
            expected_type = prop_schema.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                return f"Field '{field}' must be a string"
            elif expected_type == "integer" and not isinstance(value, int):
                return f"Field '{field}' must be an integer"
            elif expected_type == "array" and not isinstance(value, list):
                return f"Field '{field}' must be an array"
            elif expected_type == "object" and not isinstance(value, dict):
                return f"Field '{field}' must be an object"
            
            # Check min/max for integers
            if expected_type == "integer":
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    return f"Field '{field}' must be >= {prop_schema['minimum']}"
                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    return f"Field '{field}' must be <= {prop_schema['maximum']}"
        
        return None
    
    def _error_response(
        self,
        message: str,
        is_auth_error: bool = False
    ) -> MCPResponse:
        """Create an error response."""
        return MCPResponse(
            content=[{"type": "text", "text": message}],
            is_error=True,
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
        Handle natural language query tool.
        
        Flow:
        1. Try rules-based intent matching using model's nl_config
        2. If rules fail, try embedded LLM
        3. If intent matched, execute the configured function with slots
        4. Return consistent JSON response
        
        Does NOT fall back to similarity search - that's a different operation.
        """
        import time
        from domains.nl_query.intent_matcher import IntentMatcher
        from infrastructure.config import get_settings
        
        settings = get_settings()
        namespace = arguments["namespace"]
        query = arguments["query"]
        debug = arguments.get("debug", False)
        
        start_time = time.time()
        
        # TODO: Load model's nl_config from database/model_manager
        # For now, use default intent matcher
        model_nl_config = None
        
        # Create intent matcher with model config
        intent_matcher = IntentMatcher(
            confidence_threshold=0.85,
            model_nl_config=model_nl_config
        )
        
        # Step 1: Try rules-based intent matching
        match_result = await intent_matcher.match_intent(query)
        
        if match_result and match_result.confidence >= 0.85:
            # Intent matched via rules - execute configured function
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                "result": {
                    "intent": match_result.intent,
                    "parameters": match_result.parameters,
                    "structured_query": match_result.structured_query,
                },
                "query_type": match_result.intent,
                "match_method": "rules",
                "confidence": match_result.confidence,
                "query_time_ms": elapsed_ms,
                "translated_query": match_result.structured_query if debug else None,
            }
        
        # Step 2: Try LLM fallback if available
        llm_fallback = None
        try:
            from domains.nl_query.llm_fallback import LLMFallback
            llm_fallback = LLMFallback(model_name=settings.nl_model)
            if not llm_fallback.is_available():
                llm_fallback = None
        except ImportError:
            pass
        
        if llm_fallback is not None:
            try:
                llm_result = await llm_fallback.translate_query(query, namespace)
                if llm_result:
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    return {
                        "result": {
                            "intent": llm_result.get("operation", "unknown"),
                            "parameters": llm_result,
                        },
                        "query_type": llm_result.get("operation", "unknown"),
                        "match_method": "llm",
                        "confidence": 0.7,  # LLM confidence is lower than rules
                        "query_time_ms": elapsed_ms,
                        "translated_query": llm_result if debug else None,
                    }
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
        
        # Step 3: No match - return empty result with confidence 0
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "result": None,
            "query_type": "unknown",
            "match_method": "none",
            "confidence": 0.0,
            "query_time_ms": elapsed_ms,
            "translated_query": None,
        }
