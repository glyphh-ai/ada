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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compatible dict."""
        return {
            "content": self.content,
            "isError": self.is_error,
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
            "glyph_similarity_search": MCPToolSchema(
                name="glyph_similarity_search",
                description="Search for glyphs similar to a query text. Returns ranked results with similarity scores.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Model namespace to search in"
                        },
                        "query": {
                            "type": "string",
                            "description": "Query text to search for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional metadata filters",
                            "additionalProperties": True
                        }
                    },
                    "required": ["namespace", "query"]
                }
            ),
            "glyph_fact_tree": MCPToolSchema(
                name="glyph_fact_tree",
                description="Generate a fact tree to verify a claim. Returns a hierarchical verification report with citations.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Model namespace"
                        },
                        "claim": {
                            "type": "string",
                            "description": "Claim to verify"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum tree depth (default: 3)",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["namespace", "claim"]
                }
            ),
            "glyph_temporal_predict": MCPToolSchema(
                name="glyph_temporal_predict",
                description="Predict future states based on current state using temporal edges.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Model namespace"
                        },
                        "current_state": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Current state concepts"
                        },
                        "steps_ahead": {
                            "type": "integer",
                            "description": "Number of steps to predict (default: 1)",
                            "default": 1,
                            "minimum": 1,
                            "maximum": 10
                        },
                        "beam_width": {
                            "type": "integer",
                            "description": "Beam width for search (default: 5)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        }
                    },
                    "required": ["namespace", "current_state"]
                }
            ),
            "glyph_create": MCPToolSchema(
                name="glyph_create",
                description="Create a new glyph from concept text.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Model namespace"
                        },
                        "concept": {
                            "type": "string",
                            "description": "Concept text to encode"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata",
                            "additionalProperties": True
                        }
                    },
                    "required": ["namespace", "concept"]
                }
            ),
            "glyph_get": MCPToolSchema(
                name="glyph_get",
                description="Retrieve a glyph by ID.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Model namespace"
                        },
                        "glyph_id": {
                            "type": "string",
                            "description": "Glyph UUID"
                        }
                    },
                    "required": ["namespace", "glyph_id"]
                }
            ),
            "glyph_list": MCPToolSchema(
                name="glyph_list",
                description="List glyphs in a namespace with pagination.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Model namespace"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 100)",
                            "default": 100,
                            "minimum": 1,
                            "maximum": 1000
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (default: 0)",
                            "default": 0,
                            "minimum": 0
                        }
                    },
                    "required": ["namespace"]
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
            
            return MCPResponse(
                content=[{"type": "text", "text": str(result)}],
                is_error=False,
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
    
    async def _handle_glyph_similarity_search(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """Handle similarity search tool."""
        results = await self._query_service.similarity_search(
            namespace=arguments["namespace"],
            query=arguments["query"],
            top_k=arguments.get("top_k", 10),
            user_permissions=user,
            filters=arguments.get("filters"),
        )
        
        return {
            "results": [
                {
                    "glyph_id": str(r.glyph.id),
                    "concept_text": r.glyph.concept_text,
                    "similarity_score": r.similarity_score,
                    "final_score": r.final_score,
                    "metadata": r.glyph.metadata,
                }
                for r in results
            ],
            "total_count": len(results),
        }
    
    async def _handle_glyph_fact_tree(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """Handle fact tree generation tool."""
        fact_tree = await self._query_service.generate_fact_tree(
            namespace=arguments["namespace"],
            claim=arguments["claim"],
            max_depth=arguments.get("max_depth", 3),
            user_permissions=user,
        )
        
        return fact_tree.to_dict() if hasattr(fact_tree, 'to_dict') else fact_tree
    
    async def _handle_glyph_temporal_predict(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """Handle temporal prediction tool."""
        predictions = await self._query_service.predict_temporal(
            namespace=arguments["namespace"],
            current_state=arguments["current_state"],
            steps_ahead=arguments.get("steps_ahead", 1),
            beam_width=arguments.get("beam_width", 5),
            user_permissions=user,
        )
        
        return {
            "predictions": [
                {
                    "state": p.state if hasattr(p, 'state') else str(p),
                    "confidence": p.confidence if hasattr(p, 'confidence') else 0.0,
                }
                for p in predictions
            ]
        }
    
    async def _handle_glyph_create(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """Handle glyph creation tool."""
        result = await self._query_service.create_glyph(
            namespace=arguments["namespace"],
            concept=arguments["concept"],
            metadata=arguments.get("metadata", {}),
        )
        
        return {
            "glyph_id": str(result.glyph_id),
            "namespace": result.namespace,
            "created_at": result.created_at.isoformat() if result.created_at else None,
        }
    
    async def _handle_glyph_get(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """Handle glyph retrieval tool."""
        glyph = await self._query_service.get_glyph(
            namespace=arguments["namespace"],
            glyph_id=UUID(arguments["glyph_id"]),
        )
        
        return {
            "id": str(glyph.id),
            "namespace": glyph.namespace,
            "concept_text": glyph.concept_text,
            "metadata": glyph.metadata,
            "created_at": glyph.created_at.isoformat() if glyph.created_at else None,
        }
    
    async def _handle_glyph_list(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """Handle glyph listing tool."""
        glyphs = await self._query_service.list_glyphs(
            namespace=arguments["namespace"],
            limit=arguments.get("limit", 100),
            offset=arguments.get("offset", 0),
        )
        
        return {
            "glyphs": [
                {
                    "id": str(g.id),
                    "concept_text": g.concept_text,
                    "metadata": g.metadata,
                    "created_at": g.created_at.isoformat() if g.created_at else None,
                }
                for g in glyphs
            ],
            "count": len(glyphs),
        }
