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
            "gql_translate": MCPToolSchema(
                name="gql_translate",
                description="Translate a natural language query to GQL without executing it. Useful for debugging and understanding query translation.",
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
                            "description": "Natural language query to translate"
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
            
            # Get org_id/model_id and check authorization
            org_id = arguments.get("org_id")
            model_id = arguments.get("model_id")
            if org_id:
                await self._auth_service.check_access(user, org_id, model_id or "", "read")
            
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
        Handle nl_query tool. Delegates entirely to NLQueryService.
        
        Requires the model to be loaded in the runtime — no fallbacks.
        """
        from domains.nl_query.service import NLQueryService
        from domains.nl_query.intent_matcher import IntentMatcher
        from infrastructure.config import get_settings
        
        settings = get_settings()
        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        query = arguments["query"]
        debug = arguments.get("debug", False)
        
        # Verify model is loaded — fail early with clear error
        loaded_model = await self._query_service._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        # Extract NL config from the loaded model's encoder config
        from shared.encoder_config_factory import EncoderConfigFactory
        model_nl_config = EncoderConfigFactory.extract_nl_encoder_config(loaded_model.sdk_model)
        
        intent_matcher = IntentMatcher(
            confidence_threshold=0.85,
            model_nl_config=model_nl_config,
        )
        
        llm_fallback = None
        try:
            from domains.nl_query.llm_fallback import LLMFallback
            llm_fallback = LLMFallback(model_name=settings.nl_model)
            if not llm_fallback.is_available():
                llm_fallback = None
        except ImportError:
            pass
        
        nl_service = NLQueryService(
            query_service=self._query_service,
            intent_matcher=intent_matcher,
            llm_fallback=llm_fallback,
            confidence_threshold=0.85,
        )
        
        result = await nl_service.execute_nl_query(
            org_id=org_id,
            model_id=model_id,
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
        
        if result.match_method == "none":
            response["error"] = "No intent match found for query"
        
        return response


    async def _handle_gql_query(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """
        Handle gql_query tool. Executes a GQL query directly.
        
        Parses the GQL query, creates an execution plan, and returns
        results as a Fact Tree.
        """
        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        query = arguments["query"]
        enable_cache = arguments.get("enable_cache", True)
        
        # Verify model is loaded
        loaded_model = await self._query_service._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        try:
            # Import GQL components
            from glyphh.gql import (
                parse,
                GQLExecutor,
                ExecutionContext,
                GQLError,
            )
            
            # Build execution context from loaded model
            glyphs = {}
            if hasattr(loaded_model.sdk_model, 'glyphs'):
                for glyph in loaded_model.sdk_model.glyphs:
                    glyphs[glyph.identifier] = glyph
            
            context = ExecutionContext(
                model=loaded_model.sdk_model,
                glyphs=glyphs,
                encoder=getattr(loaded_model, 'encoder', None),
                similarity_calculator=getattr(loaded_model, 'similarity_calculator', None),
            )
            
            # Create executor and run query
            executor = GQLExecutor(
                context=context,
                enable_cache=enable_cache,
            )
            
            fact_tree = executor.execute(query)
            
            # Convert fact tree to dict for response
            result = fact_tree.to_dict() if hasattr(fact_tree, 'to_dict') else str(fact_tree)
            
            return {
                "result": result,
                "query_type": "gql",
                "match_method": "direct",
                "confidence": 1.0,
                "cache_stats": executor.get_cache_stats() if enable_cache else None,
            }
            
        except Exception as e:
            logger.error(f"GQL query error: {e}", exc_info=True)
            return {
                "result": None,
                "error": str(e),
                "query_type": "gql",
                "match_method": "direct",
                "confidence": 0.0,
            }
    
    async def _handle_gql_translate(
        self,
        arguments: Dict[str, Any],
        user: User,
    ) -> Dict[str, Any]:
        """
        Handle gql_translate tool. Translates NL to GQL without executing.
        
        Uses the model's GQL patterns to translate natural language
        queries to GQL syntax.
        """
        org_id = arguments["org_id"]
        model_id = arguments["model_id"]
        query = arguments["query"]
        
        # Verify model is loaded
        loaded_model = await self._query_service._model_manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        try:
            # Get GQL patterns from model config
            from glyphh.gql import NLTranslator, DEFAULT_GQL_PATTERNS
            from shared.encoder_config_factory import EncoderConfigFactory
            
            # Try to get custom patterns from model
            patterns = DEFAULT_GQL_PATTERNS
            encoder_config = EncoderConfigFactory.extract_encoder_config(loaded_model.sdk_model)
            
            if encoder_config and hasattr(encoder_config, 'gql_patterns') and encoder_config.gql_patterns:
                custom_patterns = encoder_config.gql_patterns.to_gql_patterns()
                if custom_patterns:
                    patterns = custom_patterns
            
            # Create translator
            translator = NLTranslator(
                patterns=patterns,
                dimension=encoder_config.dimension if encoder_config else 10000,
                seed=encoder_config.seed if encoder_config else 42,
            )
            
            # Translate query
            result = translator.translate(query)
            
            return {
                "success": result.success,
                "gql": result.gql,
                "pattern_name": result.pattern_name,
                "confidence": result.confidence,
                "extracted_slots": result.extracted_slots,
                "error": result.error,
                "query_type": "gql_translate",
                "match_method": "hdc_intent" if result.success else "none",
            }
            
        except Exception as e:
            logger.error(f"GQL translate error: {e}", exc_info=True)
            return {
                "success": False,
                "gql": None,
                "error": str(e),
                "query_type": "gql_translate",
                "match_method": "none",
                "confidence": 0.0,
            }
