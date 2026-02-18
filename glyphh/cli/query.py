"""
Query Command for GQL/NL queries against local or remote models.

Supports:
- Local queries against .glyphh model files
- Remote queries against deployed runtime models
- Natural language queries with --nl flag
- Multiple output formats (json, table, csv)
- Stdin piping for scripted queries

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""

import json
import sys
from typing import Optional

import click


@click.command()
@click.argument("query_text", required=False)
@click.option(
    "--model", "-m",
    help="Path to local .glyphh model file",
    type=click.Path(exists=True),
)
@click.option(
    "--runtime", "-r",
    help="Runtime URL for remote queries (e.g., http://localhost:8002)",
)
@click.option(
    "--org", "-o",
    help="Organization ID (required for remote queries)",
)
@click.option(
    "--model-id",
    help="Model ID (required for remote queries)",
)
@click.option(
    "--nl", is_flag=True,
    help="Treat query as natural language (translate to GQL first)",
)
@click.option(
    "--procedure", "-p",
    help="Execute a stored procedure by name (Requirement 9.1)",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "table", "csv"]),
    default="json",
    help="Output format (default: json)",
)
@click.option(
    "--limit", "-l",
    type=int,
    default=10,
    help="Maximum results (default: 10)",
)
@click.option(
    "--timeout", "-t",
    type=int,
    default=30,
    help="Request timeout in seconds (default: 30)",
)
def query(
    query_text: Optional[str],
    model: Optional[str],
    runtime: Optional[str],
    org: Optional[str],
    model_id: Optional[str],
    nl: bool,
    procedure: Optional[str],
    format: str,
    limit: int,
    timeout: int,
):
    """
    Execute GQL or NL queries against local or remote models.
    
    \b
    EXAMPLES:
      # Query local model with GQL
      glyphh query "FIND SIMILAR TO 'cars'" --model ./my-model.glyphh
      
      # Query local model with NL
      glyphh query "find similar to cars" --nl --model ./my-model.glyphh
      
      # Query remote runtime
      glyphh query "LIST ALL" --runtime http://localhost:8002 --org my-org --model-id my-model
      
      # Execute a stored procedure by name (Requirement 9.1)
      glyphh query --procedure find_cars --runtime http://localhost:8002 --org my-org --model-id my-model
      
      # Pipe query from stdin
      echo "COUNT ALL" | glyphh query --model ./my-model.glyphh
      
      # Output as table
      glyphh query "LIST ALL LIMIT 5" --model ./my-model.glyphh --format table
    
    \b
    QUERY TYPES:
      GQL (default): Direct GQL syntax like "FIND SIMILAR TO 'x'"
      NL (--nl):     Natural language like "find similar to x"
      Procedure:     Execute stored procedure by name with --procedure
    """
    # Handle procedure execution (Requirement 9.1)
    if procedure:
        if not runtime:
            raise click.UsageError("--runtime is required for procedure execution")
        if not org or not model_id:
            raise click.UsageError("--org and --model-id are required for procedure execution")
        
        try:
            result = _execute_procedure_remote(runtime, org, model_id, procedure, timeout)
            output = _format_result(result, format)
            click.echo(output)
            if not result.get("success", True):
                sys.exit(1)
            return
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    # Read from stdin if no query provided
    if not query_text:
        if not sys.stdin.isatty():
            query_text = sys.stdin.read().strip()
        else:
            raise click.UsageError(
                "Query required. Provide as argument or pipe from stdin.\n"
                "Example: glyphh query \"FIND SIMILAR TO 'cars'\" --model ./model.glyphh"
            )
    
    if not query_text:
        raise click.UsageError("Empty query provided")
    
    # Validate options
    if model and runtime:
        raise click.UsageError("Cannot specify both --model (local) and --runtime (remote)")
    
    if not model and not runtime:
        raise click.UsageError(
            "Must specify either --model (local) or --runtime (remote)\n"
            "Example: glyphh query \"LIST ALL\" --model ./model.glyphh"
        )
    
    if runtime and (not org or not model_id):
        raise click.UsageError(
            "--org and --model-id are required for remote queries\n"
            "Example: glyphh query \"LIST ALL\" --runtime http://localhost:8002 --org my-org --model-id my-model"
        )
    
    # Execute query
    try:
        if model:
            result = _query_local(query_text, model, nl, limit)
        else:
            result = _query_remote(query_text, runtime, org, model_id, nl, limit, timeout)
        
        # Format and output
        output = _format_result(result, format)
        click.echo(output)
        
        # Exit with error code if query failed
        if not result.get("success", True):
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _query_local(
    query_text: str,
    model_path: str,
    nl: bool,
    limit: int,
) -> dict:
    """Execute query against local .glyphh model file."""
    try:
        from glyphh.model import GlyphhModel
    except ImportError:
        return {"success": False, "error": "glyphh SDK not installed"}
    
    # Load model
    try:
        model = GlyphhModel.load(model_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}
    
    # Translate NL to GQL if needed
    gql_query = query_text
    translation_info = None
    
    if nl:
        try:
            from glyphh.gql import NLTranslator
            from glyphh.gql.patterns import DEFAULT_GQL_PATTERNS
            
            translator = NLTranslator(
                patterns=DEFAULT_GQL_PATTERNS,
                dimension=10000,
                seed=42,
            )
            
            result = translator.translate(query_text)
            
            if not result.success:
                return {
                    "success": False,
                    "error": f"NL translation failed: {result.error or 'No matching pattern'}",
                    "original_query": query_text,
                }
            
            gql_query = result.gql
            translation_info = {
                "pattern": result.pattern_name,
                "confidence": result.confidence,
                "slots": result.extracted_slots,
            }
            
        except ImportError:
            return {"success": False, "error": "NL translation not available (missing glyphh.gql)"}
        except Exception as e:
            return {"success": False, "error": f"NL translation error: {e}"}
    
    # Execute GQL query
    try:
        from glyphh.gql import GQLExecutor, ExecutionContext
        
        # Build execution context
        glyphs = {}
        if hasattr(model, 'glyphs'):
            for glyph in model.glyphs:
                glyphs[glyph.identifier] = glyph
        
        context = ExecutionContext(
            model=model,
            glyphs=glyphs,
            encoder=getattr(model, 'encoder', None),
            similarity_calculator=getattr(model, 'similarity_calculator', None),
        )
        
        executor = GQLExecutor(context=context)
        fact_tree = executor.execute(gql_query)
        
        # Convert result
        result_data = fact_tree.to_dict() if hasattr(fact_tree, 'to_dict') else str(fact_tree)
        
        response = {
            "success": True,
            "query": gql_query,
            "result": result_data,
        }
        
        if translation_info:
            response["translation"] = translation_info
        
        return response
        
    except ImportError:
        return {"success": False, "error": "GQL executor not available (missing glyphh.gql)"}
    except Exception as e:
        return {"success": False, "error": f"Query execution error: {e}", "query": gql_query}


def _query_remote(
    query_text: str,
    runtime_url: str,
    org_id: str,
    model_id: str,
    nl: bool,
    limit: int,
    timeout: int,
) -> dict:
    """Execute query against remote runtime."""
    try:
        import httpx
    except ImportError:
        return {"success": False, "error": "httpx not installed (pip install httpx)"}
    
    # Build URL
    url = f"{runtime_url.rstrip('/')}/{org_id}/{model_id}/mcp"
    
    # Choose tool based on query type
    tool = "nl_query" if nl else "gql_query"
    
    # Build request
    payload = {
        "tool": tool,
        "arguments": {
            "org_id": org_id,
            "model_id": model_id,
            "query": query_text,
        },
    }
    
    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=float(timeout),
            headers={"Content-Type": "application/json"},
        )
        
        if response.status_code == 401:
            return {"success": False, "error": "Authentication required"}
        
        if response.status_code == 403:
            return {"success": False, "error": "Not authorized to access this model"}
        
        if response.status_code == 404:
            return {"success": False, "error": "Model not found or not deployed"}
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Request failed with status {response.status_code}",
            }
        
        data = response.json()
        
        if data.get("isError"):
            return {
                "success": False,
                "error": data.get("error", "Unknown error"),
                "query": query_text,
            }
        
        return {
            "success": True,
            "query": query_text,
            "result": data.get("result"),
            "confidence": data.get("confidence"),
            "query_type": data.get("query_type"),
            "match_method": data.get("match_method"),
        }
        
    except httpx.TimeoutException:
        return {"success": False, "error": f"Request timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"success": False, "error": f"Could not connect to {runtime_url}"}
    except Exception as e:
        return {"success": False, "error": f"Request error: {e}"}


def _execute_procedure_remote(
    runtime_url: str,
    org_id: str,
    model_id: str,
    procedure_name: str,
    timeout: int,
) -> dict:
    """
    Execute a stored procedure by name (Requirement 9.1).
    
    Fetches the procedure's GQL query and executes it.
    """
    try:
        import httpx
    except ImportError:
        return {"success": False, "error": "httpx not installed (pip install httpx)"}
    
    # First, get the procedure to retrieve its GQL query
    procedure_url = f"{runtime_url.rstrip('/')}/{org_id}/{model_id}/procedures/{procedure_name}"
    
    try:
        # Get procedure
        response = httpx.get(procedure_url, timeout=float(timeout))
        
        if response.status_code == 404:
            return {"success": False, "error": f"Procedure '{procedure_name}' not found"}
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to get procedure: {response.status_code}",
            }
        
        procedure_data = response.json()
        gql_query = procedure_data.get("gql_query")
        
        if not gql_query:
            return {"success": False, "error": "Procedure has no GQL query"}
        
        # Execute the GQL query
        mcp_url = f"{runtime_url.rstrip('/')}/{org_id}/{model_id}/mcp"
        
        payload = {
            "tool": "gql_query",
            "arguments": {
                "org_id": org_id,
                "model_id": model_id,
                "query": gql_query,
            },
        }
        
        response = httpx.post(
            mcp_url,
            json=payload,
            timeout=float(timeout),
            headers={"Content-Type": "application/json"},
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Query execution failed: {response.status_code}",
            }
        
        data = response.json()
        
        if data.get("isError"):
            return {
                "success": False,
                "error": data.get("error", "Unknown error"),
                "procedure": procedure_name,
                "query": gql_query,
            }
        
        return {
            "success": True,
            "procedure": procedure_name,
            "query": gql_query,
            "result": data.get("result"),
            "query_type": "procedure",
            "match_method": "stored_procedure",
        }
        
    except httpx.TimeoutException:
        return {"success": False, "error": f"Request timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"success": False, "error": f"Could not connect to {runtime_url}"}
    except Exception as e:
        return {"success": False, "error": f"Request error: {e}"}


def _format_result(result: dict, format: str) -> str:
    """Format result for output."""
    if format == "json":
        return json.dumps(result, indent=2, default=str)
    
    if not result.get("success"):
        return f"Error: {result.get('error', 'Unknown error')}"
    
    data = result.get("result", {})
    
    if format == "table":
        return _format_table(data)
    
    if format == "csv":
        return _format_csv(data)
    
    return json.dumps(result, indent=2, default=str)


def _format_table(data) -> str:
    """Format result as tab-separated table."""
    if isinstance(data, dict):
        # Handle matches array
        if "matches" in data:
            lines = ["ID\tScore\tConcept"]
            for match in data["matches"]:
                glyph_id = match.get("id", match.get("glyph_id", "N/A"))
                score = match.get("score", match.get("similarity", 0))
                concept = str(match.get("concept", match.get("text", "")))[:50]
                lines.append(f"{glyph_id}\t{score:.3f}\t{concept}")
            return "\n".join(lines)
        
        # Handle glyphs array
        if "glyphs" in data:
            lines = ["ID\tConcept"]
            for glyph in data["glyphs"]:
                glyph_id = glyph.get("id", glyph.get("identifier", "N/A"))
                concept = str(glyph.get("concept", glyph.get("text", "")))[:60]
                lines.append(f"{glyph_id}\t{concept}")
            return "\n".join(lines)
        
        # Handle count result
        if "count" in data:
            return f"Count: {data['count']}"
        
        # Generic dict
        lines = ["Key\tValue"]
        for k, v in data.items():
            lines.append(f"{k}\t{v}")
        return "\n".join(lines)
    
    if isinstance(data, list):
        if not data:
            return "(empty)"
        
        # Assume list of dicts
        if isinstance(data[0], dict):
            keys = list(data[0].keys())
            lines = ["\t".join(keys)]
            for item in data:
                values = [str(item.get(k, ""))[:40] for k in keys]
                lines.append("\t".join(values))
            return "\n".join(lines)
        
        # List of primitives
        return "\n".join(str(item) for item in data)
    
    return str(data)


def _format_csv(data) -> str:
    """Format result as CSV."""
    def escape_csv(value) -> str:
        """Escape value for CSV."""
        s = str(value).replace('"', '""')
        if ',' in s or '"' in s or '\n' in s:
            return f'"{s}"'
        return s
    
    if isinstance(data, dict):
        # Handle matches array
        if "matches" in data:
            lines = ["id,score,concept"]
            for match in data["matches"]:
                glyph_id = match.get("id", match.get("glyph_id", ""))
                score = match.get("score", match.get("similarity", 0))
                concept = match.get("concept", match.get("text", ""))
                lines.append(f"{escape_csv(glyph_id)},{score:.3f},{escape_csv(concept)}")
            return "\n".join(lines)
        
        # Handle glyphs array
        if "glyphs" in data:
            lines = ["id,concept"]
            for glyph in data["glyphs"]:
                glyph_id = glyph.get("id", glyph.get("identifier", ""))
                concept = glyph.get("concept", glyph.get("text", ""))
                lines.append(f"{escape_csv(glyph_id)},{escape_csv(concept)}")
            return "\n".join(lines)
        
        # Handle count
        if "count" in data:
            return f"count\n{data['count']}"
        
        # Generic dict
        lines = ["key,value"]
        for k, v in data.items():
            lines.append(f"{escape_csv(k)},{escape_csv(v)}")
        return "\n".join(lines)
    
    if isinstance(data, list):
        if not data:
            return ""
        
        # List of dicts
        if isinstance(data[0], dict):
            keys = list(data[0].keys())
            lines = [",".join(escape_csv(k) for k in keys)]
            for item in data:
                values = [escape_csv(item.get(k, "")) for k in keys]
                lines.append(",".join(values))
            return "\n".join(lines)
        
        # List of primitives
        lines = ["value"]
        for item in data:
            lines.append(escape_csv(item))
        return "\n".join(lines)
    
    return str(data)
