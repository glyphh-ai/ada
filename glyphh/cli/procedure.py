"""
Procedure Command for managing stored procedures.

Supports:
- Listing stored procedures for a model
- Creating new stored procedures
- Executing stored procedures by name

Requirements: 9.1, 9.2, 9.3, 9.4
"""

import json
import sys
from typing import Optional

import click


@click.group()
def procedure():
    """
    Manage stored procedures for deployed models.
    
    \b
    EXAMPLES:
      # List procedures for a model
      glyphh procedure list --runtime http://localhost:8002 --org my-org --model-id my-model
      
      # Create a new procedure
      glyphh procedure create --name find_cars --query "FIND SIMILAR TO 'cars'" \\
        --lexicons "cars,vehicles,automobiles" --runtime http://localhost:8002 \\
        --org my-org --model-id my-model
      
      # Execute a procedure by name
      glyphh query --procedure find_cars --runtime http://localhost:8002 \\
        --org my-org --model-id my-model
    """
    pass


@procedure.command("list")
@click.option(
    "--runtime", "-r",
    required=True,
    help="Runtime URL (e.g., http://localhost:8002)",
)
@click.option(
    "--org", "-o",
    required=True,
    help="Organization ID",
)
@click.option(
    "--model-id", "-m",
    required=True,
    help="Model ID",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "table", "csv"]),
    default="table",
    help="Output format (default: table)",
)
@click.option(
    "--timeout", "-t",
    type=int,
    default=30,
    help="Request timeout in seconds (default: 30)",
)
def list_procedures(
    runtime: str,
    org: str,
    model_id: str,
    format: str,
    timeout: int,
):
    """
    List all stored procedures for a model.
    
    \b
    EXAMPLES:
      glyphh procedure list --runtime http://localhost:8002 --org my-org --model-id my-model
      glyphh procedure list -r http://localhost:8002 -o my-org -m my-model --format json
    """
    try:
        result = _list_procedures_remote(runtime, org, model_id, timeout)
        
        if not result.get("success"):
            click.echo(f"Error: {result.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
        
        output = _format_procedures(result.get("procedures", []), format)
        click.echo(output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@procedure.command("create")
@click.option(
    "--name", "-n",
    required=True,
    help="Procedure name (alphanumeric and underscores, must start with letter)",
)
@click.option(
    "--query", "-q",
    required=True,
    help="GQL query string",
)
@click.option(
    "--lexicons", "-l",
    required=True,
    help="Comma-separated list of lexicons for NL matching",
)
@click.option(
    "--description", "-d",
    default="",
    help="Optional description",
)
@click.option(
    "--runtime", "-r",
    required=True,
    help="Runtime URL (e.g., http://localhost:8002)",
)
@click.option(
    "--org", "-o",
    required=True,
    help="Organization ID",
)
@click.option(
    "--model-id", "-m",
    required=True,
    help="Model ID",
)
@click.option(
    "--timeout", "-t",
    type=int,
    default=30,
    help="Request timeout in seconds (default: 30)",
)
def create_procedure(
    name: str,
    query: str,
    lexicons: str,
    description: str,
    runtime: str,
    org: str,
    model_id: str,
    timeout: int,
):
    """
    Create a new stored procedure.
    
    \b
    EXAMPLES:
      glyphh procedure create --name find_cars --query "FIND SIMILAR TO 'cars'" \\
        --lexicons "cars,vehicles,automobiles" --runtime http://localhost:8002 \\
        --org my-org --model-id my-model
      
      glyphh procedure create -n find_reliable -q "FIND SIMILAR TO 'reliable' WHERE make = 'Toyota'" \\
        -l "reliable,dependable,trustworthy" -d "Find reliable vehicles" \\
        -r http://localhost:8002 -o my-org -m my-model
    """
    # Parse lexicons
    lexicon_list = [l.strip() for l in lexicons.split(",") if l.strip()]
    
    if not lexicon_list:
        click.echo("Error: At least one lexicon is required", err=True)
        sys.exit(1)
    
    try:
        result = _create_procedure_remote(
            runtime, org, model_id, name, query, lexicon_list, description, timeout
        )
        
        if not result.get("success"):
            click.echo(f"Error: {result.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
        
        click.echo(f"Created procedure '{name}' successfully")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _list_procedures_remote(
    runtime_url: str,
    org_id: str,
    model_id: str,
    timeout: int,
) -> dict:
    """List procedures from remote runtime."""
    try:
        import httpx
    except ImportError:
        return {"success": False, "error": "httpx not installed (pip install httpx)"}
    
    url = f"{runtime_url.rstrip('/')}/{org_id}/{model_id}/procedures"
    
    try:
        response = httpx.get(url, timeout=float(timeout))
        
        if response.status_code == 401:
            return {"success": False, "error": "Authentication required"}
        
        if response.status_code == 404:
            return {"success": False, "error": "Model not found or not deployed"}
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Request failed with status {response.status_code}",
            }
        
        data = response.json()
        
        return {
            "success": True,
            "procedures": data.get("procedures", []),
            "total": data.get("total", 0),
        }
        
    except httpx.TimeoutException:
        return {"success": False, "error": f"Request timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"success": False, "error": f"Could not connect to {runtime_url}"}
    except Exception as e:
        return {"success": False, "error": f"Request error: {e}"}


def _create_procedure_remote(
    runtime_url: str,
    org_id: str,
    model_id: str,
    name: str,
    gql_query: str,
    lexicons: list,
    description: str,
    timeout: int,
) -> dict:
    """Create procedure on remote runtime."""
    try:
        import httpx
    except ImportError:
        return {"success": False, "error": "httpx not installed (pip install httpx)"}
    
    url = f"{runtime_url.rstrip('/')}/{org_id}/{model_id}/procedures"
    
    payload = {
        "name": name,
        "gql_query": gql_query,
        "lexicons": lexicons,
        "description": description,
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
        
        if response.status_code == 409:
            return {"success": False, "error": f"Procedure '{name}' already exists"}
        
        if response.status_code == 400:
            data = response.json()
            return {"success": False, "error": data.get("detail", "Invalid request")}
        
        if response.status_code not in (200, 201):
            return {
                "success": False,
                "error": f"Request failed with status {response.status_code}",
            }
        
        return {"success": True}
        
    except httpx.TimeoutException:
        return {"success": False, "error": f"Request timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"success": False, "error": f"Could not connect to {runtime_url}"}
    except Exception as e:
        return {"success": False, "error": f"Request error: {e}"}


def _format_procedures(procedures: list, format: str) -> str:
    """Format procedures for output."""
    if not procedures:
        return "No procedures found"
    
    if format == "json":
        return json.dumps(procedures, indent=2)
    
    if format == "csv":
        lines = ["name,description,lexicons,gql_query"]
        for p in procedures:
            name = p.get("name", "")
            desc = p.get("description", "").replace('"', '""')
            lexicons = ";".join(p.get("lexicons", []))
            query = p.get("gql_query", "").replace('"', '""')
            lines.append(f'"{name}","{desc}","{lexicons}","{query}"')
        return "\n".join(lines)
    
    # Table format (default)
    lines = ["NAME\tDESCRIPTION\tLEXICONS"]
    for p in procedures:
        name = p.get("name", "")
        desc = p.get("description", "")[:30]
        lexicons = ", ".join(p.get("lexicons", [])[:3])
        if len(p.get("lexicons", [])) > 3:
            lexicons += "..."
        lines.append(f"{name}\t{desc}\t{lexicons}")
    return "\n".join(lines)
