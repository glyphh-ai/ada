"""
Discovery CLI commands - models, hub, docs, demo
Browse deployed models, marketplace, documentation, and run demos.
"""

import click
import requests
import random
from typing import Optional, List
from .auth import is_logged_in, get_current_user, get_api_url, get_auth_headers
from .ui import print_error, print_warning, print_connection_error, print_auth_error


# Prompt suggestions for discovery
SUGGESTIONS = {
    "not_logged_in": [
        ("auth login", "Login to your account"),
        ("auth signup", "Create a new account"),
        ("hub", "Browse the model marketplace"),
        ("docs quickstart", "Get started in 5 minutes"),
        ("demo", "Try an interactive demo"),
    ],
    "logged_in_no_models": [
        ("build init my_model", "Create your first model"),
        ("hub --featured", "Browse featured models"),
        ("docs quickstart", "Follow the quickstart guide"),
        ("demo", "Try the interactive demo"),
    ],
    "logged_in_has_models": [
        ("query \"FIND SIMILAR TO 'search term'\" --model <name>", "Query a deployed model"),
        ("runtime status <model>", "Check model status"),
        ("runtime logs <model>", "View model logs"),
        ("hub", "Discover more models"),
        ("build init new_model", "Create another model"),
    ],
    "after_hub": [
        ("hub --category search", "Browse search models"),
        ("hub --category support", "Browse support models"),
        ("hub --featured", "See featured models"),
        ("docs quickstart", "Learn how to get started"),
        ("auth signup", "Create an account to install models"),
    ],
    "after_docs": [
        ("docs gql", "Learn the query language"),
        ("docs sdk", "Explore the Python SDK"),
        ("demo", "Try it hands-on"),
        ("build init my_model", "Start building"),
    ],
    "after_demo": [
        ("build init my_model", "Create your own model"),
        ("docs quickstart", "Read the full guide"),
        ("hub --category starter", "Browse starter templates"),
        ("auth signup", "Create an account"),
    ],
}


def print_suggestions(context: str, count: int = 3):
    """Print prompt suggestions based on context."""
    suggestions = SUGGESTIONS.get(context, [])
    if not suggestions:
        return
    
    # Pick random suggestions (but keep first one if it's important)
    if len(suggestions) > count:
        selected = random.sample(suggestions, count)
    else:
        selected = suggestions
    
    click.echo()
    click.secho("  Try next:", fg="bright_black")
    for cmd, desc in selected:
        click.echo(f"    ", nl=False)
        click.secho(f"> {cmd}", fg="cyan", nl=False)
        click.secho(f"  {desc}", fg="bright_black")
    click.echo()


@click.command()
@click.option("--org", "-o", help="Filter by organization")
@click.option("--status", "-s", type=click.Choice(["active", "inactive", "all"]), default="active")
def models(org: Optional[str], status: str):
    """
    List your deployed models.
    
    Shows all models deployed to the Glyphh runtime that you have access to.
    """
    if not is_logged_in():
        print_auth_error("list deployed models")
        print_suggestions("not_logged_in")
        return
    
    api_url = get_api_url()
    headers = get_auth_headers()
    
    click.echo("Fetching deployed models...")
    
    try:
        params = {}
        if org:
            params["org_id"] = org
        if status != "all":
            params["status"] = status
            
        response = requests.get(
            f"{api_url}/api/v1/models",
            headers=headers,
            params=params,
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            models_list = data.get("models", [])
            
            if not models_list:
                click.echo("\nNo deployed models found.")
                click.echo("Deploy a model with: runtime deploy <package.glyphh>")
                print_suggestions("logged_in_no_models")
                return
            
            click.echo()
            click.secho(f"  {'NAME':<25} {'STATUS':<12} {'VERSION':<10} {'DEPLOYED':<20}", fg="cyan")
            click.secho("  " + "-" * 70, fg="bright_black")
            
            for model in models_list:
                name = model.get("name", "unknown")[:24]
                model_status = model.get("status", "unknown")
                version = model.get("version", "-")[:9]
                deployed = model.get("deployed_at", "-")[:19]
                
                status_color = "green" if model_status == "active" else "yellow"
                click.echo(f"  {name:<25} ", nl=False)
                click.secho(f"{model_status:<12}", fg=status_color, nl=False)
                click.echo(f" {version:<10} {deployed:<20}")
            
            click.echo()
            click.echo(f"  Total: {len(models_list)} model(s)")
            print_suggestions("logged_in_has_models")
            
        elif response.status_code == 401:
            print_error(
                "Session Expired",
                "Your session has expired. Please log in again.",
                suggestions=["Run 'auth login' to re-authenticate"]
            )
            print_suggestions("not_logged_in")
        else:
            error = response.json().get("detail", "Failed to fetch models")
            print_error("API Error", error)
            
    except requests.exceptions.ConnectionError:
        print_connection_error(api_url)
        print_suggestions("logged_in_no_models")
    except Exception as e:
        print_error("Unexpected Error", str(e))


@click.command()
@click.option("--category", "-c", type=click.Choice(["search", "support", "sales", "knowledge", "compliance", "starter", "all"]), default="all")
@click.option("--search", "-s", help="Search by name or description")
@click.option("--featured", "-f", is_flag=True, help="Show only featured models")
def hub(category: str, search: Optional[str], featured: bool):
    """
    Browse the Glyphh model marketplace.
    
    Discover pre-built models for common use cases like search,
    customer support, sales intelligence, and more.
    """
    api_url = get_api_url()
    
    click.echo("Fetching models from Glyphh Hub...")
    
    try:
        params = {}
        if category != "all":
            params["category"] = category
        if search:
            params["search"] = search
        if featured:
            params["featured"] = "true"
            
        response = requests.get(
            f"{api_url}/api/v1/hub/models",
            params=params,
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            models_list = data.get("models", [])
            
            if not models_list:
                click.echo("\nNo models found matching your criteria.")
                print_suggestions("after_hub")
                return
            
            click.echo()
            click.secho(f"  {'NAME':<30} {'CATEGORY':<12} {'DOWNLOADS':<10} {'FEATURED'}", fg="cyan")
            click.secho("  " + "-" * 70, fg="bright_black")
            
            for model in models_list:
                name = model.get("name", "unknown")[:29]
                cat = model.get("category", "-")[:11]
                downloads = str(model.get("downloads", 0))[:9]
                is_featured = "★" if model.get("featured") else ""
                
                click.echo(f"  {name:<30} {cat:<12} {downloads:<10} ", nl=False)
                if is_featured:
                    click.secho(is_featured, fg="yellow")
                else:
                    click.echo()
            
            click.echo()
            click.echo(f"  Total: {len(models_list)} model(s)")
            click.echo()
            click.echo("  Use 'hub --category <cat>' to filter by category")
            click.echo("  Visit https://glyphh.ai/hub for the full experience")
            print_suggestions("after_hub")
            
        else:
            # Fallback to showing categories if API not available
            _show_hub_offline()
            
    except requests.exceptions.ConnectionError:
        _show_hub_offline()
    except Exception as e:
        print_error("Unexpected Error", str(e))


def _show_hub_offline():
    """Show hub info when offline or API unavailable."""
    click.echo()
    click.secho("  GLYPHH HUB - Model Marketplace", fg="magenta", bold=True)
    click.echo()
    click.secho("  (Offline mode - showing categories)", fg="yellow")
    click.echo()
    click.secho("  Categories:", fg="cyan")
    click.echo("    search      - Semantic search and retrieval models")
    click.echo("    support     - Customer support and FAQ models")
    click.echo("    sales       - Sales intelligence and lead scoring")
    click.echo("    knowledge   - Knowledge base and documentation")
    click.echo("    compliance  - Regulatory and compliance checking")
    click.echo("    starter     - Getting started templates")
    click.echo()
    click.echo("  Visit https://glyphh.ai/hub to browse all models")
    print_suggestions("after_hub")


@click.command()
@click.argument("topic", required=False)
def docs(topic: Optional[str]):
    """
    View Glyphh documentation.
    
    \b
    TOPICS:
      quickstart   Getting started guide
      gql          GQL query language reference
      sdk          SDK API reference
      runtime      Runtime deployment guide
      cli          CLI command reference
      concepts     Core concepts and architecture
    
    \b
    Examples:
      glyphh docs              # Show documentation overview
      glyphh docs quickstart   # Show quickstart guide
      glyphh docs gql          # Show GQL reference
    """
    click.echo()
    
    if not topic:
        # Show documentation overview
        click.secho("  GLYPHH DOCUMENTATION", fg="magenta", bold=True)
        click.echo()
        click.secho("  Quick Links:", fg="cyan")
        click.echo("    docs quickstart   Getting started in 5 minutes")
        click.echo("    docs gql          GQL query language reference")
        click.echo("    docs sdk          Python SDK API reference")
        click.echo("    docs runtime      Runtime deployment guide")
        click.echo("    docs cli          CLI command reference")
        click.echo("    docs concepts     Core concepts explained")
        click.echo()
        click.echo("  Full documentation: https://docs.glyphh.ai")
        print_suggestions("after_docs")
        return
    
    topic = topic.lower()
    
    docs_content = {
        "quickstart": """
  QUICKSTART GUIDE
  ================
  
  1. Initialize a new model:
     > build init my_model
     
     Creates a new model directory with default configuration.
     Options:
       --template <name>    Use a starter template (text, product, document)
       --dimensions <n>     Vector dimensions (default: 10000)
  
  2. Add concepts from your data:
     > build add my_model --source data.csv --column text
     
     Encodes your data into hyperdimensional vectors.
     Supported sources:
       --source file.csv      CSV file (use --column to specify text column)
       --source file.json     JSON file (use --field for nested paths)
       --source file.txt      Plain text (one concept per line)
       --source directory/    Directory of text files
     
     Options:
       --batch-size <n>     Process in batches (default: 1000)
       --normalize          Normalize vectors after encoding
  
  3. Test similarity:
     > test similarity my_model "search query"
     
     Returns top matches with similarity scores.
     Options:
       --top <n>            Number of results (default: 5)
       --threshold <0-1>    Minimum similarity score
       --explain            Show why matches were selected
  
  4. Package for deployment:
     > package create my_model
     
     Creates a .glyphh package file ready for deployment.
     Options:
       --output <path>      Custom output path
       --compress           Enable compression (smaller file, slower load)
       --include-metadata   Include source metadata in package
  
  5. Deploy to runtime:
     > runtime deploy my_model.glyphh
     
     Deploys your model to the Glyphh runtime.
     Options:
       --name <name>        Custom deployment name
       --replicas <n>       Number of instances (default: 1)
       --region <region>    Deployment region
  
  6. Query your model:
     > query "FIND SIMILAR TO 'cars'" --model my_model
     
     Execute GQL queries against your deployed model.
     See 'docs gql' for full query syntax.

  Type 'docs <topic>' for more details on any topic.
""",
        "gql": """
  GQL - GLYPHH QUERY LANGUAGE
  ===========================
  
  BASIC QUERIES
  -------------
  Find similar items:
    > query "FIND SIMILAR TO 'machine learning'"
    > query "FIND SIMILAR TO 'running shoes' LIMIT 10"
  
  Filter results:
    > query "FIND SIMILAR TO 'laptop' WHERE category = 'electronics'"
    > query "FIND SIMILAR TO 'budget phone' WHERE price < 500"
  
  SIMILARITY THRESHOLD
  --------------------
  Only return high-confidence matches:
    > query "FIND SIMILAR TO 'query' WITH THRESHOLD 0.8"
  
  Threshold guide:
    0.9+   Very high similarity (near duplicates)
    0.7-0.9  Strong similarity (same topic/category)
    0.5-0.7  Moderate similarity (related concepts)
    <0.5   Weak similarity (loosely related)
  
  ORDERING & LIMITS
  -----------------
    > query "FIND SIMILAR TO 'query' ORDER BY similarity DESC LIMIT 20"
    > query "FIND SIMILAR TO 'query' ORDER BY date DESC LIMIT 10"
  
  AGGREGATIONS
  ------------
  Count matches:
    > query "COUNT SIMILAR TO 'error' WHERE status = 'open'"
  
  NATURAL LANGUAGE
  ----------------
  Queries are auto-converted to GQL:
    > query "find products similar to running shoes"
    > query "show me documents about machine learning"
    > query "what's related to customer complaints?"
  
  COMBINING CONDITIONS
  --------------------
    > query "FIND SIMILAR TO 'query' WHERE category = 'tech' AND status = 'active'"
    > query "FIND SIMILAR TO 'query' WHERE price BETWEEN 100 AND 500"
  
  FIELD SELECTION
  ---------------
    > query "FIND SIMILAR TO 'query' RETURN title, score, category"
""",
        "sdk": """
  PYTHON SDK REFERENCE
  ====================
  
  INSTALLATION
  ------------
    pip install glyphh
  
  LOADING MODELS
  --------------
    from glyphh import Model
    
    # Load from local file
    model = Model.load("my_model.glyphh")
    
    # Load from runtime (requires auth)
    model = Model.from_runtime("my_model")
    
    # Load with custom config
    model = Model.load("my_model.glyphh", cache=True, lazy=False)
  
  QUERYING
  --------
    # Basic query
    results = model.query("FIND SIMILAR TO 'search term'")
    
    # With options
    results = model.query(
        "FIND SIMILAR TO 'search'",
        limit=10,
        threshold=0.7,
        include_scores=True
    )
    
    # Natural language
    results = model.search("products like running shoes")
  
  ENCODING
  --------
    # Encode single text
    vector = model.encode("some text")
    
    # Encode batch
    vectors = model.encode_batch(["text1", "text2", "text3"])
    
    # Get raw hypervector
    hv = model.encode("text", as_binary=True)
  
  SIMILARITY
  ----------
    # Compare two texts
    score = model.similarity("text1", "text2")
    
    # Compare text to vector
    score = model.similarity("text", existing_vector)
    
    # Batch comparison
    scores = model.similarity_batch("query", ["doc1", "doc2", "doc3"])
  
  ADDING CONCEPTS
  ---------------
    # Add single concept
    model.add("new concept text", metadata={"category": "tech"})
    
    # Add batch
    model.add_batch(texts, metadata_list)
    
    # Save changes
    model.save("updated_model.glyphh")
  
  ASYNC SUPPORT
  -------------
    import asyncio
    from glyphh import AsyncModel
    
    async def main():
        model = await AsyncModel.load("my_model.glyphh")
        results = await model.query("FIND SIMILAR TO 'search'")
    
    asyncio.run(main())
""",
        "runtime": """
  RUNTIME DEPLOYMENT
  ==================
  
  DEPLOYING
  ---------
  Deploy a packaged model:
    > runtime deploy my_model.glyphh
    
  Deploy with options:
    > runtime deploy my_model.glyphh --name prod-model --replicas 2
    > runtime deploy my_model.glyphh --region us-west-2
  
  STATUS & MONITORING
  -------------------
  Check deployment status:
    > runtime status my_model
    
  Output shows:
    - Status (deploying, running, stopped, error)
    - Replicas (current/desired)
    - Endpoint URL
    - Last updated timestamp
  
  List all deployments:
    > runtime list
  
  LOGS
  ----
  View recent logs:
    > runtime logs my_model
    
  Stream logs in real-time:
    > runtime logs my_model --follow
    
  Filter logs:
    > runtime logs my_model --since 1h
    > runtime logs my_model --level error
  
  SCALING
  -------
  Scale replicas:
    > runtime scale my_model --replicas 3
    
  Auto-scaling (Pro plan):
    > runtime scale my_model --min 1 --max 10 --target-cpu 70
  
  MANAGEMENT
  ----------
  Stop a deployment:
    > runtime stop my_model
    
  Restart:
    > runtime restart my_model
    
  Delete:
    > runtime delete my_model
  
  ENVIRONMENT
  -----------
  Set environment variables:
    > runtime env set my_model KEY=value
    > runtime env list my_model
  
  Configuration:
    GLYPHH_API_URL    API endpoint (default: https://api.glyphh.ai)
    GLYPHH_API_KEY    API key for authentication
""",
        "cli": """
  CLI COMMAND REFERENCE
  =====================
  
  AUTHENTICATION
  --------------
    auth login           Login to your account
    auth login -e EMAIL  Login with email
    auth signup          Create a new account
    auth logout          Logout and clear credentials
    auth whoami          Show current user info
    auth status          Show auth status and API URL
    auth set-api URL     Set custom API URL (self-hosted)
  
  BUILD
  -----
    build init NAME              Create new model
    build init NAME --template   Use starter template
    build add MODEL --source     Add concepts from data
    build validate MODEL         Validate model config
    build info MODEL             Show model details
    build export MODEL           Export model config
  
  TEST
  ----
    test similarity MODEL "query"     Test similarity search
    test similarity MODEL "q" --top 10  Limit results
    test encode MODEL "text"          Test text encoding
    test benchmark MODEL              Run performance benchmark
  
  PACKAGE
  -------
    package create MODEL         Create .glyphh package
    package create MODEL -o PATH Custom output path
    package info FILE.glyphh     Show package details
    package validate FILE.glyphh Validate package integrity
    package extract FILE.glyphh  Extract package contents
  
  RUNTIME
  -------
    runtime deploy FILE.glyphh   Deploy to runtime
    runtime status NAME          Check deployment status
    runtime list                 List all deployments
    runtime logs NAME            View logs
    runtime logs NAME --follow   Stream logs
    runtime scale NAME --replicas N  Scale deployment
    runtime stop NAME            Stop deployment
    runtime restart NAME         Restart deployment
    runtime delete NAME          Delete deployment
  
  QUERY
  -----
    query "GQL QUERY"            Execute GQL query
    query "GQL" --model NAME     Query specific model
    query "GQL" --format json    Output as JSON
    query "GQL" --explain        Show query explanation
  
  HUB
  ---
    hub                          Browse model hub
    hub --featured               Show featured models
    hub search "query"           Search models
    hub info MODEL               Show model details
    hub download MODEL           Download model
  
  DOCS
  ----
    docs                         Show docs overview
    docs quickstart              Quickstart guide
    docs gql                     GQL reference
    docs sdk                     Python SDK reference
    docs runtime                 Runtime guide
    docs concepts                Core concepts
    docs cli                     This reference
  
  GENERAL
  -------
    help                         Show help
    home                         Return to home screen
    clear                        Clear screen
    exit / quit                  Exit the shell
""",
        "concepts": """
  CORE CONCEPTS
  =============
  
  HYPERDIMENSIONAL COMPUTING (HDC)
  --------------------------------
  Glyphh uses high-dimensional binary vectors (10,000+ dimensions)
  to represent concepts. Unlike traditional embeddings:
  
    Traditional ML          Glyphh HDC
    ─────────────────────   ─────────────────────
    Dense float vectors     Sparse binary vectors
    100-1000 dimensions     10,000+ dimensions
    GPU-intensive           CPU-friendly
    Black-box similarity    Interpretable matching
    Slow to update          Instant updates
  
  Key properties:
    • Similarity is measured by Hamming distance (bit comparison)
    • Vectors can be combined with simple binary operations
    • Noise-tolerant: small changes don't affect results
    • No retraining needed to add new concepts
  
  MODELS
  ------
  A Glyphh model contains:
    • Encoded concepts (hypervectors)
    • Vocabulary/tokenization rules
    • Metadata mappings
    • Query configuration
  
  Models are packaged as .glyphh files:
    > build init my_model        # Create model
    > build add my_model ...     # Add concepts
    > package create my_model    # Package for deployment
  
  GQL (GLYPHH QUERY LANGUAGE)
  ---------------------------
  SQL-like syntax for semantic queries:
  
    FIND SIMILAR TO 'query'              Basic search
    FIND SIMILAR TO 'q' LIMIT 10         Limit results
    FIND SIMILAR TO 'q' WHERE x = 'y'    Filter results
    FIND SIMILAR TO 'q' WITH THRESHOLD 0.8  Min similarity
    COUNT SIMILAR TO 'q'                 Count matches
  
  Natural language is auto-converted:
    "find products like running shoes" → FIND SIMILAR TO 'running shoes'
  
  RUNTIME
  -------
  Managed infrastructure for serving models:
    • Auto-scaling based on load
    • Global edge caching
    • Real-time query analytics
    • Zero-downtime deployments
  
  Deploy with:
    > runtime deploy my_model.glyphh
  
  SIMILARITY SCORES
  -----------------
  Scores range from 0.0 to 1.0:
  
    1.0      Identical
    0.9+     Near duplicates
    0.7-0.9  Same topic/category
    0.5-0.7  Related concepts
    0.3-0.5  Loosely related
    <0.3     Unrelated
  
  Use thresholds to filter:
    > query "FIND SIMILAR TO 'x' WITH THRESHOLD 0.7"
""",
    }
    
    if topic in docs_content:
        click.secho(docs_content[topic], fg="white")
        print_suggestions("after_docs")
    else:
        click.secho(f"  Unknown topic: {topic}", fg="red")
        click.echo()
        click.echo("  Available topics: quickstart, gql, sdk, runtime, cli, concepts")
        click.echo()


@click.command()
@click.option("--model", "-m", default="demo", help="Demo model to use")
@click.option("--reel", type=click.Choice(["product", "churn"]), default=None,
              help="Play an animated demo reel (product or churn)")
def demo(model: str, reel: str):
    """
    Run an interactive demo.
    
    Experience Glyphh's capabilities with a pre-built demo model.
    Try semantic search, similarity matching, and GQL queries.
    
    Use --reel to watch an animated walkthrough:
      demo --reel product   Product search flow
      demo --reel churn     Customer churn prediction flow
    """
    if reel == "product":
        from .screens.reel_product import show_reel_product
        show_reel_product()
        return
    elif reel == "churn":
        from .screens.reel_churn import show_reel_churn
        show_reel_churn()
        return

    click.echo()
    click.secho("  GLYPHH INTERACTIVE DEMO", fg="magenta", bold=True)
    click.echo()
    click.secho("  This demo uses a pre-built product catalog model.", fg="white")
    click.echo()
    
    # Demo data
    demo_products = [
        {"name": "Running Shoes Pro", "category": "footwear", "desc": "Lightweight running shoes with cushioned sole"},
        {"name": "Trail Hiking Boots", "category": "footwear", "desc": "Waterproof boots for mountain trails"},
        {"name": "Wireless Earbuds", "category": "electronics", "desc": "Bluetooth earbuds with noise cancellation"},
        {"name": "Smart Watch", "category": "electronics", "desc": "Fitness tracker with heart rate monitor"},
        {"name": "Yoga Mat", "category": "fitness", "desc": "Non-slip exercise mat for yoga and pilates"},
        {"name": "Resistance Bands", "category": "fitness", "desc": "Set of elastic bands for strength training"},
    ]
    
    click.secho("  Sample queries to try:", fg="cyan")
    click.echo("    > FIND SIMILAR TO 'comfortable shoes for exercise'")
    click.echo("    > FIND SIMILAR TO 'music on the go'")
    click.echo("    > FIND SIMILAR TO 'workout equipment'")
    click.echo()
    click.echo("  Type 'exit' to quit the demo.")
    click.echo()
    
    while True:
        try:
            query = input(click.style("  demo> ", fg="magenta")).strip()
            
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            
            # Simple keyword matching for demo
            query_lower = query.lower()
            
            # Extract search term from GQL-like syntax
            if "similar to" in query_lower:
                import re
                match = re.search(r"similar to ['\"](.+?)['\"]", query_lower)
                if match:
                    search_term = match.group(1)
                else:
                    search_term = query_lower.replace("find", "").replace("similar to", "").strip()
            else:
                search_term = query_lower
            
            # Score products based on keyword overlap
            results = []
            search_words = set(search_term.split())
            
            for product in demo_products:
                product_text = f"{product['name']} {product['desc']} {product['category']}".lower()
                product_words = set(product_text.split())
                
                # Simple overlap score
                overlap = len(search_words & product_words)
                
                # Boost for partial matches
                for sw in search_words:
                    for pw in product_words:
                        if sw in pw or pw in sw:
                            overlap += 0.5
                
                if overlap > 0:
                    results.append((product, min(0.95, 0.5 + overlap * 0.15)))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            
            if results:
                click.echo()
                click.secho("  Results:", fg="green")
                for product, score in results[:3]:
                    click.echo(f"    [{score:.2f}] {product['name']}")
                    click.secho(f"           {product['desc']}", fg="bright_black")
                click.echo()
            else:
                click.echo()
                click.secho("  No matching products found.", fg="yellow")
                click.echo()
                
        except KeyboardInterrupt:
            click.echo()
            break
        except EOFError:
            break
    
    click.echo()
    click.secho("  Thanks for trying Glyphh!", fg="cyan")
    click.echo("  Get started: build init my_model")
    click.echo()
