"""
CLI catalog commands — platform model catalog.

catalog list         Browse available models on the platform
catalog search       Search models by name/category
catalog download     Download a .glyphh model from the catalog
catalog info         Show details about a catalog model
"""

import click
from pathlib import Path

from .. import theme
from ..auth import get_token, get_api_url, is_logged_in


@click.group("catalog")
def catalog_group():
    """Platform model catalog."""
    pass


@catalog_group.command("list")
@click.option("--category", "-c", help="Filter by category")
@click.option("--page", default=1, help="Page number")
def catalog_list(category, page):
    """Browse available models on the platform."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
        return

    token = get_token()
    api_url = get_api_url()

    try:
        import httpx

        params = {"page": page, "page_size": 20}
        if category:
            params["category"] = category

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{api_url}/models/",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )

        if res.status_code != 200:
            click.secho(f"  Failed to fetch catalog: {res.text}", fg=theme.ERROR)
            return

        data = res.json()
        items = data.get("items", [])

        if not items:
            click.secho("  No models found in catalog.", fg=theme.MUTED)
            return

        click.echo()
        header = f"  {'NAME':<24} {'STATUS':<12} {'CATEGORY':<14} DESCRIPTION"
        click.secho(header, fg=theme.TEXT_DIM)
        click.secho("  " + "─" * 80, fg=theme.TEXT_DIM)

        for m in items:
            name = (m.get("name") or m.get("id", ""))[:22]
            status = (m.get("status") or "—")[:10]
            cat = (m.get("category") or "—")[:12]
            desc = (m.get("description") or "")[:30]

            status_color = theme.SUCCESS if status == "deployed" else theme.WARNING if status == "draft" else theme.MUTED

            click.echo(
                click.style(f"  {name:<24} ", fg=theme.TEXT)
                + click.style(f"{status:<12} ", fg=status_color)
                + click.style(f"{cat:<14} ", fg=theme.INFO)
                + click.style(desc, fg=theme.MUTED)
            )

        total = data.get("total", len(items))
        pages = data.get("pages", 1)
        click.echo()
        click.secho(f"  Page {page}/{pages} ({total} total)", fg=theme.TEXT_DIM)
        click.echo()

    except Exception as e:
        click.secho(f"  Could not reach catalog: {e}", fg=theme.ERROR)


@catalog_group.command("search")
@click.argument("query")
def catalog_search(query):
    """Search models by name or category."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
        return

    token = get_token()
    api_url = get_api_url()

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{api_url}/models/",
                params={"search": query, "page_size": 20},
                headers={"Authorization": f"Bearer {token}"},
            )

        if res.status_code != 200:
            click.secho(f"  Search failed: {res.text}", fg=theme.ERROR)
            return

        items = res.json().get("items", [])
        if not items:
            click.secho(f"  No models matching '{query}'.", fg=theme.MUTED)
            return

        click.echo()
        for m in items:
            name = m.get("name") or m.get("id", "")
            desc = m.get("description") or ""
            cat = m.get("category") or ""
            click.echo(
                click.style(f"  {name}", fg=theme.TEXT)
                + (click.style(f"  [{cat}]", fg=theme.INFO) if cat else "")
            )
            if desc:
                click.secho(f"    {desc}", fg=theme.MUTED)
        click.echo()

    except Exception as e:
        click.secho(f"  Search failed: {e}", fg=theme.ERROR)


@catalog_group.command("download")
@click.argument("model_name")
@click.option("--dir", "-d", "dest_dir", default=".", type=click.Path(), help="Download destination")
def catalog_download(model_name, dest_dir):
    """Download a model from the platform catalog."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
        return

    token = get_token()
    api_url = get_api_url()

    try:
        import httpx

        click.secho(f"  Downloading {model_name}...", fg=theme.MUTED)

        with httpx.Client(timeout=120) as client:
            res = client.get(
                f"{api_url}/catalog/{model_name}/download",
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=True,
            )

        if res.status_code == 404:
            click.secho(f"  Model '{model_name}' not found in catalog.", fg=theme.WARNING)
            return

        if res.status_code != 200:
            click.secho(f"  Download failed: {res.text}", fg=theme.ERROR)
            return

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        out_file = dest / f"{model_name}.glyphh"
        out_file.write_bytes(res.content)

        size_kb = len(res.content) / 1024
        click.secho(f"  ✓ {out_file.name} ({size_kb:.1f} KB)", fg=theme.SUCCESS)
        click.secho(f"    Deploy with: model deploy {out_file}", fg=theme.TEXT_DIM)

    except Exception as e:
        click.secho(f"  Download failed: {e}", fg=theme.ERROR)


@catalog_group.command("info")
@click.argument("model_name")
def catalog_info(model_name):
    """Show details about a catalog model."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
        return

    token = get_token()
    api_url = get_api_url()

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{api_url}/catalog/{model_name}",
                headers={"Authorization": f"Bearer {token}"},
            )

        if res.status_code == 404:
            click.secho(f"  Model '{model_name}' not found.", fg=theme.WARNING)
            return

        if res.status_code != 200:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)
            return

        m = res.json()
        click.echo()
        click.secho(f"  {m.get('name', model_name)}", fg=theme.TEXT)
        if m.get("description"):
            click.secho(f"  {m['description']}", fg=theme.MUTED)
        click.echo()
        if m.get("version"):
            click.secho(f"  version:  {m['version']}", fg=theme.TEXT_DIM)
        if m.get("author"):
            click.secho(f"  author:   {m['author']}", fg=theme.TEXT_DIM)
        if m.get("category"):
            click.secho(f"  category: {m['category']}", fg=theme.INFO)
        click.echo()

    except Exception as e:
        click.secho(f"  Could not fetch info: {e}", fg=theme.ERROR)


# ── Handler for interactive shell ──

def handle_catalog(func: str | None, args: str = ""):
    """Route catalog subcommands from the interactive shell."""
    if func == "list":
        ctx = click.Context(catalog_list)
        catalog_list.invoke(ctx, category=None, page=1)
    elif func == "search":
        if not args.strip():
            click.secho("  usage: catalog search <query>", fg=theme.MUTED)
            return
        ctx = click.Context(catalog_search)
        catalog_search.invoke(ctx, query=args.strip())
    elif func == "download":
        if not args.strip():
            click.secho("  usage: catalog download <model-name> [--dir path]", fg=theme.MUTED)
            return
        ctx = click.Context(catalog_download)
        catalog_download.invoke(ctx, model_name=args.strip(), dest_dir=".")
    elif func == "info":
        if not args.strip():
            click.secho("  usage: catalog info <model-name>", fg=theme.MUTED)
            return
        ctx = click.Context(catalog_info)
        catalog_info.invoke(ctx, model_name=args.strip())
    else:
        click.echo()
        click.secho("  usage:", fg=theme.MUTED)
        click.secho("    catalog list                Browse platform models", fg=theme.MUTED)
        click.secho("    catalog search <query>      Search by name/category", fg=theme.MUTED)
        click.secho("    catalog download <name>     Download .glyphh model", fg=theme.MUTED)
        click.secho("    catalog info <name>         Show model details", fg=theme.MUTED)
        click.echo()
