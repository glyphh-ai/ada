"""
Load secrets from vault into os.environ and provide setup commands.

Ported from ada's CLI — secrets are stored encrypted in ~/.glyphh/vault
rather than in plaintext .env files.
"""

import getpass
import os

import click

from . import theme

# ANSI shortcuts for direct print() calls (setup prompts don't use click)
G = "\033[32m"
B = "\033[94m"
D = "\033[90m"
P = "\033[95m"
PINK = "\033[38;5;211m"
R = "\033[0m"
BOLD = "\033[1m"
W = "\033[97m"

_VAULT_TO_ENV = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "pipedream_client_id": "PIPEDREAM_CLIENT_ID",
    "pipedream_client_secret": "PIPEDREAM_CLIENT_SECRET",
    "pipedream_project_id": "PIPEDREAM_PROJECT_ID",
}


def load_vault_env() -> None:
    """Load secrets from vault into os.environ."""
    try:
        from domains.brain.skills.vault import get_vault
        vault = get_vault()

        for vault_key, env_key in _VAULT_TO_ENV.items():
            val = vault.get(vault_key)
            if val:
                os.environ.setdefault(env_key, val)
    except Exception:
        pass


def require_api_key() -> bool:
    """Check for API key. If missing, prompt to set it. Returns True if key is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True

    print(f"\n  {PINK}Ada needs an Anthropic API key to start.{R}")
    print(f"  {D}Get one at:{R} {B}https://console.anthropic.com/settings/keys{R}\n")
    setup_key(quiet=True)

    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def setup_key(quiet: bool = False) -> None:
    """Set or update the Anthropic API key."""
    from domains.brain.skills.vault import get_vault

    vault = get_vault()
    current = os.environ.get("ANTHROPIC_API_KEY") or vault.get("anthropic_api_key") or ""

    if current:
        masked = current[:8] + "..." + current[-4:]
        print(f"\n  {D}Current key:{R} {masked}")
        print(f"  {D}Enter new key or press Enter to keep:{R}")
    elif not quiet:
        print(f"\n  {D}No API key set. Ada needs this for her internal LLM.{R}")
        print(f"  {D}Get one at:{R} {B}https://console.anthropic.com/settings/keys{R}")
        print()

    try:
        key = getpass.getpass(f"  {P}API key{D}>{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not key:
        if current:
            print(f"  {D}Keeping existing key.{R}\n")
        else:
            print(f"  {D}No key set.{R}\n")
        return

    if not key.startswith("sk-"):
        print(f"  {PINK}Doesn't look like an Anthropic key (should start with sk-){R}\n")
        return

    vault.set("anthropic_api_key", key)
    os.environ["ANTHROPIC_API_KEY"] = key
    print(f"  {G}Key encrypted and saved.{R}")
    print(f"  {D}Restart Ada to activate.{R}\n")


def setup_model() -> None:
    """Change which LLM model Ada uses internally."""
    env_path = _get_env_path()
    current = os.environ.get("ADA_MODEL", "claude-haiku-4-5-20251001")

    models = [
        ("claude-haiku-4-5-20251001", "Haiku 4.5", f"{G}fast, cheap — recommended{R}"),
        ("claude-sonnet-4-6", "Sonnet 4.6", "smarter, slower"),
    ]

    print(f"\n  {D}Current model:{R} {current}\n")
    for i, (model_id, name, note) in enumerate(models, 1):
        marker = f" {G}●{R}" if model_id == current else "  "
        print(f"  {marker} {i}. {B}{name}{R}  {D}{model_id}{R}  {note}")
    print()

    try:
        choice = input(f"  {P}Choice (1-{len(models)}){D}>{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not choice:
        print(f"  {D}No change.{R}\n")
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            chosen = models[idx][0]
            os.environ["ADA_MODEL"] = chosen
            # Persist to env file
            _set_env_var("ADA_MODEL", chosen)
            print(f"  {G}Model set to {chosen}.{R}")
            print(f"  {D}Restart Ada to activate.{R}\n")
        else:
            print(f"  {PINK}Invalid choice.{R}\n")
    except ValueError:
        print(f"  {PINK}Invalid choice.{R}\n")


def setup_claude_code(port: int = 8002) -> None:
    """Auto-configure Claude Code to use Ada as an MCP server."""
    import json
    from pathlib import Path

    url = f"http://localhost:{port}/mcp"

    # Claude Code config location
    config_path = Path.home() / ".claude" / "settings.json"

    try:
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            config = {}

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        config["mcpServers"]["ada"] = {
            "transport": "http",
            "url": url,
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))
        print(f"\n  {G}Claude Code configured.{R}")
        print(f"  {D}Ada MCP server added to {config_path}{R}\n")
    except Exception as e:
        print(f"\n  {PINK}Failed: {e}{R}\n")


def _get_env_path() -> str:
    """Path to Ada's env file."""
    from pathlib import Path
    home = Path.home() / ".glyphh"
    home.mkdir(exist_ok=True)
    return str(home / "env")


def _set_env_var(key: str, value: str) -> None:
    """Persist an env var to ~/.glyphh/env."""
    env_path = _get_env_path()
    lines = []
    found = False

    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)
