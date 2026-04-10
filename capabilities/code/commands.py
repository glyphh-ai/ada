"""
Custom CLI commands for the Ada Code model.

Provides init, compile, and status commands available in the scoped model REPL:
    model model-code> init /path/to/repo
    model model-code> compile /path/to/repo
    model model-code> status

Convention: register(ctx) returns {name: {"handler": fn, "help": str}}.
Each handler receives (args: str, ctx: dict) where ctx has:
    model_id, runtime_url, org_id, token, headers, source_dir
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import click
    from glyphh.cli import theme
except ImportError:
    # Fallback for standalone use
    import click

    class _FallbackTheme:
        PRIMARY = "magenta"
        ACCENT = "bright_magenta"
        MUTED = "bright_black"
        SUCCESS = "green"
        WARNING = "yellow"
        ERROR = "red"
        INFO = "cyan"
        TEXT = "white"
        TEXT_DIM = "bright_black"

    theme = _FallbackTheme()

_MODEL_DIR = Path(__file__).parent
_GLYPHH_DIR = Path.home() / ".ada"
_STATE_FILE = _GLYPHH_DIR / "code.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return ""
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"{bar}  {current}/{total} ({pct:.0%})"


def _hook_cmd(subcommand: str, *args: str) -> str:
    """Build a portable hook command."""
    import shutil
    suffix = " ".join(str(a) for a in args)
    if shutil.which("ada-hook"):
        return f"ada-hook {subcommand} {suffix}".strip()
    # Fall back to running hooks module directly from model source
    hooks_main = _MODEL_DIR / "hooks" / "__init__.py"
    if hooks_main.exists():
        return f"{sys.executable} -c \"import sys; sys.path.insert(0, '{_MODEL_DIR}'); from hooks import main; sys.argv = ['hooks', '{subcommand}'] + '{suffix}'.split(); main()\"".strip()
    return f"{sys.executable} -m hooks {subcommand} {suffix}".strip()


def _deploy_model(ctx: dict, project_dir: Path | None = None) -> bool:
    """Deploy the code model to the runtime via API."""
    try:
        from glyphh.cli.packaging import package_model
        import httpx

        ada_file = package_model(_MODEL_DIR)

        if project_dir:
            import shutil
            dest_dir = project_dir / ".ada"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / ada_file.name
            shutil.copy2(ada_file, dest)
            click.secho(f"  Model stored: {dest}", fg=theme.TEXT_DIM)

        try:
            with httpx.Client(timeout=60) as client:
                with open(ada_file, "rb") as f:
                    r = client.post(
                        f"{ctx['runtime_url']}/{ctx['org_id']}/code/model/deploy",
                        files={"file": (ada_file.name, f, "application/octet-stream")},
                        headers=ctx["headers"],
                    )
            return r.status_code in (200, 201)
        finally:
            ada_file.unlink(missing_ok=True)
    except Exception as e:
        click.secho(f"  Deploy warning: {e}", fg=theme.WARNING)
        return False


def _compile_repo(repo_path: str, ctx: dict) -> tuple[int, list[str]]:
    """Compile the repository into the Ada index."""
    import io

    cmd = [
        sys.executable, str(_MODEL_DIR / "compile.py"), repo_path,
        "--runtime-url", ctx["runtime_url"],
        "--org-id", ctx["org_id"],
    ]
    if ctx.get("token"):
        cmd.extend(["--token", ctx["token"]])

    env = {**__import__("os").environ, "PYTHONWARNINGS": "ignore"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stderr_lines: list[str] = []
    current_phase = ""

    def _read_stderr():
        nonlocal current_phase
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.rstrip("\n")
            stderr_lines.append(line)
            if not line.startswith("PROGRESS:"):
                continue
            parts = line.split(":")
            if len(parts) != 4:
                continue
            phase, cur_s, tot_s = parts[1], parts[2], parts[3]
            try:
                cur, tot = int(cur_s), int(tot_s)
            except ValueError:
                continue
            if phase != current_phase:
                if current_phase:
                    click.echo("\r" + " " * 72 + "\r", nl=False)
                current_phase = phase
                phase_labels = {
                    "extract": "Extracting",
                    "relationships": "Building graph",
                    "upload": "Encoding & uploading",
                }
                label = phase_labels.get(phase, phase.capitalize())
                click.secho(f"         {label}...", fg=theme.TEXT_DIM)
            bar = _render_bar(cur, tot)
            click.echo(f"\r         {bar}", nl=False)
            if cur >= tot:
                click.echo()

    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()

    stdout_buf = io.StringIO()
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_buf.write(line)

    proc.wait()
    t.join(timeout=5)
    stdout = stdout_buf.getvalue()

    if proc.returncode != 0:
        err = "\n".join(stderr_lines[-5:]) if stderr_lines else "(no output)"
        click.secho(f"  Compile error: {err[:200]}", fg=theme.ERROR)
        return 0, []

    file_count = 0
    job_ids = []
    for line in stdout.strip().split("\n"):
        if "files indexed" in line:
            try:
                file_count = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        if "Encoded:" in line:
            try:
                file_count = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        if "\u2192 job " in line:
            try:
                job_id = line.split("\u2192 job ")[1].strip()
                if job_id and job_id != "?":
                    job_ids.append(job_id)
            except (IndexError, ValueError):
                pass
    return file_count, job_ids


def _configure_claude_code(repo_path: str, mcp_url: str, ctx: dict, is_upgrade: bool = False):
    """Configure Claude Code: MCP server, hooks, permissions, rules."""
    repo = Path(repo_path).resolve()
    claude_dir = repo / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    # 1. Add MCP server to Claude Code
    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "--transport", "http", "ada", mcp_url],
            capture_output=True,
            text=True,
            cwd=str(repo),
        )
        if result.returncode == 0:
            click.secho("  + MCP server added to Claude Code", fg=theme.SUCCESS)
        else:
            click.secho(f"  x MCP failed: {result.stderr.strip()[:80]}", fg=theme.WARNING)
            click.secho(f"    Run: claude mcp add --transport http ada {mcp_url}", fg=theme.TEXT_DIM)
    except FileNotFoundError:
        click.secho("  x Claude Code CLI not found (npm install -g @anthropic-ai/claude-code)", fg=theme.WARNING)
        click.secho(f"    Run: claude mcp add --transport http ada {mcp_url}", fg=theme.TEXT_DIM)

    # 2. Migrate: remove Ada section from CLAUDE.md if previously injected
    target_claude_md = repo / "CLAUDE.md"
    _GLYPHH_MARKER = "# Ada Code Intelligence"
    if target_claude_md.exists():
        existing = target_claude_md.read_text()
        if _GLYPHH_MARKER in existing:
            marker_pos = existing.index(_GLYPHH_MARKER)
            cleaned = existing[:marker_pos].rstrip()
            if cleaned:
                target_claude_md.write_text(cleaned + "\n")
            else:
                target_claude_md.unlink()
            click.secho("  + Migrated Ada instructions out of CLAUDE.md", fg=theme.SUCCESS)

    # 3. Add hooks + permissions to .claude/settings.json
    settings_file = claude_dir / "settings.json"
    settings = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            pass

    permissions = settings.setdefault("permissions", {})
    allow_list = permissions.setdefault("allow", [])
    if "mcp__ada__*" not in allow_list:
        allow_list.append("mcp__ada__*")

    hooks = settings.setdefault("hooks", {})
    ada_dir = repo / ".ada"

    # SessionStart: reset the search-gate flag
    session_hooks = hooks.setdefault("SessionStart", [])
    reset_cmd = f"rm -f {ada_dir}/.search_used"
    session_hooks[:] = [
        h for h in session_hooks
        if ".search_used" not in h.get("hooks", [{}])[0].get("command", "")
    ]
    session_hooks.append({
        "hooks": [{"type": "command", "command": reset_cmd}],
    })

    # PreToolUse: gate Grep/Glob/Bash(grep|find) until ada_search called
    pre_hooks = hooks.setdefault("PreToolUse", [])
    gate_cmd = _hook_cmd("search-gate", ada_dir)
    pre_hooks[:] = [
        h for h in pre_hooks
        if "search-gate" not in h.get("hooks", [{}])[0].get("command", "")
        and ".search_used" not in h.get("hooks", [{}])[0].get("command", "")
        and "enforce-ada-search" not in h.get("hooks", [{}])[0].get("command", "")
    ]
    pre_hooks.append({
        "matcher": "Grep|Glob|Bash",
        "hooks": [{"type": "command", "command": gate_cmd}],
    })

    # PostToolUse: set search-gate flag after ada_search
    post_hooks = hooks.setdefault("PostToolUse", [])
    flag_cmd = f"mkdir -p {ada_dir} && touch {ada_dir}/.search_used"
    post_hooks[:] = [
        h for h in post_hooks
        if ".search_used" not in h.get("hooks", [{}])[0].get("command", "")
    ]
    post_hooks.append({
        "matcher": "mcp__ada__ada_search",
        "hooks": [{"type": "command", "command": flag_cmd}],
    })

    # PostToolUse: incremental compile after git commits
    compile_cmd = _hook_cmd("post-git-compile", repo)
    if is_upgrade:
        post_hooks[:] = [
            h for h in post_hooks
            if "post-git-compile" not in h.get("hooks", [{}])[0].get("command", "")
            and "post-commit-compile" not in h.get("hooks", [{}])[0].get("command", "")
        ]
    else:
        post_hooks[:] = [
            h for h in post_hooks
            if "post-commit-compile" not in h.get("hooks", [{}])[0].get("command", "")
        ]
    if not any("post-git-compile" in h.get("hooks", [{}])[0].get("command", "") for h in post_hooks):
        post_hooks.append({
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": compile_cmd}],
        })

    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    click.secho("  + Hooks and permissions configured", fg=theme.SUCCESS)

    # 4. Write .claude/rules/ada.md
    rules_dir = claude_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    rules_file = rules_dir / "ada.md"
    rules_file.write_text(
        "# Ada Code Intelligence\n"
        "\n"
        "This project has a Ada semantic index.\n"
        "Use ada MCP tools for codebase search.\n"
        "\n"
        "## Rules\n"
        "\n"
        "ALWAYS call ada_search before using Grep, Glob, or the Agent tool.\n"
        "NEVER use the Agent tool to explore the codebase. Use ada_search.\n"
        "NEVER scan directories or read multiple files speculatively.\n"
        "Use ada_context INSTEAD of Read when you have a specific question\n"
        "about a file. It returns only the matching sections, not the full file.\n"
        "Only fall back to Read when you need the complete file.\n"
        "Call ada_related before editing any file to understand blast radius.\n"
        "\n"
        "## Session Memory\n"
        "\n"
        "Use ada_session_write to record important decisions, changes, and\n"
        "context as you work. This encodes notes into HDC vectors so they persist\n"
        "across context window compactions.\n"
        "Use ada_session_recall to retrieve prior notes by semantic similarity\n"
        "before re-reading files or re-exploring code you already investigated.\n"
    )
    click.secho("  + Navigation rules written to .claude/rules/ada.md", fg=theme.SUCCESS)

    # 5. Write .ada/manifest.yaml
    ada_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ada_dir / "manifest.yaml"
    manifest_path.write_text(
        "# Auto-generated by ada code init -- do not edit\n"
        "model_id: code\n"
        "name: Ada Code\n"
    )

    # 6. Add .ada/ to .gitignore
    gitignore = repo / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".ada/" not in content:
            with open(gitignore, "a") as f:
                f.write("\n# Ada local index\n.ada/\n")
            click.secho("  + .ada/ added to .gitignore", fg=theme.SUCCESS)
    else:
        gitignore.write_text("# Ada local index\n.ada/\n")
        click.secho("  + .gitignore created", fg=theme.SUCCESS)


# ---------------------------------------------------------------------------
# Command handlers — each receives (args: str, ctx: dict)
# ---------------------------------------------------------------------------

def _cmd_init(args: str, ctx: dict):
    """Setup or upgrade a repository for Ada Code."""
    repo = str(Path(args.strip() or ".").resolve())

    if not Path(repo).is_dir():
        click.secho(f"  Not a directory: {repo}", fg=theme.ERROR)
        return

    if not ctx:
        click.secho("  Not logged in. Run: ada auth login", fg=theme.ERROR)
        return

    # Detect upgrade
    is_upgrade = False
    if _STATE_FILE.exists():
        try:
            prev = json.loads(_STATE_FILE.read_text())
            if prev.get("repo") == repo:
                is_upgrade = True
        except (json.JSONDecodeError, KeyError):
            pass

    mode = "upgrade" if is_upgrade else "init"
    click.echo()
    click.secho(f"  Ada Code  ·  {mode}", fg=theme.TEXT, bold=True)
    click.secho(f"  {repo}", fg=theme.TEXT_DIM)
    click.echo()

    # Step 1: Deploy model
    click.secho("  [1/4] Deploying model...", fg=theme.MUTED)
    _deploy_model(ctx, project_dir=Path(repo))

    # Step 2: Clear existing index
    click.secho("  [2/4] Clearing index...", fg=theme.MUTED)
    try:
        import httpx
        with httpx.Client(timeout=30) as client:
            r = client.delete(
                f"{ctx['runtime_url']}/{ctx['org_id']}/code/data",
                headers=ctx["headers"],
            )
            if r.status_code == 200:
                deleted = r.json().get("glyphs_deleted", 0)
                if deleted:
                    click.secho(f"         {deleted} stale glyphs removed", fg=theme.TEXT_DIM)
    except Exception:
        pass

    # Step 3: Compile + encode
    click.secho("  [3/4] Compiling codebase...", fg=theme.MUTED)
    file_count, job_ids = _compile_repo(repo, ctx)
    click.secho(f"         {file_count} files indexed", fg=theme.TEXT_DIM)

    # Step 4: Configure Claude Code
    click.secho("  [4/4] Configuring Claude Code...", fg=theme.MUTED)
    mcp_url = f"{ctx['runtime_url']}/{ctx['org_id']}/code/mcp"
    _configure_claude_code(repo, mcp_url, ctx, is_upgrade=is_upgrade)

    # Save state
    _GLYPHH_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({
        "repo": repo,
        "runtime_url": ctx["runtime_url"],
        "org_id": ctx["org_id"],
        "mcp_url": mcp_url,
        "file_count": file_count,
    }))

    click.echo()
    dot = click.style("*", fg=theme.SUCCESS)
    label = "upgraded" if is_upgrade else "ready"
    click.echo(f"  {dot} {click.style(label, fg=theme.SUCCESS)}")
    click.echo()
    click.secho(f"  Repo:      {repo}", fg=theme.TEXT_DIM)
    click.secho(f"  Files:     {file_count} indexed", fg=theme.TEXT_DIM)
    click.secho(f"  MCP:       {mcp_url}", fg=theme.ACCENT)
    click.echo()
    click.secho("  Restart Claude Code to activate.", fg=theme.MUTED)
    click.secho("  VS Code: Cmd+Shift+P > 'Claude Code: Restart'", fg=theme.TEXT_DIM)
    click.echo()


def _cmd_compile(args: str, ctx: dict):
    """Recompile the codebase index."""
    if not ctx:
        click.secho("  Not logged in. Run: ada auth login", fg=theme.ERROR)
        return

    repo = str(Path(args.strip() or ".").resolve())
    click.secho(f"  Compiling: {repo}", fg=theme.TEXT)
    count, job_ids = _compile_repo(repo, ctx)
    click.secho(f"  Done: {count} files indexed", fg=theme.SUCCESS)


def _cmd_status(args: str, ctx: dict):
    """Show current Ada Code status."""
    if not _STATE_FILE.exists():
        click.secho("  Ada Code not initialized.", fg=theme.TEXT_DIM)
        click.secho("  Run: init /path/to/repo", fg=theme.MUTED)
        return

    state = json.loads(_STATE_FILE.read_text())

    click.echo()
    org_id = state.get("org_id", (ctx or {}).get("org_id", "?"))
    click.secho(f"  Org:     {org_id}", fg=theme.TEXT_DIM)
    click.secho(f"  Repo:    {state.get('repo', '?')}", fg=theme.TEXT_DIM)
    click.secho(f"  Files:   {state.get('file_count', '?')} indexed", fg=theme.TEXT_DIM)
    click.secho(f"  MCP:     {state.get('mcp_url', '?')}", fg=theme.ACCENT)
    click.echo()


# ---------------------------------------------------------------------------
# Registration — called by runtime model REPL
# ---------------------------------------------------------------------------

def register(ctx):
    """Return custom commands for the model REPL."""
    return {
        "init": {
            "handler": _cmd_init,
            "help": "Setup or upgrade a repository",
        },
        "compile": {
            "handler": _cmd_compile,
            "help": "Recompile the codebase index",
        },
        "code-status": {
            "handler": _cmd_status,
            "help": "Show Ada Code status",
        },
    }
