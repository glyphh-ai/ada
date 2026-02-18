from __future__ import annotations

"""
Interactive REPL shell for Glyphh CLI.
Provides a persistent session with command history and auto-completion.
Supports both CLI commands and natural language queries via the assistant.
"""

import click
import shlex
from typing import Optional, List
from pathlib import Path

from .auth import get_current_user
from .banner import print_banner, QUICK_ACTIONS, QUICK_ACTIONS_WEB, handle_quick_action, match_quick_action_phrase
from .security import sanitize_query, validate_command_arg, SecurityError
from . import theme
from .ui import print_error, print_command_error, print_info

# Try to import readline for history/completion (not available on all platforms)
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

# History file
HISTORY_FILE = Path.home() / ".glyphh" / "history"

# Available commands for completion
COMMANDS = [
    "help", "exit", "quit", "clear", "home", "welcome", "why",
    "auth login", "auth logout", "auth signup", "auth whoami", "auth status",
    "build init", "build add", "build validate", "build info",
    "settings theme", "settings show",
    "test similarity", "test encode",
    "package create", "package info",
    "runtime deploy", "runtime status", "runtime logs",
    "query", "procedure list",
    "models", "hub", "docs", "demo",
]

# Known CLI command prefixes (first word of commands)
CLI_COMMAND_PREFIXES = {
    "help", "exit", "quit", "q", "clear", "home", "welcome", "why",
    "auth", "build", "settings", "test", "package", "runtime",
    "query", "procedure", "models", "hub", "docs", "demo",
}

# Commands that require local SDK (filesystem access, model building)
LOCAL_ONLY_COMMANDS = {"build", "test", "package", "settings"}


def setup_readline():
    """Setup readline for history and completion."""
    if not HAS_READLINE:
        return
    
    # Load history
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_FILE.exists():
        try:
            readline.read_history_file(str(HISTORY_FILE))
        except Exception:
            pass
    
    # Set history length
    readline.set_history_length(1000)
    
    # Setup completion
    def completer(text: str, state: int) -> Optional[str]:
        buf = readline.get_line_buffer().lstrip()
        options = []
        if " " in buf:
            # Sub-command completion: match full command strings
            for cmd in COMMANDS:
                if cmd.startswith(buf) and cmd != buf:
                    # Return only the part after what's already typed
                    suffix = cmd[len(buf) - len(text):]
                    options.append(suffix)
        else:
            # Top-level: match first word of commands + deduplicate
            seen = set()
            for cmd in COMMANDS:
                first = cmd.split()[0]
                if first.startswith(text) and first not in seen:
                    seen.add(first)
                    options.append(first)
        if state < len(options):
            return options[state]
        return None
    
    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    
    # macOS uses libedit, which needs different keybinding syntax
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def save_history():
    """Save command history."""
    if HAS_READLINE:
        try:
            readline.write_history_file(str(HISTORY_FILE))
        except:
            pass


def _get_cols() -> int:
    """Get terminal column width."""
    return 120


def _is_web() -> bool:
    """Check if running in a web terminal session."""
    import os
    return os.environ.get("GLYPHH_WEB_TERMINAL") == "1"


def print_help():
    """Print help information."""
    web = _is_web()
    
    if web:
        sections = [
            ("Authentication", [
                ("auth login", "Login to your account"),
                ("auth logout", "Logout"),
                ("auth signup", "Create a new account"),
                ("auth whoami", "Show current user"),
            ]),
            ("Discovery", [
                ("hub", "Browse the model hub"),
                ("models", "List your deployed models"),
                ("docs", "Documentation"),
                ("demo", "Interactive demos"),
            ]),
            ("Runtime", [
                ("runtime deploy", "Deploy a model"),
                ("runtime status", "Check runtime status"),
                ("query", "Query a deployed model"),
                ("procedure list", "List stored procedures"),
            ]),
        ]
    else:
        sections = [
            ("Authentication", [
                ("auth login", "Login to your account"),
                ("auth logout", "Logout"),
                ("auth signup", "Create a new account"),
                ("auth whoami", "Show current user"),
            ]),
            ("Build", [
                ("build init", "Initialize a new model"),
                ("build add", "Add data to a model"),
                ("test similarity", "Test similarity"),
                ("package create", "Package a model"),
            ]),
            ("Deploy & Query", [
                ("runtime deploy", "Deploy a model"),
                ("runtime status", "Check runtime status"),
                ("query", "Query a model"),
                ("procedure list", "List stored procedures"),
            ]),
            ("Discovery", [
                ("hub", "Browse the model hub"),
                ("models", "List your deployed models"),
                ("docs", "Documentation"),
            ]),
        ]
    
    click.echo()
    
    box_width = 62
    inner_width = box_width - 4
    
    click.secho("  ╭─ ", fg=theme.TEXT_HIGHLIGHT, nl=False)
    click.secho("COMMANDS", fg=theme.TEXT_HIGHLIGHT, bold=True, nl=False)
    click.secho(" " + "─" * (box_width - 14) + "╮", fg=theme.TEXT_HIGHLIGHT)
    
    click.secho("  │" + " " * inner_width + " │", fg=theme.TEXT_HIGHLIGHT)
    
    for section_name, commands in sections:
        section_text = f" {section_name}"
        padding = inner_width - len(section_text)
        click.secho("  │", fg=theme.TEXT_HIGHLIGHT, nl=False)
        click.secho(section_text, fg=theme.WARNING, nl=False)
        click.secho(" " * max(0, padding) + " │", fg=theme.TEXT_HIGHLIGHT)
        
        for cmd, desc in commands:
            cmd_col = 18
            desc_space = inner_width - cmd_col - 3
            click.secho("  │", fg=theme.TEXT_HIGHLIGHT, nl=False)
            click.secho(f"   {cmd:<{cmd_col}}", fg=theme.TEXT, nl=False)
            click.secho(desc[:desc_space], fg=theme.MUTED, nl=False)
            remaining = desc_space - len(desc[:desc_space])
            click.secho(" " * remaining + " │", fg=theme.TEXT_HIGHLIGHT)
        
        click.secho("  │" + " " * inner_width + " │", fg=theme.TEXT_HIGHLIGHT)
    
    click.secho("  ╰" + "─" * (box_width - 2) + "╯", fg=theme.TEXT_HIGHLIGHT)
    click.echo()


def get_prompt() -> str:
    """Get the shell prompt."""
    user = get_current_user()
    
    if user:
        email = user.get("email", "user")
        # Show just the username part
        username = email.split("@")[0] if "@" in email else email
        return click.style(f"glyphh:{username}", fg=theme.PRIMARY) + click.style("> ", fg=theme.TEXT)
    else:
        return click.style("glyphh", fg=theme.PRIMARY) + click.style("> ", fg=theme.TEXT)


def is_cli_command(text: str) -> bool:
    """Check if text looks like a CLI command vs natural language."""
    if not text:
        return False
    first_word = text.split()[0].lower()
    return first_word in CLI_COMMAND_PREFIXES


def _confidence_color(score: float) -> str:
    """Return color for confidence score: green >= 0.7, yellow >= 0.5, red below."""
    if score >= 0.7:
        return "green"
    elif score >= 0.5:
        return "yellow"
    return "red"


def _print_routing_trace(response):
    """Show a brief inline trace of how the Glyphh model routed the query.
    
    Surfaces the extracted concept and match info so the user can see
    the model thinking — pure Glyphh, no LLM.
    """
    why = response.metadata.get("why", {})
    extracted = why.get("extracted", {})
    matched_q = why.get("matched_question", "")
    score = why.get("score", response.confidence)
    
    if not extracted:
        return
    
    # Show extracted concept: verb/object/domain
    verb = extracted.get("verb", "")
    obj = extracted.get("object", "")
    domain = extracted.get("domain", "")
    
    parts = []
    if verb:
        parts.append(f"verb={verb}")
    if obj:
        parts.append(f"object={obj}")
    if domain:
        parts.append(f"domain={domain}")
    
    concept_str = " ".join(parts)
    color = _confidence_color(score)
    
    click.echo()
    click.secho("  ⟶ ", fg=theme.MUTED, nl=False)
    
    if matched_q:
        click.secho(f"matched ", fg=theme.MUTED, nl=False)
        click.secho(f'"{matched_q}"', fg=theme.TEXT_HIGHLIGHT, nl=False)
        click.secho(f" (", fg=theme.MUTED, nl=False)
        click.secho(f"{score:.0%}", fg=color, nl=False)
        click.secho(f")", fg=theme.MUTED)
    else:
        click.secho(f"no match above threshold (", fg=theme.MUTED, nl=False)
        click.secho(f"{score:.0%}", fg=color, nl=False)
        click.secho(f")", fg=theme.MUTED)
    
    click.secho(f"    [{concept_str}]", fg="bright_black")
    
    # Show normalization if it happened
    norm_method = why.get("normalization_method")
    norm_query = why.get("normalized_query")
    if norm_method and norm_query:
        label = "spell-fix" if norm_method == "offline_spell_fix" else norm_method
        click.secho(f"    normalized via {label}: ", fg="bright_black", nl=False)
        click.secho(f'"{norm_query}"', fg=theme.MUTED)


def _extract_runnable_command(response) -> list[str] | None:
    """Extract a runnable CLI command from an assistant response.
    
    If the response maps to a real CLI command (e.g. 'glyphh auth login'),
    return it as args suitable for run_command(). Returns None if the
    response isn't a command or confidence is too low.
    """
    if not response.command or response.confidence < 0.5:
        return None
    
    cmd = response.command.strip()
    # Strip 'glyphh ' prefix if present
    if cmd.startswith("glyphh "):
        cmd = cmd[len("glyphh "):]
    
    # Only auto-execute if the first word is a known CLI command
    try:
        args = shlex.split(cmd)
    except ValueError:
        return None
    
    if not args:
        return None
    
    first_word = args[0].lower()
    if first_word not in CLI_COMMAND_PREFIXES:
        return None
    
    # Don't auto-execute commands with placeholders like <model_name>
    for arg in args:
        if "<" in arg and ">" in arg:
            return None
    
    return args


def print_assistant_response(response):
    """Print an assistant response using a card component with streaming text."""
    from .components import Card, print_card
    from .streaming import stream_text
    
    # Determine status based on state and confidence
    status = None
    if response.state == "DONE":
        if response.confidence > 0.5:
            status = "success"
        elif response.confidence > 0.3:
            status = "info"
    elif response.state == "ASK":
        status = "info"
    elif response.state == "theme.ERROR":
        status = None
    
    # Build metadata — no trace/method on regular responses
    metadata = {}
    if response.command:
        metadata["try"] = response.command
    
    # Build card
    card = Card(
        title="glyphh ai",
        body=response.content if response.content else None,
        status=status,
        metadata=metadata if metadata else None,
    )
    
    print_card(card, stream=True)
    
    # Show code separately if available (outside card for better formatting)
    if response.code:
        click.secho("  ─── code ───", fg=theme.MUTED)
        for line in response.code.split('\n'):
            stream_text(f"  {line}", fg=theme.SUCCESS)
        click.echo()
    
    # Confidence line — always last, right above the prompt
    color = _confidence_color(response.confidence)
    click.secho("  confidence: ", fg=theme.MUTED, nl=False)
    click.secho(f"{response.confidence:.0%}", fg=color)
    click.echo()


def print_conversational_response(response):
    """Print an LLM-formatted response conversationally — no card, no trace.
    
    Word-wraps at 80 chars so text doesn't stretch edge-to-edge.
    """
    import textwrap
    from .streaming import stream_text

    max_width = 76  # 80 minus the 4-char indent

    click.echo()
    for line in response.content.split("\n"):
        if not line.strip():
            click.echo()
            continue
        # Wrap long lines
        wrapped = textwrap.wrap(line, width=max_width) or [""]
        for wl in wrapped:
            stream_text(f"  {wl}", fg=theme.TEXT)
    click.echo()


def print_low_confidence_fallback(response):
    """When confidence is below threshold, show quick actions instead."""
    web = _is_web()
    
    click.echo()
    click.secho("  I'm not confident enough to answer that.", fg=theme.MUTED)
    color = _confidence_color(response.confidence)
    click.secho("  confidence: ", fg=theme.MUTED, nl=False)
    click.secho(f"{response.confidence:.0%}", fg=color)
    click.echo()
    click.secho("  Type ", fg=theme.MUTED, nl=False)
    click.secho("help", fg=theme.TEXT_HIGHLIGHT, nl=False)
    click.secho(" for commands.", fg=theme.MUTED)
    click.echo()


def print_why(response):
    """Print explainability details for the last response."""
    from .components import Card, print_card

    why = response.metadata.get("why") if response.metadata else None

    if not why:
        click.echo()
        click.secho("  No explanation available for the last response.", fg=theme.MUTED)
        if response.match_method:
            click.secho(f"  method: {response.match_method}", fg=theme.MUTED)
        color = _confidence_color(response.confidence)
        click.secho("  confidence: ", fg=theme.MUTED, nl=False)
        click.secho(f"{response.confidence:.0%}", fg=color)
        click.echo()
        return

    extracted = why.get("extracted", {})
    runners_up = why.get("runners_up", [])

    lines = []
    lines.append(f'query:    "{why.get("query", "")}"')

    # Show extracted attributes (skip if phrase match)
    if extracted.get("match_type") != "exact_phrase":
        lines.append(f'verb:     {extracted.get("verb", "?")}')
        lines.append(f'object:   {extracted.get("object", "?")}')
        lines.append(f'domain:   {extracted.get("domain", "?")}')
    else:
        lines.append("match:    exact phrase")

    lines.append("")

    if why.get("original_query") and why.get("normalized_query"):
        lines.append(f'original: "{why["original_query"]}"')
        lines.append(f'fixed:    "{why["normalized_query"]}"')
        lines.append(f'method:   {why.get("normalization_method", "unknown")}')
        lines.append("")

    matched_q = why.get("matched_question")
    if matched_q:
        lines.append(f'matched:  "{matched_q}" ({why.get("matched_content_type", "")})')
    else:
        reason = why.get("reason", "No match found")
        lines.append(f'result:   {reason}')

    lines.append(f'score:    {why.get("score", 0):.2f}')

    card_metadata = {
        "method": response.match_method or "unknown",
    }
    if why.get("normalization_method"):
        card_metadata["normalization"] = why["normalization_method"]

    card = Card(
        title="why this answer",
        body="\n".join(lines),
        status="info",
        metadata=card_metadata,
    )
    print_card(card)

    # Similarity table — top 5 stack-ranked
    if runners_up:
        click.echo()
        click.secho("  ─── similarity table (top 5) ───", fg=theme.MUTED)
        click.echo()

        # Header
        click.secho("  ", nl=False)
        click.secho(f"{'#':<4}{'score':<8}{'type':<14}", fg=theme.MUTED, nl=False)
        click.secho("question", fg=theme.MUTED)
        click.secho("  ", nl=False)
        click.secho("─" * 56, fg=theme.MUTED)

        # Include the matched result as #1 if it exists
        all_entries = []
        if matched_q:
            all_entries.append({
                "score": why.get("score", 0),
                "question": matched_q,
                "content_type": why.get("matched_content_type", ""),
            })
        all_entries.extend(runners_up)

        for rank, entry in enumerate(all_entries[:5], 1):
            score = entry["score"]
            color = _confidence_color(score)
            q_text = entry["question"]
            if len(q_text) > 30:
                q_text = q_text[:27] + "..."
            ct = entry.get("content_type", "")

            click.secho("  ", nl=False)
            if rank == 1 and matched_q:
                click.secho(f"{'>' + str(rank):<4}", fg=color, nl=False)
            else:
                click.secho(f"{rank:<4}", fg=theme.MUTED, nl=False)
            click.secho(f"{score:.2f}    ", fg=color, nl=False)
            click.secho(f"{ct:<14}", fg=theme.MUTED, nl=False)
            click.secho(q_text, fg=theme.TEXT)

        click.echo()


def _maybe_show_welcome():
    """Show welcome screen on first-ever CLI visit.

    Uses a flag in ~/.glyphh/config.json to track whether the user
    has already seen the welcome. Only shows on web sessions.
    """
    if not _is_web():
        return

    from .theme import _load_config, _save_config
    config = _load_config()
    if config.get("has_seen_welcome"):
        return

    from .screens import show_welcome_new
    user = get_current_user()
    first_name = user.get("first_name") if user else None
    show_welcome_new(first_name=first_name)

    config["has_seen_welcome"] = True
    _save_config(config)

def run_command(cli, args: List[str]):
    """Run a CLI command with the given arguments.
    
    Uses direct invocation to support interactive prompts.
    """
    try:
        # Create a new context and invoke directly (supports prompts)
        with cli.make_context(cli.name, args) as ctx:
            cli.invoke(ctx)
    except click.ClickException as e:
        e.show()
    except SystemExit:
        pass  # Normal exit from commands
    except Exception as e:
        error_msg = str(e) if str(e) else e.__class__.__name__
        print_command_error(" ".join(args), error_msg)


@click.command()
@click.pass_context
def shell(ctx):
    """
    Start an interactive Glyphh shell.
    
    The shell provides a persistent session with command history,
    tab completion, and a streamlined interface for working with Glyphh.
    
    Supports both CLI commands and natural language queries.
    Type commands like 'auth login' or ask questions like 'how do I deploy?'
    """
    # Get the parent CLI group
    cli = ctx.parent.command if ctx.parent else None
    
    # Initialize assistant (lazy load)
    assistant = None
    last_response = None  # Track last response for "why" command
    
    def get_assistant():
        nonlocal assistant
        if assistant is None:
            try:
                from glyphh.assistant.core import Assistant, AssistantConfig
                import os
                runtime_url = os.environ.get("GLYPHH_RUNTIME_URL")
                platform_url = os.environ.get("GLYPHH_API_URL") or "http://localhost:8001"
                config = AssistantConfig(
                    runtime_url=runtime_url,
                    platform_url=platform_url,
                    threshold=0.5,
                )
                assistant = Assistant(config)
                assistant.load()
            except Exception as e:
                click.secho(f"  Note: Assistant unavailable ({e})", fg="bright_black")
                return None
        return assistant
    
    setup_readline()
    print_banner()
    
    # Check if this is a new user and show welcome experience
    _maybe_show_welcome()
    
    try:
        while True:
            try:
                # Get input
                prompt = get_prompt()
                line = input(prompt).strip()
                
                if not line:
                    continue
                
                # Sanitize input for security
                line = sanitize_query(line)
                if not line:
                    continue
                
                # Check if it's a CLI command or natural language
                if is_cli_command(line):
                    # Parse the command
                    try:
                        args = shlex.split(line)
                    except ValueError as e:
                        print_error("Parse Error", f"Could not parse command: {e}")
                        continue
                    
                    # Validate command arguments
                    try:
                        validated_args = [args[0]]  # Command name is safe
                        for arg in args[1:]:
                            # Allow paths for file-related commands
                            allow_paths = args[0] in ('build', 'package', 'runtime')
                            validated_args.append(validate_command_arg(arg, allow_paths=allow_paths))
                        args = validated_args
                    except SecurityError as e:
                        print_error("Security Error", str(e))
                        continue
                    
                    cmd = args[0].lower()
                    
                    # Built-in commands
                    if cmd in ("exit", "quit", "q"):
                        break
                    elif cmd in ("clear", "home"):
                        click.clear()
                        print_banner()
                        continue
                    elif cmd == "welcome":
                        from .screens import show_welcome_new
                        user = get_current_user()
                        first_name = user.get("first_name") if user else None
                        show_welcome_new(first_name=first_name)
                        continue
                    elif cmd == "help":
                        print_help()
                        continue
                    elif cmd == "why":
                        if last_response:
                            print_why(last_response)
                        else:
                            click.echo()
                            click.secho("  Ask a question first, then type 'why' to see how I got the answer.", fg=theme.MUTED)
                            click.echo()
                        continue
                    
                    # Block local-only commands in web terminal
                    if cmd in LOCAL_ONLY_COMMANDS and _is_web():
                        click.echo()
                        click.secho("  that command requires the local SDK.", fg=theme.MUTED)
                        click.secho("  install it with: ", fg=theme.MUTED, nl=False)
                        click.secho("pip install glyphh", fg=theme.TEXT_HIGHLIGHT)
                        click.echo()
                        click.secho("  from the web you can: ", fg=theme.MUTED, nl=False)
                        click.secho("hub", fg=theme.TEXT_HIGHLIGHT, nl=False)
                        click.secho(", ", fg=theme.MUTED, nl=False)
                        click.secho("models", fg=theme.TEXT_HIGHLIGHT, nl=False)
                        click.secho(", ", fg=theme.MUTED, nl=False)
                        click.secho("runtime deploy", fg=theme.TEXT_HIGHLIGHT, nl=False)
                        click.secho(", ", fg=theme.MUTED, nl=False)
                        click.secho("query", fg=theme.TEXT_HIGHLIGHT)
                        click.echo()
                        continue
                    
                    # Run the command through the CLI
                    if cli:
                        try:
                            run_command(cli, args)
                        except SystemExit:
                            pass  # Click raises SystemExit on --help, etc.
                        except Exception as e:
                            print_command_error(line, str(e))
                    else:
                        print_error("Unknown Command", f"'{cmd}' is not a recognized command.\n\nType 'help' to see available commands.")
                else:
                    # Check for quick action shortcuts (1, 2, 3, 4, etc.)
                    quick_actions = QUICK_ACTIONS_WEB if _is_web() else QUICK_ACTIONS
                    if line in quick_actions:
                        handle_quick_action(line)
                        # Update assistant history so follow-up detection works
                        asst = get_assistant()
                        if asst:
                            asst._update_history(line, f"[quick action {line}]")
                        continue
                    
                    # Check for natural language phrases that map to quick actions
                    qa_key = match_quick_action_phrase(line)
                    if qa_key:
                        # Exact phrase match = 100% confidence (deterministic)
                        from glyphh.assistant.core import AssistantResponse
                        last_response = AssistantResponse(
                            state="DONE",
                            content=f"Matched quick action [{qa_key}]",
                            confidence=1.0,
                            match_method="phrase_match",
                            metadata={
                                "why": {
                                    "query": line,
                                    "extracted": {"match_type": "exact_phrase"},
                                    "matched_question": line,
                                    "matched_content_type": "quick_action",
                                    "score": 1.0,
                                    "runners_up": [],
                                },
                            },
                        )
                        handle_quick_action(qa_key)
                        # Update assistant history so follow-up detection works
                        asst = get_assistant()
                        if asst:
                            asst._update_history(line, last_response.content)
                        color = _confidence_color(1.0)
                        click.secho("  confidence: ", fg=theme.MUTED, nl=False)
                        click.secho("100%", fg=color)
                        click.echo()
                        continue
                    
                    # Natural language query - use assistant
                    asst = get_assistant()
                    if asst:
                        from .spinner import GridSpinner
                        spinner = GridSpinner("thinking")
                        spinner.start()
                        try:
                            response = asst.ask(line)
                        finally:
                            spinner.stop()
                        last_response = response
                        
                        # LLM-formatted responses (from platform) get
                        # conversational rendering — no card, no trace.
                        # Offline/corrected responses keep the full card + trace.
                        is_llm_response = response.match_method in ("llm", "action", "llm-only", "llm_assisted")
                        
                        if not is_llm_response:
                            # Show routing trace for offline responses
                            _print_routing_trace(response)
                        
                        # If assistant returns a quick action command, execute it directly
                        if response.command and response.command in quick_actions:
                            click.secho(f"  -> {response.content}", fg=theme.MUTED)
                            handle_quick_action(response.command)
                            color = _confidence_color(response.confidence)
                            click.secho("  confidence: ", fg=theme.MUTED, nl=False)
                            click.secho(f"{response.confidence:.0%}", fg=color)
                            click.echo()
                        elif is_llm_response:
                            # LLM already formatted this — always show it
                            print_conversational_response(response)
                        elif response.state == "theme.ERROR" or response.confidence < asst.config.threshold:
                            # Below threshold — show quick actions instead of a weak answer
                            print_low_confidence_fallback(response)
                        else:
                            # Offline response — check for runnable command or show card
                            runnable = _extract_runnable_command(response)
                            if runnable and cli:
                                # Auto-execute the matched command
                                click.secho(f"  running: ", fg=theme.MUTED, nl=False)
                                click.secho(" ".join(runnable), fg=theme.TEXT_HIGHLIGHT)
                                click.echo()
                                try:
                                    run_command(cli, runnable)
                                except SystemExit:
                                    pass
                                except Exception as e:
                                    print_command_error(" ".join(runnable), str(e))
                            else:
                                # Show full response card
                                print_assistant_response(response)
                    else:
                        print_info("Tip", "Try a command like 'help', 'models', or 'docs'")
                
            except KeyboardInterrupt:
                click.echo()  # New line after ^C
                continue
            except EOFError:
                break
                
    finally:
        save_history()
        click.echo()
        click.secho("Goodbye!", fg="cyan")
