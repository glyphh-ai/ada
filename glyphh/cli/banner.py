from __future__ import annotations

"""
Banner and quick actions for the Glyphh CLI shell.
"""

import os
import click
from .auth import get_current_user, is_web_session
from . import theme
from .screens import show_what_is_glyphh, show_build_model, show_find_model, show_pricing
from .screens import show_reel_churn, show_reel_product, show_get_started
from .streaming import stream_text, stream_echo


def _is_web() -> bool:
    """Check if running in a web terminal session."""
    return is_web_session()


# Quick action shortcuts - these trigger specific handlers
QUICK_ACTIONS = {
    "1": "_show_what_is_glyphh",
    "2": "_show_build_model",
    "3": "_show_find_model",
    "4": "_show_pricing",
    "5": "_show_reel_churn",
    "6": "_show_reel_product",
    "7": "_show_what_is_glyphh",
}

# Web-mode overrides: no build, add "get started" to guide local install
QUICK_ACTIONS_WEB = {
    "1": "_show_what_is_glyphh",
    "2": "_show_find_model",
    "3": "_show_pricing",
    "4": "_show_get_started",
    "5": "_show_reel_churn",
    "6": "_show_reel_product",
    "7": "_show_what_is_glyphh",
}

# Natural language phrases that map to quick actions
QUICK_ACTION_PHRASES = {
    "1": [
        "what is glyphh", "what is glyphh ai", "what is glyphh?", "what is glyphh ai?",
        "tell me about glyphh", "about glyphh", "what does glyphh do",
    ],
    "2": [
        "build a model", "create a model", "how to build a model", "make a model",
        "new model", "start a model",
    ],
    "3": [
        "find a model", "search models", "browse models", "model marketplace",
        "hub", "marketplace",
    ],
    "4": [
        "pricing", "how much", "cost", "plans", "free tier",
    ],
}


def print_banner():
    """Print the welcome banner."""
    user = get_current_user()
    web = _is_web()

    click.echo()

    # ASCII text
    stream_text("        _             _     _             _", fg=theme.PRIMARY)
    stream_text("   __ _| |_   _ _ __ | |__ | |__     __ _(_)", fg=theme.ACCENT)
    stream_text("  / _` | | | | | '_ \\| '_ \\| '_ \\   / _` | |", fg="cyan")
    stream_text(" | (_| | | |_| | |_) | | | | | | | | (_| | |", fg="cyan")
    stream_text("  \\__, |_|\\__, | .__/|_| |_|_| |_|  \\__,_|_|", fg="bright_cyan")
    stream_text("  |___/   |___/|_|", fg="bright_cyan")
    click.echo()
    stream_text("  when your llm can't afford to be wrong", fg="bright_cyan")

    click.echo()

    if user:
        click.secho("  + ", fg=theme.SUCCESS, nl=False)
        click.echo("logged in as ", nl=False)
        click.secho(user.get('email'), fg=theme.TEXT_HIGHLIGHT)
    else:
        click.secho("  o ", fg=theme.WARNING, nl=False)
        click.echo("not logged in", nl=False)
        click.secho(" -> ", fg=theme.MUTED, nl=False)
        click.secho("auth login", fg=theme.TEXT_HIGHLIGHT, nl=False)
        click.echo(" or ", nl=False)
        click.secho("auth signup", fg=theme.TEXT_HIGHLIGHT)

    click.echo()

    click.echo()


def handle_quick_action(action_key: str) -> bool:
    """
    Handle a quick action shortcut.
    
    Returns True if the action was handled, False otherwise.
    """
    actions = QUICK_ACTIONS_WEB if _is_web() else QUICK_ACTIONS
    action = actions.get(action_key)
    if not action:
        return False
    
    if action == "_show_what_is_glyphh":
        show_what_is_glyphh()
        return True
    elif action == "_show_build_model":
        show_build_model()
        return True
    elif action == "_show_find_model":
        show_find_model()
        return True
    elif action == "_show_pricing":
        show_pricing()
        return True
    elif action == "_show_reel_churn":
        show_reel_churn()
        return True
    elif action == "_show_reel_product":
        show_reel_product()
        return True
    elif action == "_show_get_started":
        show_get_started()
        return True
    
    return False


def match_quick_action_phrase(text: str) -> str | None:
    """Match natural language input to a quick action key, or None."""
    import re
    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    for key, phrases in QUICK_ACTION_PHRASES.items():
        for phrase in phrases:
            if normalized == re.sub(r"[^\w\s]", "", phrase.lower()).strip():
                return key
    return None
