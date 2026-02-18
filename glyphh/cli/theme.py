"""
Glyphh brand color theme for CLI.
Defines consistent colors used across CLI and Studio terminal.

Supports light and dark modes via config.

Brand Colors:
- Primary Purple: #9333ea (violet-600)
- Light Purple: #a855f7 (violet-500)
- Dark Purple: #7c3aed (violet-600)
"""

import os
import json
from pathlib import Path
from typing import Optional

# Config directory
GLYPHH_DIR = Path.home() / ".glyphh"
CONFIG_FILE = GLYPHH_DIR / "config.json"


def _load_config() -> dict:
    """Load config from disk."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except:
            pass
    return {}


def _save_config(config: dict):
    """Save config to disk."""
    GLYPHH_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_theme_name() -> str:
    """Get current theme name from config or environment."""
    # Environment override
    if theme := os.environ.get("GLYPHH_THEME"):
        return theme
    
    config = _load_config()
    return config.get("theme", "dark")


def set_theme(theme_name: str) -> bool:
    """Set the theme in config. Returns True if valid."""
    if theme_name not in THEMES:
        return False
    
    config = _load_config()
    config["theme"] = theme_name
    _save_config(config)
    return True


def get_available_themes() -> list:
    """Get list of available theme names."""
    return list(THEMES.keys())


# Theme definitions
THEMES = {
    "dark": {
        # Primary brand colors (ANSI names for Click)
        "PRIMARY": "magenta",
        "ACCENT": "bright_magenta",
        "MUTED": "bright_black",
        
        # Semantic colors
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "cyan",
        
        # Text colors
        "TEXT": "white",
        "TEXT_DIM": "bright_black",
        "TEXT_HIGHLIGHT": "cyan",
        
        # xterm.js theme
        "XTERM": {
            "background": "#0a0a0a",
            "foreground": "#e0e0e0",
            "cursor": "#a855f7",
            "cursorAccent": "#0a0a0a",
            "selectionBackground": "#9333ea40",
            "black": "#0a0a0a",
            "red": "#ef4444",
            "green": "#22c55e",
            "yellow": "#eab308",
            "blue": "#3b82f6",
            "magenta": "#9333ea",
            "cyan": "#06b6d4",
            "white": "#e0e0e0",
            "brightBlack": "#9ca3af",
            "brightRed": "#f87171",
            "brightGreen": "#4ade80",
            "brightYellow": "#facc15",
            "brightBlue": "#60a5fa",
            "brightMagenta": "#a855f7",
            "brightCyan": "#22d3ee",
            "brightWhite": "#ffffff",
        }
    },
    "light": {
        # Primary brand colors - adjusted for light backgrounds
        "PRIMARY": "magenta",
        "ACCENT": "magenta",  # No bright on light bg
        "MUTED": "bright_black",
        
        # Semantic colors
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "blue",  # Blue more visible on light
        
        # Text colors
        "TEXT": "black",
        "TEXT_DIM": "bright_black",
        "TEXT_HIGHLIGHT": "blue",  # Blue pops on light
        
        # xterm.js theme
        "XTERM": {
            "background": "#fafafa",
            "foreground": "#1a1a1a",
            "cursor": "#9333ea",
            "cursorAccent": "#fafafa",
            "selectionBackground": "#9333ea30",
            "black": "#1a1a1a",
            "red": "#dc2626",
            "green": "#16a34a",
            "yellow": "#ca8a04",
            "blue": "#2563eb",
            "magenta": "#9333ea",
            "cyan": "#0891b2",
            "white": "#fafafa",
            "brightBlack": "#6b7280",
            "brightRed": "#ef4444",
            "brightGreen": "#22c55e",
            "brightYellow": "#eab308",
            "brightBlue": "#3b82f6",
            "brightMagenta": "#a855f7",
            "brightCyan": "#06b6d4",
            "brightWhite": "#ffffff",
        }
    },
}


def _get_theme() -> dict:
    """Get the current theme dict."""
    name = get_theme_name()
    return THEMES.get(name, THEMES["dark"])


# Dynamic color getters
def _color(key: str) -> str:
    return _get_theme()[key]


# Export colors as module-level for backward compatibility
# These are evaluated at import time, so we use a class to make them dynamic
class _ThemeColors:
    @property
    def PRIMARY(self) -> str:
        return _color("PRIMARY")
    
    @property
    def ACCENT(self) -> str:
        return _color("ACCENT")
    
    @property
    def MUTED(self) -> str:
        return _color("MUTED")
    
    @property
    def SUCCESS(self) -> str:
        return _color("SUCCESS")
    
    @property
    def WARNING(self) -> str:
        return _color("WARNING")
    
    @property
    def ERROR(self) -> str:
        return _color("ERROR")
    
    @property
    def INFO(self) -> str:
        return _color("INFO")
    
    @property
    def TEXT(self) -> str:
        return _color("TEXT")
    
    @property
    def TEXT_DIM(self) -> str:
        return _color("TEXT_DIM")
    
    @property
    def TEXT_HIGHLIGHT(self) -> str:
        return _color("TEXT_HIGHLIGHT")


# Module-level dynamic color access.
# Every read of PRIMARY, ACCENT, etc. resolves from the current theme
# so changes via set_theme() take effect immediately — no restart needed.
_COLOR_NAMES = {
    "PRIMARY", "ACCENT", "MUTED",
    "SUCCESS", "WARNING", "ERROR", "INFO",
    "TEXT", "TEXT_DIM", "TEXT_HIGHLIGHT",
}


def __getattr__(name: str):
    if name in _COLOR_NAMES:
        return _get_theme()[name]
    if name == "XTERM_THEME":
        return _get_theme()["XTERM"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_color(name: str) -> str:
    """Get a color by name from current theme. Use for dynamic theming."""
    return _get_theme().get(name, "white")


def get_xterm_theme() -> dict:
    """Get xterm theme dict for current theme."""
    return _get_theme()["XTERM"]


def get_xterm_theme_js() -> str:
    """Get xterm theme as JavaScript object literal."""
    xterm = get_xterm_theme()
    lines = ["theme: {"]
    for key, value in xterm.items():
        lines.append(f"          {key}: '{value}',")
    lines.append("        }")
    return "\n".join(lines)
