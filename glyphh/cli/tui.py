"""
Glyphh TUI — Terminal User Interface prototype.

A full-screen interactive interface built on Textual that mirrors
the existing shell experience with panels, navigation, and the
Glyphh routing model underneath.

Launch with: glyphh tui
"""

from __future__ import annotations

import os
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Static,
    Input,
    RichLog,
)
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table


# ── Brand colors mapped to Rich/Textual CSS ──────────────────────

PURPLE = "#9333ea"
LIGHT_PURPLE = "#a855f7"
CYAN = "#06b6d4"
MUTED = "#6b7280"
BG = "#0a0a0a"
SURFACE = "#111111"
BORDER = "#1e1e1e"


# ── Sidebar navigation items ─────────────────────────────────────

NAV_SECTIONS = [
    ("AUTH", [
        ("auth login", "Login"),
        ("auth logout", "Logout"),
        ("auth signup", "Sign up"),
        ("auth whoami", "Who am I"),
    ]),
    ("BUILD", [
        ("build init", "Init model"),
        ("build add", "Add data"),
        ("test similarity", "Test similarity"),
        ("package create", "Package"),
    ]),
    ("DEPLOY", [
        ("runtime deploy", "Deploy"),
        ("runtime status", "Status"),
        ("query", "Query"),
        ("procedure list", "Procedures"),
    ]),
    ("DISCOVER", [
        ("hub", "Model Hub"),
        ("models", "My Models"),
        ("docs", "Docs"),
    ]),
]


# ── CSS ───────────────────────────────────────────────────────────

APP_CSS = """
Screen {
    background: #0a0a0a;
    layout: vertical;
    padding: 1 0;
    margin: 0;
}

#app-grid {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 22 1fr;
    grid-gutter: 0;
    height: 1fr;
}

#sidebar {
    width: 22;
    background: #111111;
    border-right: solid #1e1e1e;
    padding: 1 0;
}

#sidebar .nav-section-title {
    color: #6b7280;
    padding: 1 1 0 1;
    text-style: bold;
}

#sidebar .nav-item {
    padding: 0 1 0 2;
    color: #e0e0e0;
    height: 1;
}

#sidebar .nav-item:hover {
    background: #1e1e1e;
    color: #a855f7;
}

#sidebar .nav-item.--highlight {
    background: #9333ea 20%;
    color: #a855f7;
}

#main-area {
    height: 1fr;
}

#output-log {
    height: 1fr;
    padding: 0 1;
    scrollbar-size: 1 1;
}

#input-bar {
    height: 3;
    padding: 0 1;
    background: #111111;
    border-top: solid #1e1e1e;
}

#prompt-input {
    background: #0a0a0a;
    border: tall #9333ea;
    color: #e0e0e0;
    padding: 0 1;
}

#prompt-input:focus {
    border: tall #a855f7;
}

#status-bar {
    height: 1;
    background: #111111;
    color: #6b7280;
    padding: 0 1;
}

#brand-header {
    padding: 0 1;
    color: #a855f7;
    text-style: bold;
    height: 1;
}

#confidence-display {
    dock: right;
    width: auto;
    padding: 0 1;
    color: #6b7280;
}

"""


# ── Widgets ───────────────────────────────────────────────────────

class Sidebar(Vertical):
    """Navigation sidebar with command categories."""

    def compose(self) -> ComposeResult:
        yield Static("glyphh ai", id="brand-header")
        yield Static("")  # spacer

        for section_title, items in NAV_SECTIONS:
            yield Static(section_title, classes="nav-section-title")
            for cmd, label in items:
                item = Static(f"  {label}", classes="nav-item")
                item.cmd = cmd  # stash the command on the widget
                yield item

    def on_click(self, event) -> None:
        """Handle clicks on nav items."""
        widget = event.widget if hasattr(event, 'widget') else None
        if widget and hasattr(widget, "cmd"):
            self.app.execute_command(widget.cmd)


class StatusBar(Horizontal):
    """Bottom status bar showing user + confidence."""

    confidence = reactive(0.0)
    user_display = reactive("not logged in")
    last_method = reactive("")

    def compose(self) -> ComposeResult:
        yield Static("", id="status-text")
        yield Static("", id="confidence-display")

    def watch_confidence(self, value: float) -> None:
        color = "green" if value >= 0.7 else "yellow" if value >= 0.5 else "red"
        display = self.query_one("#confidence-display", Static)
        if value > 0:
            method_str = f"  [{self.last_method}]" if self.last_method else ""
            display.update(Text.from_markup(
                f"confidence: [{color}]{value:.0%}[/{color}]{method_str}"
            ))
        else:
            display.update("")

    def watch_user_display(self, value: str) -> None:
        status = self.query_one("#status-text", Static)
        status.update(Text.from_markup(f"[#6b7280]{value}[/]"))


# ── Main App ──────────────────────────────────────────────────────

class GlyphhTUI(App):
    """Glyphh Terminal User Interface."""

    CSS = APP_CSS
    TITLE = "glyphh ai"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_log", "Clear", show=False),
        Binding("escape", "focus_input", "Input", show=False),
        Binding("f1", "show_help", "Help", show=False),
        Binding("ctrl+w", "toggle_why", "Why", show=False),
    ]

    assistant = None
    last_response = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="app-grid"):
            yield Sidebar(id="sidebar")
            with Vertical(id="main-area"):
                yield RichLog(id="output-log", highlight=True, markup=True, wrap=True)
                with Horizontal(id="input-bar"):
                    yield Input(
                        placeholder="ask a question or type a command...",
                        id="prompt-input",
                    )
                yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Initialize on startup."""
        log = self.query_one("#output-log", RichLog)
        self._print_banner(log)
        self._update_user_status()
        self.query_one("#prompt-input", Input).focus()

    def _print_banner(self, log: RichLog) -> None:
        """Print the welcome banner to the output log."""
        log.write(Text(""))
        log.write(Text("   __ _| |_   _ _ __ | |__ | |__     __ _(_)", style=f"bold {LIGHT_PURPLE}"))
        log.write(Text("  / _` | | | | | '_ \\| '_ \\| '_ \\   / _` | |", style="cyan"))
        log.write(Text(" | (_| | | |_| | |_) | | | | | | | | (_| | |", style="cyan"))
        log.write(Text("  \\__, |_|\\__, | .__/|_| |_|_| |_|  \\__,_|_|", style="bright_cyan"))
        log.write(Text("  |___/   |___/|_|", style="bright_cyan"))
        log.write(Text(""))
        log.write(Text("  when your llm can't afford to be wrong", style="bright_cyan"))
        log.write(Text(""))
        log.write(Text("  F1 help  |  ctrl+w why  |  ctrl+l clear  |  ctrl+q quit", style="#6b7280"))
        log.write(Text(""))

    def _update_user_status(self) -> None:
        """Update the status bar with current user info."""
        try:
            from .auth import get_current_user
            user = get_current_user()
            status_bar = self.query_one("#status-bar", StatusBar)
            if user:
                email = user.get("email", "user")
                status_bar.user_display = f"● {email}"
            else:
                status_bar.user_display = "○ not logged in → auth login"
        except Exception:
            pass

    def _get_assistant(self):
        """Lazy-load the Glyphh assistant."""
        if self.assistant is None:
            try:
                from glyphh.assistant.core import Assistant, AssistantConfig
                runtime_url = os.environ.get("GLYPHH_RUNTIME_URL")
                config = AssistantConfig(runtime_url=runtime_url, threshold=0.5)
                self.assistant = Assistant(config)
                self.assistant.load()
            except Exception as e:
                log = self.query_one("#output-log", RichLog)
                log.write(Text(f"  assistant unavailable: {e}", style="#6b7280"))
                return None
        return self.assistant

    def execute_command(self, cmd: str) -> None:
        """Execute a command string (from sidebar click or input)."""
        log = self.query_one("#output-log", RichLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        # Show what we're running
        log.write(Text(""))
        log.write(Text(f"  glyphh> {cmd}", style=f"bold {LIGHT_PURPLE}"))

        # Check if it's a known CLI command prefix
        from .shell import CLI_COMMAND_PREFIXES
        first_word = cmd.split()[0].lower() if cmd else ""

        if first_word in CLI_COMMAND_PREFIXES:
            self._handle_cli_command(cmd, log, status_bar)
        else:
            self._handle_nl_query(cmd, log, status_bar)

    def _handle_cli_command(self, cmd: str, log: RichLog, status_bar: StatusBar) -> None:
        """Handle a CLI command (help, auth, build, etc.)."""
        first_word = cmd.split()[0].lower()

        if first_word == "help":
            self._show_help_content(log)
        elif first_word in ("exit", "quit", "q"):
            self.exit()
        elif first_word in ("clear", "home"):
            log.clear()
            self._print_banner(log)
        elif first_word == "why":
            self._show_why(log)
        else:
            # For other CLI commands, show what would run
            log.write(Text(f"  → would run: glyphh {cmd}", style="#6b7280"))
            log.write(Text(f"  (CLI command execution coming in next iteration)", style="#6b7280"))

    def _handle_nl_query(self, query: str, log: RichLog, status_bar: StatusBar) -> None:
        """Route a natural language query through the Glyphh model."""
        asst = self._get_assistant()
        if not asst:
            log.write(Text("  no assistant available — try a command instead", style="#6b7280"))
            return

        response = asst.ask(query)
        self.last_response = response

        # Update confidence in status bar
        status_bar.confidence = response.confidence
        status_bar.last_method = response.match_method or ""

        # Show routing trace
        why = response.metadata.get("why", {}) if response.metadata else {}
        extracted = why.get("extracted", {})
        matched_q = why.get("matched_question", "")
        score = why.get("score", response.confidence)

        if extracted:
            verb = extracted.get("verb", "")
            obj = extracted.get("object", "")
            domain = extracted.get("domain", "")
            parts = [f"{k}={v}" for k, v in [("verb", verb), ("object", obj), ("domain", domain)] if v]
            color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"

            if matched_q:
                log.write(Text.from_markup(
                    f'  ⟶ matched "[bold]{matched_q}[/bold]" ([{color}]{score:.0%}[/{color}])'
                ))
            else:
                log.write(Text.from_markup(
                    f'  ⟶ no match above threshold ([{color}]{score:.0%}[/{color}])'
                ))
            log.write(Text(f"    [{' '.join(parts)}]", style="#6b7280"))

        # Show response content
        if response.confidence < (asst.config.threshold if asst else 0.5):
            log.write(Text(""))
            log.write(Text("  I'm not confident enough to answer that.", style="#6b7280"))
            log.write(Text("  Try a command from the sidebar, or type 'help'.", style="#6b7280"))
        else:
            if response.content:
                log.write(Text(""))
                # Render as a card-like panel
                panel = Panel(
                    response.content,
                    title="glyphh ai",
                    border_style=PURPLE,
                    padding=(0, 1),
                )
                log.write(panel)

            if response.command:
                log.write(Text(f"  try: {response.command}", style="cyan"))

            if response.code:
                log.write(Text("  ─── code ───", style="#6b7280"))
                for line in response.code.split("\n"):
                    log.write(Text(f"  {line}", style="green"))

        log.write(Text(""))

    def _show_help_content(self, log: RichLog) -> None:
        """Render help as a Rich table in the output log."""
        table = Table(
            title="Commands",
            border_style=PURPLE,
            show_header=True,
            header_style=f"bold {LIGHT_PURPLE}",
            padding=(0, 1),
        )
        table.add_column("Command", style="white", min_width=18)
        table.add_column("Description", style="#6b7280")

        for section_title, items in NAV_SECTIONS:
            table.add_row(f"[bold #6b7280]{section_title}[/]", "")
            for cmd, label in items:
                table.add_row(f"  {cmd}", label)

        log.write(Text(""))
        log.write(table)
        log.write(Text(""))

    def _show_why(self, log: RichLog) -> None:
        """Show explainability for the last response."""
        if not self.last_response:
            log.write(Text("  ask a question first, then ctrl+w to see why", style="#6b7280"))
            return

        response = self.last_response
        why = response.metadata.get("why", {}) if response.metadata else {}

        if not why:
            log.write(Text("  no explanation available", style="#6b7280"))
            return

        extracted = why.get("extracted", {})
        runners_up = why.get("runners_up", [])

        log.write(Text(""))
        lines = []
        lines.append(f'query:    "{why.get("query", "")}"')

        if extracted.get("match_type") != "exact_phrase":
            lines.append(f'verb:     {extracted.get("verb", "?")}')
            lines.append(f'object:   {extracted.get("object", "?")}')
            lines.append(f'domain:   {extracted.get("domain", "?")}')
        else:
            lines.append("match:    exact phrase")

        matched_q = why.get("matched_question")
        if matched_q:
            lines.append(f'matched:  "{matched_q}"')
        lines.append(f'score:    {why.get("score", 0):.2f}')

        panel = Panel(
            "\n".join(lines),
            title="why this answer",
            border_style="cyan",
            padding=(0, 1),
        )
        log.write(panel)

        # Similarity table
        if runners_up:
            table = Table(
                title="similarity (top 5)",
                border_style="#6b7280",
                show_header=True,
                header_style="bold #6b7280",
            )
            table.add_column("#", width=3)
            table.add_column("Score", width=6)
            table.add_column("Question")

            all_entries = []
            if matched_q:
                all_entries.append({"score": why.get("score", 0), "question": matched_q})
            all_entries.extend(runners_up)

            for rank, entry in enumerate(all_entries[:5], 1):
                s = entry["score"]
                color = "green" if s >= 0.7 else "yellow" if s >= 0.5 else "red"
                q = entry["question"][:40]
                marker = ">" if rank == 1 and matched_q else " "
                table.add_row(f"{marker}{rank}", f"[{color}]{s:.2f}[/]", q)

            log.write(table)

        log.write(Text(""))

    # ── Input handling ────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in the input bar."""
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        self.execute_command(text)

    # ── Actions ───────────────────────────────────────────────────

    def action_clear_log(self) -> None:
        log = self.query_one("#output-log", RichLog)
        log.clear()
        self._print_banner(log)

    def action_focus_input(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def action_show_help(self) -> None:
        log = self.query_one("#output-log", RichLog)
        self._show_help_content(log)

    def action_toggle_why(self) -> None:
        log = self.query_one("#output-log", RichLog)
        self._show_why(log)


# ── CLI entry point ───────────────────────────────────────────────

def run_tui():
    """Launch the Glyphh TUI."""
    import os
    # Set TEXTUAL_DRIVER env to avoid alternate screen sizing issues
    os.environ.setdefault("TEXTUAL_MARGIN", "0")
    app = GlyphhTUI()
    app.run(inline=False, size=None)


if __name__ == "__main__":
    run_tui()
