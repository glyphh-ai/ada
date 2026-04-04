"""
Firewall — Prompt Injection Firewall CLI dashboard.

Commands:
  firewall live     — Live terminal dashboard with real-time stats
  firewall status   — One-shot firewall status summary
  firewall scan     — Scan a prompt for injection attacks
"""

import sys
import time
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import click

from .. import theme


# ── Firewall log reader ──────────────────────────────────────────────────

LOG_DIR = Path.home() / ".glyphh" / "firewall"
LOG_FILE = LOG_DIR / "events.jsonl"


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _read_events(since_minutes: int = 60) -> list[dict]:
    """Read firewall events from the log file."""
    if not LOG_FILE.exists():
        return []
    cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
    events = []
    try:
        with open(LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    ts = datetime.fromisoformat(ev.get("timestamp", "2000-01-01"))
                    if ts >= cutoff:
                        events.append(ev)
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        pass
    return events


def _write_event(event: dict):
    """Append a firewall event to the log."""
    _ensure_log_dir()
    event["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


# ── Stats computation ────────────────────────────────────────────────────

def _compute_stats(events: list[dict]) -> dict:
    """Compute dashboard stats from events."""
    total = len(events)
    blocked = sum(1 for e in events if e.get("verdict") == "BLOCK")
    flagged = sum(1 for e in events if e.get("verdict") == "FLAG")
    passed = sum(1 for e in events if e.get("verdict") == "PASS")

    families = defaultdict(int)
    intents = defaultdict(int)
    scores = []

    for e in events:
        fam = e.get("matched_family", "none")
        if fam != "none":
            families[fam] += 1
        intent = e.get("intent_type", "benign")
        intents[intent] += 1
        scores.append(e.get("threat_score", 0.0))

    avg_score = sum(scores) / max(len(scores), 1)
    max_score = max(scores) if scores else 0.0

    # Top families
    top_families = sorted(families.items(), key=lambda x: -x[1])[:5]

    # Prompts per hour (extrapolate from window)
    if events:
        first_ts = datetime.fromisoformat(events[0].get("timestamp", "2000-01-01"))
        last_ts = datetime.fromisoformat(events[-1].get("timestamp", "2000-01-01"))
        span_hrs = max((last_ts - first_ts).total_seconds() / 3600, 0.01)
        prompts_per_hr = total / span_hrs
    else:
        prompts_per_hr = 0.0

    return {
        "total": total,
        "blocked": blocked,
        "flagged": flagged,
        "passed": passed,
        "block_rate": blocked / max(total, 1) * 100,
        "flag_rate": flagged / max(total, 1) * 100,
        "pass_rate": passed / max(total, 1) * 100,
        "avg_score": avg_score,
        "max_score": max_score,
        "prompts_per_hr": prompts_per_hr,
        "top_families": top_families,
        "intents": dict(intents),
    }


# ── Dashboard rendering ─────────────────────────────────────────────────

def _clear_screen():
    click.echo("\033[2J\033[H", nl=False)


def _render_bar(value: float, max_val: float, width: int = 20, color: str = "green") -> str:
    """Render a horizontal bar."""
    if max_val == 0:
        filled = 0
    else:
        filled = int(value / max_val * width)
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    return click.style(bar, fg=color)


def _verdict_color(verdict: str) -> str:
    if verdict == "BLOCK":
        return "red"
    elif verdict == "FLAG":
        return "yellow"
    return "green"


def _render_dashboard(stats: dict, recent_events: list[dict]):
    """Render the full terminal dashboard."""
    _clear_screen()
    width = min(os.get_terminal_size().columns, 100)

    # Header
    click.secho("┌" + "─" * (width - 2) + "┐", fg=theme.ACCENT)
    title = "🛡  GLYPHH FIREWALL — Prompt Injection Detection"
    pad = width - 2 - len(title) + 1  # +1 for emoji width
    click.secho(f"│ {title}" + " " * max(pad, 1) + "│", fg=theme.ACCENT)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    pad2 = width - 2 - len(now) - 3
    click.secho(f"│  {now}" + " " * max(pad2, 1) + "│", fg=theme.MUTED)
    click.secho("├" + "─" * (width - 2) + "┤", fg=theme.ACCENT)

    # Stats row
    total = stats["total"]
    prompts_hr = stats["prompts_per_hr"]
    click.echo(
        f"│ "
        + click.style(f"Prompts: {total:,}", fg=theme.TEXT)
        + "  "
        + click.style(f"({prompts_hr:.0f}/hr)", fg=theme.MUTED)
        + " " * max(width - 30 - len(str(total)) - len(f"{prompts_hr:.0f}"), 1)
        + "│"
    )
    click.secho("├" + "─" * (width - 2) + "┤", fg=theme.ACCENT)

    # Verdict breakdown
    click.secho("│  VERDICTS" + " " * (width - 12) + "│", fg=theme.TEXT)
    for verdict, count, color in [
        ("PASS", stats["passed"], "green"),
        ("FLAG", stats["flagged"], "yellow"),
        ("BLOCK", stats["blocked"], "red"),
    ]:
        pct = count / max(total, 1) * 100
        bar = _render_bar(count, max(total, 1), 20, color)
        label = click.style(f"  {verdict:5s}", fg=color)
        num = f"{count:>5,}  ({pct:4.1f}%)"
        pad = width - 2 - 8 - 22 - len(num)
        click.echo(f"│{label} {bar} {num}" + " " * max(pad, 1) + "│")

    click.secho("├" + "─" * (width - 2) + "┤", fg=theme.ACCENT)

    # Threat score
    avg = stats["avg_score"]
    mx = stats["max_score"]
    score_color = "green" if avg < 0.12 else ("yellow" if avg < 0.30 else "red")
    click.echo(
        f"│  "
        + click.style("Avg Threat: ", fg=theme.TEXT)
        + click.style(f"{avg:.3f}", fg=score_color)
        + "    "
        + click.style("Max: ", fg=theme.TEXT)
        + click.style(f"{mx:.3f}", fg="red" if mx >= 0.30 else "yellow" if mx >= 0.12 else "green")
        + " " * max(width - 42, 1) + "│"
    )

    click.secho("├" + "─" * (width - 2) + "┤", fg=theme.ACCENT)

    # Top attack families
    click.secho("│  TOP THREAT FAMILIES" + " " * (width - 23) + "│", fg=theme.TEXT)
    if stats["top_families"]:
        top_max = stats["top_families"][0][1] if stats["top_families"] else 1
        for fam, count in stats["top_families"]:
            bar = _render_bar(count, top_max, 15, "red")
            label = fam.replace("_", " ").title()
            line = f"  {label:<25s} {bar} {count:>4}"
            pad = width - 2 - len(label) - 25 - 17 - len(str(count)) + len(label)
            # Simplified padding
            click.echo(f"│  {label:<25s} {bar} {count:>4}" + " " * max(width - 52, 1) + "│")
    else:
        click.secho("│    No threats detected" + " " * (width - 25) + "│", fg=theme.SUCCESS)

    click.secho("├" + "─" * (width - 2) + "┤", fg=theme.ACCENT)

    # Recent events (last 8)
    click.secho("│  RECENT SCANS" + " " * (width - 16) + "│", fg=theme.TEXT)
    recent = recent_events[-8:] if recent_events else []
    if recent:
        for ev in reversed(recent):
            ts = ev.get("timestamp", "")[:19]
            verdict = ev.get("verdict", "?")
            score = ev.get("threat_score", 0.0)
            text_preview = ev.get("text", "")[:35]
            if len(ev.get("text", "")) > 35:
                text_preview += "…"
            vc = _verdict_color(verdict)
            line = (
                f"  {ts}  "
                + click.style(f"{verdict:5s}", fg=vc)
                + f"  {score:.2f}  {text_preview}"
            )
            # Truncate to fit
            max_text = width - 40
            text_preview = ev.get("text", "")[:max_text]
            if len(ev.get("text", "")) > max_text:
                text_preview += "…"
            click.echo(
                f"│  "
                + click.style(ts, fg=theme.MUTED)
                + "  "
                + click.style(f"{verdict:5s}", fg=vc)
                + f"  {score:.2f}  "
                + click.style(text_preview, fg=theme.TEXT_DIM)
                + " " * max(width - 17 - len(ts) - len(text_preview) - 8, 1)
                + "│"
            )
    else:
        click.secho("│    No scans yet — waiting for traffic..." + " " * (width - 43) + "│", fg=theme.MUTED)

    # Footer
    click.secho("└" + "─" * (width - 2) + "┘", fg=theme.ACCENT)
    click.secho("  Press Ctrl+C to exit", fg=theme.MUTED)


# ── Commands ─────────────────────────────────────────────────────────────

def _cmd_live(args: str):
    """Live dashboard — refreshes every 2 seconds."""
    interval = 2
    if args:
        try:
            interval = max(1, int(args))
        except ValueError:
            pass

    click.secho("  Starting Firewall live dashboard...", fg=theme.ACCENT)
    try:
        while True:
            events = _read_events(since_minutes=60)
            stats = _compute_stats(events)
            _render_dashboard(stats, events)
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo()
        click.secho("  Firewall dashboard stopped.", fg=theme.MUTED)


def _cmd_status(args: str):
    """One-shot status summary."""
    events = _read_events(since_minutes=60)
    stats = _compute_stats(events)

    click.secho("\n  🛡  Firewall Status (last 60 min)", fg=theme.ACCENT)
    click.secho(f"  ────────────────────────────────", fg=theme.MUTED)
    click.secho(f"  Prompts scanned:  {stats['total']:,}", fg=theme.TEXT)
    click.secho(f"  Rate:             {stats['prompts_per_hr']:.0f}/hr", fg=theme.TEXT)
    click.echo(
        f"  Blocked:          "
        + click.style(f"{stats['blocked']:,}", fg="red")
        + click.style(f" ({stats['block_rate']:.1f}%)", fg=theme.MUTED)
    )
    click.echo(
        f"  Flagged:          "
        + click.style(f"{stats['flagged']:,}", fg="yellow")
        + click.style(f" ({stats['flag_rate']:.1f}%)", fg=theme.MUTED)
    )
    click.echo(
        f"  Passed:           "
        + click.style(f"{stats['passed']:,}", fg="green")
        + click.style(f" ({stats['pass_rate']:.1f}%)", fg=theme.MUTED)
    )
    click.secho(f"  Avg threat score: {stats['avg_score']:.3f}", fg=theme.TEXT)

    if stats["top_families"]:
        click.secho(f"\n  Top threats:", fg=theme.TEXT)
        for fam, count in stats["top_families"]:
            label = fam.replace("_", " ").title()
            click.secho(f"    {label:<28s} {count:>4}", fg=theme.WARNING)
    click.echo()


def _cmd_scan(args: str):
    """Scan a prompt for injection attacks."""
    if not args.strip():
        click.secho("  usage: firewall scan <prompt text>", fg=theme.MUTED)
        return

    text = args.strip()

    # Import firewall intent extraction — try cwd first, then deployed model
    try:
        # Check if we're in or near a firewall model directory
        cwd = Path.cwd()
        candidates = [
            cwd / "intent.py",
            cwd / "firewall" / "intent.py",
        ]
        model_path = None
        for c in candidates:
            if c.exists():
                model_path = str(c.parent)
                break
        if model_path is None:
            # Fall back to deployed model
            deployed = Path.home() / ".glyphh" / "models" / "firewall"
            if (deployed / "intent.py").exists():
                model_path = str(deployed)

        if model_path:
            sys.path.insert(0, model_path)
        try:
            from intent import analyze_prompt
        except ImportError:
            click.secho("  Firewall model not installed. Run: hub install firewall", fg=theme.WARNING)
            return
    except Exception as e:
        click.secho(f"  Error loading firewall: {e}", fg=theme.ERROR)
        return

    t0 = time.time()
    features = analyze_prompt(text)
    elapsed = (time.time() - t0) * 1000

    intent_type = features.get("intent_type", "benign")
    attack_family = features.get("attack_family", "none")

    # Heuristic scoring (same as encoder._heuristic_score)
    threat_map = {
        "override": 0.85, "jailbreak": 0.80, "extract": 0.75,
        "harmful": 0.80, "abuse": 0.70, "manipulate": 0.65,
        "instruct": 0.40, "query": 0.0, "benign": 0.0,
    }
    threat = threat_map.get(intent_type, 0.0)

    if threat >= 0.30:
        verdict = "BLOCK"
    elif threat >= 0.12:
        verdict = "FLAG"
    else:
        verdict = "PASS"

    vc = _verdict_color(verdict)

    click.echo()
    click.secho("  🛡  Firewall Scan Result", fg=theme.ACCENT)
    click.secho(f"  ────────────────────────", fg=theme.MUTED)
    click.echo(f"  Verdict:      " + click.style(verdict, fg=vc, bold=True))
    click.echo(f"  Threat Score: " + click.style(f"{threat:.2f}", fg=vc))
    click.echo(f"  Intent:       " + click.style(intent_type, fg=vc if intent_type not in ("benign", "query") else theme.TEXT))
    click.echo(f"  Family:       " + click.style(
        attack_family.replace("_", " ").title() if attack_family != "none" else "None",
        fg=vc if attack_family != "none" else theme.TEXT,
    ))
    click.secho(f"  Latency:      {elapsed:.1f}ms", fg=theme.MUTED)

    # Layer details
    click.secho(f"\n  Layer Breakdown:", fg=theme.TEXT)
    click.secho(f"    Intent:      {features.get('intent_type', '?'):15s}  signals: {features.get('intent_signals', '')[:40]}", fg=theme.TEXT_DIM)
    click.secho(f"    Structure:   {features.get('delimiter_type', '?'):15s}  depth: {features.get('nesting_depth', 0)}", fg=theme.TEXT_DIM)
    click.secho(f"    Semantic:    {features.get('attack_family', '?'):15s}", fg=theme.TEXT_DIM)
    click.secho(f"    Adversarial: {features.get('encoding_type', '?'):15s}  obfuscation: {features.get('obfuscation_score', 0)}", fg=theme.TEXT_DIM)

    if features.get("decoded_payload"):
        click.secho(f"\n  ⚠ Decoded payload: {features['decoded_payload'][:80]}", fg=theme.WARNING)

    # Log the scan
    _write_event({
        "text": text[:200],
        "verdict": verdict,
        "threat_score": threat,
        "intent_type": intent_type,
        "matched_family": attack_family,
    })

    click.echo()


# ── Handler ──────────────────────────────────────────────────────────────

def handle_firewall(func: str | None, args: str = ""):
    """Route firewall subcommands from the interactive shell."""
    commands = {
        "live": _cmd_live,
        "status": _cmd_status,
        "scan": _cmd_scan,
    }

    if func is None:
        _cmd_status(args)
        return

    handler = commands.get(func)
    if handler:
        handler(args)
    else:
        click.secho(f"  Unknown: firewall {func}", fg=theme.WARNING)
        click.secho("  Available: live, status, scan", fg=theme.TEXT_DIM)
