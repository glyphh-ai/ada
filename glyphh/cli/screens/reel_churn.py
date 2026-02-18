"""
Customer churn prediction demo reel.

Shows the Glyphh flow for a churn analysis query:
  CS executive asks which customers are likely to churn →
  Glyphh maps intent to stored procedure via VSA → runs
  churn-risk-list procedure → low confidence (no timeframe) →
  clarify → re-process with 30d window → returns ranked list
  of at-risk customers → LLM formats actionable report
"""

import time
import click
from .. import theme
from ..streaming import stream_text, stream_echo
from .demo_utils import (
    header, status_line, flow_arrow, type_prompt, similarity_tree,
    run_reel, SYSTEM_CPS, FLOW_CPS,
)


def _scene_question():
    header("SCENARIO")
    stream_text("  A customer success executive asks their", fg=theme.MUTED, cps=FLOW_CPS)
    stream_text("  internal AI assistant a broad question.", fg=theme.MUTED, cps=FLOW_CPS)
    click.echo()
    type_prompt("what customers are likely to churn?")


def _scene_first_pass():
    header("GLYPHH PROCESSING")

    status_line("→", "input", '"what customers are likely to churn?"')
    flow_arrow()
    status_line("◆", "intent → procedure (vsa)",
                "verb=list  object=churn-risk  domain=customers",
                value_color=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_text("    ↳ vsa encodes the intent and matches it to the right", fg=theme.MUTED, cps=SYSTEM_CPS)
    stream_text("      stored procedure in the model — no LLM needed here", fg=theme.MUTED, cps=SYSTEM_CPS)
    flow_arrow()
    status_line("◆", "normalize (stored proc)",
                '"LIST SIMILAR TO \'churn-risk\' ORDER BY risk DESC"',
                value_color=theme.TEXT_HIGHLIGHT)
    flow_arrow()
    status_line("◆", "GQL (model procedure)",
                "LIST SIMILAR TO 'churn-risk-pattern' "
                "WHERE type='behavior' ORDER BY risk DESC LIMIT 20",
                value_color=theme.ACCENT)
    flow_arrow()

    status_line("◆", "similarity (glyphh vsa)", "cortex → layer → segment → role",
                value_color=theme.TEXT)
    click.echo()

    stream_text("    results: 14 customers matched across risk tiers", fg=theme.TEXT, cps=SYSTEM_CPS)
    click.echo()

    stream_echo(("    [0.72] ", theme.WARNING, False),
                ("Acme Corp", theme.TEXT, True),
                ("         logins -40%, downgraded plan, 2 failed charges", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(
        cortex="0.72",
        layer="behavioral 0.78 · financial 0.65",
        segment="engagement 0.76 · billing 0.68",
        role="login_freq=declining 0.89 · plan=downgraded 0.82 · payment=failed 0.78")

    stream_echo(("    [0.68] ", theme.WARNING, False),
                ("Northwind Ltd", theme.TEXT, True),
                ("      feature usage -60%, no support contact in 45d", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(
        cortex="0.68",
        layer="behavioral 0.74 · financial 0.61",
        segment="engagement 0.71 · support 0.55",
        role="feature_use=minimal 0.85 · support=silent 0.72 · nps=declining 0.48")

    stream_echo(("    [0.51] ", theme.WARNING, False),
                ("Contoso Inc", theme.TEXT, False),
                ("        mixed signals — usage stable but billing issues", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(
        cortex="0.51",
        layer="behavioral 0.55 · financial 0.47",
        segment="engagement 0.58 · billing 0.52",
        role="login_freq=stable 0.42 · payment=late 0.65 · tenure=6mo 0.38")

    stream_echo(("    ... ", theme.MUTED, False),
                ("+ 11 more customers below 0.50 threshold", theme.MUTED, False),
                cps=SYSTEM_CPS)
    click.echo()

    stream_echo(("  ⚠ ", theme.WARNING, False),
                ("confidence: ", theme.MUTED, False), ("58%", theme.WARNING, False),
                ("  — broad match but no timeframe specified", theme.MUTED, False),
                cps=SYSTEM_CPS)


def _scene_clarify():
    header("CLARIFICATION")
    click.echo()
    stream_echo(("  ? ", theme.TEXT_HIGHLIGHT, False),
                ("What timeframe are you looking at? ", theme.MUTED, False),
                ("Next 30 days", theme.TEXT_HIGHLIGHT, False),
                (", ", theme.MUTED, False),
                ("next quarter", theme.TEXT_HIGHLIGHT, False),
                (", or ", theme.MUTED, False),
                ("all time horizons", theme.TEXT_HIGHLIGHT, False),
                ("?", theme.MUTED, False), cps=SYSTEM_CPS)
    click.echo()
    time.sleep(0.6)
    type_prompt("next 30 days, highest risk first")


def _scene_second_pass():
    header("RE-PROCESSING")

    status_line("→", "input", '"next 30 days, highest risk first"')
    flow_arrow()
    status_line("◆", "merge",
                "original + timeframe=30d + sort=risk_desc",
                value_color=theme.TEXT_HIGHLIGHT)
    flow_arrow()
    status_line("◆", "GQL (model procedure)",
                "LIST SIMILAR TO 'imminent-churn-pattern' "
                "WHERE window='30d' AND type='behavior' "
                "ORDER BY risk DESC LIMIT 10",
                value_color=theme.ACCENT)
    flow_arrow()

    status_line("◆", "similarity (glyphh vsa)", "cortex → layer → segment → role",
                value_color=theme.TEXT)
    click.echo()

    stream_text("    results: 6 customers at imminent risk (30d window)", fg=theme.TEXT, cps=SYSTEM_CPS)
    click.echo()

    stream_echo(("    [0.93] ", theme.SUCCESS, False),
                ("Acme Corp", theme.TEXT, True),
                ("         usage collapse + billing failure", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(
        cortex="0.93",
        layer="behavioral 0.95 · financial 0.90",
        segment="engagement 0.94 · billing 0.91",
        role="login_freq=near-zero 0.97 · plan=downgraded 0.92 · "
             "payment=at-risk 0.89")

    stream_echo(("    [0.87] ", theme.SUCCESS, False),
                ("Northwind Ltd", theme.TEXT, True),
                ("      silent disengagement pattern", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(
        cortex="0.87",
        layer="behavioral 0.91 · financial 0.82",
        segment="engagement 0.90 · support 0.78",
        role="feature_use=zero 0.94 · support=no-contact 0.88 · "
             "nps=not-submitted 0.72")

    stream_echo(("    [0.79] ", theme.SUCCESS, False),
                ("Fabrikam Co", theme.TEXT, True),
                ("        rapid downgrade + cancellation signal", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(
        cortex="0.79",
        layer="behavioral 0.82 · financial 0.76",
        segment="engagement 0.80 · billing 0.78",
        role="plan=cancelled-pending 0.91 · usage=declining 0.84 · "
             "tenure=4mo 0.55")

    stream_echo(("    ... ", theme.MUTED, False),
                ("+ 3 more (Woodgrove Bank 0.74, Tailspin 0.71, Datum Corp 0.68)", theme.MUTED, False),
                cps=SYSTEM_CPS)
    click.echo()

    stream_echo(("  ✓ ", theme.SUCCESS, False),
                ("confidence: ", theme.MUTED, False), ("91%", theme.SUCCESS, False),
                ("  — strong 30d churn signals across 6 accounts", theme.MUTED, False),
                cps=SYSTEM_CPS)


def _scene_llm_response():
    header("LLM FORMATTING RESPONSE")
    stream_text("  ─── response ───", fg=theme.MUTED, cps=SYSTEM_CPS)
    click.echo()

    for line in [
        "6 customers are at high risk of churning in the next 30 days",
        "(91% overall confidence).",
        "",
        "Top 3 — immediate action required:",
        "",
        "  1. Acme Corp (93% risk)",
        "     Logins near zero, downgraded Pro → Starter, 2 failed payments.",
        "     → Executive outreach within 48h, offer Pro extension.",
        "",
        "  2. Northwind Ltd (87% risk)",
        "     Zero feature usage for 3 weeks, no support contact in 45d.",
        "     → Re-engagement campaign, schedule success check-in.",
        "",
        "  3. Fabrikam Co (79% risk)",
        "     Cancellation pending, rapid usage decline since onboarding.",
        "     → Onboarding rescue — assign dedicated CSM.",
        "",
        "Also flagged: Woodgrove Bank (74%), Tailspin (71%), Datum Corp (68%).",
        "",
        "Pattern: 4 of 6 match the 'silent disengagement' churn pattern",
        "seen in 62% of lost accounts this quarter.",
    ]:
        if line:
            stream_text(f"  {line}", fg=theme.TEXT, cps=FLOW_CPS)
        else:
            click.echo()

    click.echo()
    stream_text("  ─── cited from: customer_health_v2.glyphh ───",
                fg=theme.MUTED, cps=SYSTEM_CPS)
    click.echo()
    stream_echo(("  confidence: ", theme.MUTED, False), ("91%", theme.SUCCESS, False),
                cps=SYSTEM_CPS)


def _scene_outro():
    click.echo()
    stream_echo(("  your llm handles ambiguity. ", theme.MUTED, False),
                ("glyphh handles facts.", "cyan", True), cps=FLOW_CPS)
    click.echo()
    stream_echo(("  -> ", "cyan", False), ("demo --reel product", theme.TEXT_HIGHLIGHT, False),
                (" product search demo", theme.MUTED, False), cps=SYSTEM_CPS)
    stream_echo(("  -> ", "cyan", False), ("auth signup", theme.TEXT_HIGHLIGHT, False),
                (" create your account", theme.MUTED, False), cps=SYSTEM_CPS)
    click.echo()


SCENES = [
    ("CS exec asks about churn risk",    _scene_question),
    ("glyphh maps intent → procedure",   _scene_first_pass),
    ("no timeframe → clarify",           _scene_clarify),
    ("re-process with 30d window",       _scene_second_pass),
    ("llm formats ranked report",        _scene_llm_response),
    ("done",                             _scene_outro),
]


def show_reel_churn():
    """Run the customer churn prediction demo reel."""
    run_reel(SCENES)
