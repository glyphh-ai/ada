"""
Product search demo reel.

Shows the Glyphh flow for a natural language product query:
  user question → extract → GQL → low confidence → clarify →
  re-process → high confidence → LLM response
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
    stream_text("  A user asks a natural language question", fg=theme.MUTED, cps=FLOW_CPS)
    stream_text("  through their LLM-powered app.", fg=theme.MUTED, cps=FLOW_CPS)
    click.echo()
    type_prompt("what shoes are good for running on trails?")


def _scene_first_pass():
    header("GLYPHH PROCESSING")

    status_line("→", "input", '"what shoes are good for running on trails?"')
    flow_arrow()
    status_line("◆", "intent → procedure (vsa)",
                "verb=find  object=shoes  domain=footwear",
                value_color=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_text("    ↳ vsa encodes the intent and matches it to the right", fg=theme.MUTED, cps=SYSTEM_CPS)
    stream_text("      stored procedure in the model — no LLM needed here", fg=theme.MUTED, cps=SYSTEM_CPS)
    flow_arrow()
    status_line("◆", "normalize (stored proc)", '"FIND SIMILAR TO \'trail running shoes\'"',
                value_color=theme.TEXT_HIGHLIGHT)
    flow_arrow()
    status_line("◆", "GQL (model procedure)",
                "FIND SIMILAR TO 'trail running shoes' WHERE category='footwear'",
                value_color=theme.ACCENT)
    flow_arrow()

    status_line("◆", "similarity (glyphh vsa)", "cortex → layer → segment → role",
                value_color=theme.TEXT)
    click.echo()

    stream_echo(("    [0.62] ", theme.WARNING, False),
                ("Trail Hiking Boots", theme.TEXT, False),
                ("  waterproof boots for mountain trails", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(cortex="0.62", layer="semantic 0.71 · physical 0.48",
                    segment="attributes 0.65 · usage 0.58",
                    role="terrain=trail 0.91 · type=boot 0.33 · weight=heavy 0.22")

    stream_echo(("    [0.58] ", theme.WARNING, False),
                ("Running Shoes Pro", theme.TEXT, False),
                ("   lightweight cushioned sole", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(cortex="0.58", layer="semantic 0.64 · physical 0.51",
                    segment="attributes 0.60 · usage 0.55",
                    role="terrain=road 0.42 · type=shoe 0.88 · weight=light 0.71")

    stream_echo(("    [0.41] ", "red", False),
                ("Cross Trainers", theme.TEXT, False),
                ("        multi-surface grip", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(cortex="0.41", layer="semantic 0.45 · physical 0.36",
                    segment="attributes 0.39 · usage 0.43",
                    role="terrain=gym 0.18 · type=shoe 0.82 · weight=medium 0.44")
    click.echo()

    stream_echo(("  ⚠ ", theme.WARNING, False),
                ("confidence: ", theme.MUTED, False), ("58%", theme.WARNING, False),
                ("  — ambiguous: trails + running overlap two categories", theme.MUTED, False),
                cps=SYSTEM_CPS)


def _scene_clarify():
    header("CLARIFICATION")
    click.echo()
    stream_echo(("  ? ", theme.TEXT_HIGHLIGHT, False),
                ("Are you looking for ", theme.MUTED, False),
                ("trail running shoes", theme.TEXT_HIGHLIGHT, False),
                (" or ", theme.MUTED, False),
                ("hiking boots for trails", theme.TEXT_HIGHLIGHT, False),
                ("?", theme.MUTED, False), cps=SYSTEM_CPS)
    click.echo()
    time.sleep(0.6)
    type_prompt("trail running shoes, lightweight")


def _scene_second_pass():
    header("RE-PROCESSING")

    status_line("→", "input", '"trail running shoes, lightweight"')
    flow_arrow()
    status_line("◆", "merge", "original + clarification → refined query",
                value_color=theme.TEXT_HIGHLIGHT)
    flow_arrow()
    status_line("◆", "GQL (model procedure)",
                "FIND SIMILAR TO 'lightweight trail running shoes' "
                "WHERE category='footwear' AND weight='light'",
                value_color=theme.ACCENT)
    flow_arrow()

    status_line("◆", "similarity (glyphh vsa)", "cortex → layer → segment → role",
                value_color=theme.TEXT)
    click.echo()

    stream_echo(("    [0.94] ", theme.SUCCESS, False),
                ("Trail Runner Lite", theme.TEXT, True),
                ("     lightweight trail shoe, grip sole, breathable", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(cortex="0.94", layer="semantic 0.96 · physical 0.91",
                    segment="attributes 0.95 · usage 0.93",
                    role="terrain=trail 0.98 · type=shoe 0.97 · weight=light 0.95")

    stream_echo(("    [0.87] ", theme.SUCCESS, False),
                ("Running Shoes Pro", theme.TEXT, False),
                ("    cushioned, 8oz, road-to-trail", theme.MUTED, False),
                cps=SYSTEM_CPS)
    similarity_tree(cortex="0.87", layer="semantic 0.89 · physical 0.84",
                    segment="attributes 0.88 · usage 0.85",
                    role="terrain=road-trail 0.78 · type=shoe 0.97 · weight=light 0.92")
    click.echo()

    stream_echo(("  ✓ ", theme.SUCCESS, False),
                ("confidence: ", theme.MUTED, False), ("94%", theme.SUCCESS, False),
                ("  — deterministic match with citation", theme.MUTED, False),
                cps=SYSTEM_CPS)


def _scene_llm_response():
    header("LLM FORMATTING RESPONSE")
    stream_text("  ─── response ───", fg=theme.MUTED, cps=SYSTEM_CPS)
    click.echo()

    for line in [
        "Based on your product catalog, the best match for",
        "lightweight trail running shoes is the Trail Runner Lite",
        "(94% confidence).",
        "",
        "It's a breathable, lightweight shoe with aggressive grip",
        "designed specifically for trail running. At 7oz it's one",
        "of the lightest in the catalog.",
        "",
        "Runner-up: Running Shoes Pro (87%) — a versatile",
        "road-to-trail option if you want pavement flexibility.",
    ]:
        if line:
            stream_text(f"  {line}", fg=theme.TEXT, cps=FLOW_CPS)
        else:
            click.echo()

    click.echo()
    stream_text("  ─── cited from: product_catalog_v3.glyphh ───",
                fg=theme.MUTED, cps=SYSTEM_CPS)
    click.echo()
    stream_echo(("  confidence: ", theme.MUTED, False), ("94%", theme.SUCCESS, False),
                cps=SYSTEM_CPS)


def _scene_outro():
    click.echo()
    stream_echo(("  your llm handles ambiguity. ", theme.MUTED, False),
                ("glyphh handles facts.", "cyan", True), cps=FLOW_CPS)
    click.echo()
    stream_echo(("  -> ", "cyan", False), ("2", theme.TEXT_HIGHLIGHT, False),
                (" build your first model", theme.MUTED, False), cps=SYSTEM_CPS)
    stream_echo(("  -> ", "cyan", False), ("3", theme.TEXT_HIGHLIGHT, False),
                (" browse pre-built models", theme.MUTED, False), cps=SYSTEM_CPS)
    click.echo()


SCENES = [
    ("user asks a question",       _scene_question),
    ("glyphh processes the query",  _scene_first_pass),
    ("low confidence → clarify",    _scene_clarify),
    ("re-process with context",     _scene_second_pass),
    ("llm formats the answer",      _scene_llm_response),
    ("done",                        _scene_outro),
]


def show_reel_product():
    """Run the product search demo reel."""
    run_reel(SCENES)
