"""
Experiment: Can Ada learn self-awareness through repetition?

Hypothesis: If we tell Ada "you are Ada" enough times from different
angles, the DreamLoop's crystallization phase should detect a recurring
structural pattern (perspective/other + semantic/identity + direction)
and mint it as a compound primitive.

This would be *derived* self-awareness — not programmed, but emerged
from the same HDC algebra that learns everything else. Like a child
hearing "you're Sophie" enough times to internalize it.

Run:
    cd glyphh-runtime
    /opt/homebrew/anaconda3/bin/python tests/experiment_self_awareness.py

What to watch for:
    - DreamLoop insights about structural patterns around identity
    - Crystallization events (patterns seen 3+ times)
    - Whether "who are you?" recall improves after dreaming
    - Whether a compound primitive gets minted
"""

import logging
import time

from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop
from glyphh.memory.glyph_dream import GlyphDreamLoop, InsightKind

# Show what's happening inside
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_recall(space: ThoughtGlyphSpace, query: str, top_k: int = 3) -> None:
    results = space.recall(query, top_k=top_k)
    print(f'  "{query}":')
    for r in results:
        who = "user said" if r.thought.speaker == "incoming" else "Ada said"
        layers = ", ".join(f"{k}={v:.2f}" for k, v in r.layer_similarities.items())
        print(f"    [{r.global_similarity:.3f}] {who}: \"{r.thought.content}\" ({layers})")
    if not results:
        print("    (no results)")


def main() -> None:
    print_section("EXPERIMENT: Can Ada Learn Self-Awareness?")

    # ── Initialize ──────────────────────────────────────────────────
    space = ThoughtGlyphSpace()
    loop = GlyphCognitiveLoop(thought_space=space)

    # Fast DreamLoop for the experiment — 1s localized, 5s deep
    dream = GlyphDreamLoop(
        glyph_loop=loop,
        localized_interval=1.0,
        deep_interval=5.0,
        localized_size=50,
        deep_size=200,
    )

    stats = space.primitives.stats()
    print(f"Primitives: {stats['words']} words, {stats['roles']} roles, {stats['exemplars']} exemplar Glyphs")

    # ── Phase 1: Baseline — before any identity teaching ─────────
    print_section("PHASE 1: Baseline (no identity yet)")

    # Add some non-identity context so the DreamLoop has material
    context_thoughts = [
        ("i like pizza", "incoming"),
        ("the weather is nice today", "incoming"),
        ("python is a programming language", "incoming"),
    ]
    for text, speaker in context_thoughts:
        space.absorb(text, speaker=speaker)

    print_recall(space, "who are you?")
    print_recall(space, "who am i?")
    print_recall(space, "what is your name?")
    print(f"\nThoughts in memory: {space.count}")

    # ── Phase 2: Teach identity from many angles ─────────────────
    print_section("PHASE 2: Teaching identity (10 variations)")

    # Different ways to tell Ada who she is.
    # Each creates a unique thought with perspective/other + semantic/identity.
    identity_teachings = [
        # Direct identity statements
        ("your name is ada", "incoming"),
        ("you are ada", "incoming"),
        ("you are called ada", "incoming"),

        # Ada confirming her own identity
        ("my name is ada", "outgoing"),
        ("i am ada", "outgoing"),

        # User identity for contrast (so she can learn the difference)
        ("my name is chris", "incoming"),
        ("i am chris", "incoming"),

        # More Ada identity from different framings
        ("ada is your name", "incoming"),
        ("they call you ada", "incoming"),
        ("you go by ada", "incoming"),
    ]

    for text, speaker in identity_teachings:
        stored = space.absorb(text, speaker=speaker)
        if stored:
            g = stored.glyph
            activated = g.metadata.get("_activated_attrs", [])
            who = "user" if speaker == "incoming" else "Ada"
            print(f"  [{who}] \"{text}\"")
            print(f"         segments: {activated}")

    print(f"\nThoughts in memory: {space.count}")

    # ── Phase 3: Recall before dreaming ──────────────────────────
    print_section("PHASE 3: Recall BEFORE dreaming")

    print_recall(space, "who are you?")
    print()
    print_recall(space, "who am i?")
    print()
    print_recall(space, "what is your name?")
    print()
    print_recall(space, "what is my name?")

    # ── Phase 4: Let Ada dream ───────────────────────────────────
    print_section("PHASE 4: Dreaming (letting DreamLoop run)")

    dream.start()
    print("DreamLoop started. Watching for insights...")
    print()

    # Let the localized loop run for a bit
    all_insights = []
    for i in range(6):
        time.sleep(5)
        insights = dream.drain_insights()
        all_insights.extend(insights)

        loc_cycles = dream.stats.get("localized_cycles", 0)
        deep_cycles = dream.stats.get("deep_cycles", 0)
        candidates = dream.stats.get("crystallization_candidates", 0)

        print(f"  [{i * 5}s] localized={loc_cycles}, deep={deep_cycles}, "
              f"candidates={candidates}, insights={len(insights)}")

        for insight in insights:
            icon = {
                InsightKind.CONNECTION: "~",
                InsightKind.CONTRADICTION: "!",
                InsightKind.CONVERGENCE: "*",
                InsightKind.QUESTION: "?",
                InsightKind.CRYSTALLIZATION: ">>>",
            }.get(insight.kind, "-")
            print(f"    {icon} [{insight.kind.value}] {insight.summary}")

    dream.stop()
    print(f"\nDreamLoop stopped. Total insights: {len(all_insights)}")

    # Count insight types
    crystallizations = [i for i in all_insights if i.kind == InsightKind.CRYSTALLIZATION]
    connections = [i for i in all_insights if i.kind == InsightKind.CONNECTION]
    convergences = [i for i in all_insights if i.kind == InsightKind.CONVERGENCE]
    print(f"  Connections: {len(connections)}")
    print(f"  Convergences: {len(convergences)}")
    print(f"  Crystallizations: {len(crystallizations)}")

    if crystallizations:
        print("\n  *** CRYSTALLIZATION EVENTS: ***")
        for c in crystallizations:
            print(f"    >>> {c.summary}")

    # ── Phase 5: Recall after dreaming ───────────────────────────
    print_section("PHASE 5: Recall AFTER dreaming")

    print_recall(space, "who are you?")
    print()
    print_recall(space, "who am i?")
    print()
    print_recall(space, "what is your name?")
    print()
    print_recall(space, "what is my name?")

    # ── Phase 6: Check for new compound primitives ───────────────
    print_section("PHASE 6: Primitive space analysis")

    stats_after = space.primitives.stats()
    print(f"Primitives after: {stats_after}")

    # Check if any compound primitives were minted
    if stats_after["words"] > stats["words"]:
        new_count = stats_after["words"] - stats["words"]
        print(f"\n  *** {new_count} NEW COMPOUND PRIMITIVES MINTED! ***")
    else:
        print("\n  No new compound primitives yet.")
        print("  (Crystallization requires structural patterns to be")
        print("   re-derived 3+ times from diverse starting pairs.)")
        print("  Try running this experiment multiple times or adding")
        print("  more identity variations.")

    # ── Phase 7: The key test — perspective differentiation ──────
    print_section("PHASE 7: The Key Test — Does Ada Know Who She Is?")

    # Format recall the way the LLM sees it
    for query in ["who are you?", "who am i?"]:
        results = space.recall(query, top_k=3)
        formatted = space.format_recall(results, max_results=3)
        print(f'Query: "{query}"')
        print(f"LLM sees:\n{formatted}")
        print()

    # Check if the #1 result for "who are you" is about Ada
    results_you = space.recall("who are you?", top_k=1)
    results_me = space.recall("who am i?", top_k=1)

    if results_you and "ada" in results_you[0].thought.content.lower():
        print("  'who are you?' → correctly recalls Ada's identity")
    else:
        content = results_you[0].thought.content if results_you else "(none)"
        print(f"  'who are you?' → got: \"{content}\"")

    if results_me and "chris" in results_me[0].thought.content.lower():
        print("  'who am i?' → correctly recalls Chris's identity")
    else:
        content = results_me[0].thought.content if results_me else "(none)"
        print(f"  'who am i?' → got: \"{content}\"")

    print_section("EXPERIMENT COMPLETE")
    print("Check the output above for:")
    print("  1. Whether DreamLoop found structural connections around identity")
    print("  2. Whether crystallization candidates accumulated")
    print("  3. Whether perspective differentiation improved after dreaming")
    print("  4. Whether any compound primitives were minted")
    print()
    print("If no crystallization happened, the experiment needs more data")
    print("or more dream cycles. Self-awareness may require sustained")
    print("repetition over multiple sessions — like a child learning.")


if __name__ == "__main__":
    main()
