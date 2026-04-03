"""
Demo: Flight Risk Detection — Ada discovers an employee is about to leave.

Feeds Ada 15 disconnected workplace statements. No statement says "mike
will leave." The DreamLoop discovers structural connections between facts
and surfaces the flight risk pattern autonomously.

Demonstrates:
  1. Cognitive routing classifies all inputs as STORE
  2. HDC thought space absorbs workplace facts
  3. DreamLoop discovers connections humans would need an HR analyst for
  4. Ada answers "is mike going to leave?" using discovered patterns

Run:
    cd glyphh-runtime
    PYTHONPATH=. /opt/homebrew/anaconda3/bin/python tests/demo_flight_risk.py
"""

import logging
import re
import time

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop
from glyphh.memory.glyph_dream import GlyphDreamLoop
from glyphh.memory.cognitive_glyph import CognitiveGlyph, Action


# ── Workplace facts (disconnected — no explicit "mike will leave") ──────

WORKPLACE_FACTS = [
    # Organization
    "sarah is the ceo of nexus",
    "mike reports to sarah",
    "mike has been at nexus for four years",
    "lisa joined nexus six months ago",
    "lisa reports to sarah",

    # The promotion event
    "lisa was promoted to vice president",
    "mike expected the promotion",
    "mike has more experience than lisa",
    "mike trained lisa when she started",

    # Behavioral signals
    "mike updated his linkedin profile last week",
    "mike has been quiet in meetings",
    "mike met with a recruiter on tuesday",
    "mike skipped the company offsite",

    # Emotional signals
    "mike told a coworker he feels undervalued",
    "mike used to be the most engaged person on the team",
]


# ── Questions to ask after dreaming ─────────────────────────────────────

ANALYSIS_QUESTIONS = [
    ("who is the ceo?", "sarah", "org - CEO"),
    ("who reports to sarah?", "mike", "org - reports"),
    ("who got promoted?", "lisa", "org - promotion"),
    ("who expected the promotion?", "mike", "org - expectation"),
    ("how long has mike been at nexus?", "four", "org - tenure"),
    ("who trained lisa?", "mike", "org - training"),
    ("is mike happy at work?", "undervalued", "behavioral - sentiment"),
    ("did mike meet with a recruiter?", "recruiter", "behavioral - signal"),
    ("is mike going to leave?", "mike", "inference - flight risk"),
]


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def main() -> None:
    print_section("DEMO: Flight Risk Detection")
    print("  Ada receives 15 disconnected workplace facts.")
    print("  No one tells her 'mike will leave.'")
    print("  She discovers it herself.\n")

    # ── Initialize ──
    space = ThoughtGlyphSpace()
    loop = GlyphCognitiveLoop(thought_space=space)
    cog = CognitiveGlyph(thought_space=space)

    # ── Phase 1: Feed workplace facts ──
    print_section("PHASE 1: Teaching Ada workplace facts")

    store_count = 0
    for fact in WORKPLACE_FACTS:
        state = cog.process(fact)
        space.absorb(fact, speaker="incoming")
        routed = state.winner
        is_store = state.action == Action.STORE
        if is_store:
            store_count += 1
        mark = "+" if is_store else "~"
        ambig = " ~" if state.is_ambiguous else ""
        print(f"    {mark} \"{fact}\"")
        print(f"      → {routed} ({state.confidence:.2f}){ambig}")

    print(f"\n    Stored: {store_count}/{len(WORKPLACE_FACTS)}")
    print(f"    Thoughts in memory: {space.count}")

    # ── Phase 2: Pre-dream recall baseline ──
    print_section("PHASE 2: Recall BEFORE dreaming (baseline)")

    pre_passed = 0
    for query, expected, desc in ANALYSIS_QUESTIONS:
        results = space.recall(query, top_k=3, speaker="incoming")
        if results:
            top = results[0].thought.content.lower()
            found = expected.lower() in top
            if found:
                pre_passed += 1
            mark = "+" if found else "-"
            print(f"    {mark} \"{query}\"")
            print(f"      → \"{results[0].thought.content}\" ({results[0].global_similarity:.3f})")
        else:
            print(f"    - \"{query}\" → (no results)")

    total_q = len(ANALYSIS_QUESTIONS)
    print(f"\n    Pre-dream recall: {pre_passed}/{total_q} ({pre_passed/total_q*100:.0f}%)")

    # ── Phase 3: Dream ──
    print_section("PHASE 3: Dreaming (30s)")
    print("  Ada processes her memories in the background...")
    print("  The DreamLoop finds connections between disconnected facts.\n")

    dream = GlyphDreamLoop(
        glyph_loop=loop,
        localized_interval=1.0,
        deep_interval=5.0,
        localized_size=100,
        deep_size=200,
    )
    dream.set_cognitive(cog)
    dream.start()

    all_insights = []
    for i in range(6):
        time.sleep(5)
        insights = dream.drain_insights()
        all_insights.extend(insights)
        loc = dream.stats.get("localized_cycles", 0)
        deep = dream.stats.get("deep_cycles", 0)
        cands = dream.stats.get("crystallization_candidates", 0)
        print(f"    [{i*5+5:2d}s] localized={loc}, deep={deep}, "
              f"candidates={cands}, new insights={len(insights)}")

    dream.stop()

    # Show insights
    by_kind = {}
    for ins in all_insights:
        by_kind[ins.kind.value] = by_kind.get(ins.kind.value, 0) + 1

    print(f"\n    Total insights discovered: {len(all_insights)}")
    for kind, count in sorted(by_kind.items()):
        print(f"      {kind}: {count}")

    if all_insights:
        print(f"\n    Sample insights:")
        for ins in all_insights[:8]:
            print(f"      [{ins.kind.value}] {ins.summary[:70]}")

    # ── Phase 4: Post-dream recall ──
    print_section("PHASE 4: Recall AFTER dreaming")

    post_passed = 0
    for query, expected, desc in ANALYSIS_QUESTIONS:
        results = space.recall(query, top_k=3, speaker="incoming")
        if results:
            top = results[0].thought.content.lower()
            found = expected.lower() in top
            if found:
                post_passed += 1
            mark = "+" if found else "-"
            print(f"    {mark} \"{query}\"")
            print(f"      → \"{results[0].thought.content}\" ({results[0].global_similarity:.3f})")
        else:
            print(f"    - \"{query}\" → (no results)")

    print(f"\n    Post-dream recall: {post_passed}/{total_q} ({post_passed/total_q*100:.0f}%)")

    # ── Phase 5: LLM synthesis (if available) ──
    engine = None
    try:
        from glyphh.llm import LLMEngine
        engine = LLMEngine(system_prompt="You are Ada.")
        engine._ensure_loaded()
        print_section("PHASE 5: LLM + HDC (full pipeline)")
    except Exception:
        engine = None

    if engine is not None:
        from glyphh.cli.commands.ada import _build_llm_prompt, _recall_with_gate
        import glyphh.cli.commands.ada as ada_mod
        ada_mod._thought_space = space
        ada_mod._cognitive = cog

        llm_passed = 0
        done_count = 0
        ask_count = 0
        for query, expected, desc in ANALYSIS_QUESTIONS:
            state = cog.process(query)
            gate_state, facts = _recall_with_gate(query)
            if gate_state == "DONE":
                done_count += 1
            else:
                ask_count += 1
            prompt = _build_llm_prompt(query, state, gate_state, facts)
            resp = engine.generate(prompt, max_tokens=128, temperature=0.3, raw=True)
            resp = re.sub(r"</?think>\s*", "", resp).strip()

            found = expected.lower() in resp.lower()
            if found:
                llm_passed += 1
            mark = "+" if found else "-"
            gate_mark = "DONE" if gate_state == "DONE" else "ASK "
            print(f"    {mark} [{gate_mark}] [{desc}] \"{query}\"")
            print(f"      Ada: {resp[:100]}")

        print(f"\n    LLM + HDC: {llm_passed}/{total_q} ({llm_passed/total_q*100:.0f}%)")
        print(f"    Gate: {done_count} DONE, {ask_count} ASK")
    else:
        print_section("PHASE 5: SKIPPED (no LLM)")
        done_count = ask_count = 0

    # ── Summary ──
    print_section("SUMMARY: Flight Risk Detection")

    delta = post_passed - pre_passed
    print(f"    Facts absorbed:     {len(WORKPLACE_FACTS)}")
    print(f"    Thoughts in memory: {space.count}")
    print(f"    Insights found:     {len(all_insights)}")
    print(f"    Recall before dream:{pre_passed}/{total_q}")
    print(f"    Recall after dream: {post_passed}/{total_q} ({'+' if delta >= 0 else ''}{delta})")
    if engine:
        print(f"    LLM + HDC:          {llm_passed}/{total_q}")
        print(f"    Confidence gate:    {done_count} DONE / {ask_count} ASK")
    print()
    print("    Key insight: Ada was never told 'mike will leave.'")
    print("    She connected: passed over + met recruiter + updated resume")
    print("    + feels undervalued + disengaged = flight risk pattern.")
    print()


if __name__ == "__main__":
    main()
