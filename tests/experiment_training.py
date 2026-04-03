"""
Experiment: Full Ada pipeline — CognitiveGlyph + HDC Recall + LLM Synthesis.

Tests the complete 3-tier architecture:
  Tier 1: CognitiveGlyph routes input (question/statement/emotion/etc.)
  Tier 2: HDC thought space recalls relevant memories
  Tier 3: LLM synthesizes natural language response

Three test modes:
  - HDC only: raw recall, no LLM (baseline)
  - LLM + HDC: cognitive routing + recall + LLM synthesis
  - Cognitive routing: does the right agent fire?

Run:
    cd glyphh-runtime
    PYTHONPATH=. /opt/homebrew/anaconda3/bin/python tests/experiment_training.py
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
from glyphh.memory.glyph_dream import GlyphDreamLoop, InsightKind
from glyphh.memory.cognitive_glyph import CognitiveGlyph, Action


# ── Training data: natural conversation ─────────────────────────────────

TRAINING_DATA = [
    # ── Identity ──
    ("my name is chris", "incoming"),
    ("your name is ada", "incoming"),
    ("you are ada", "incoming"),
    ("i am chris", "incoming"),
    ("my name is ada", "outgoing"),
    ("i am ada", "outgoing"),

    # ── Family & relationships ──
    ("my wife is named sarah", "incoming"),
    ("i have two kids", "incoming"),
    ("my son is named jake", "incoming"),
    ("my daughter is named emma", "incoming"),
    ("jake is seven years old", "incoming"),
    ("emma is four years old", "incoming"),
    ("we have a dog named max", "incoming"),
    ("max is a golden retriever", "incoming"),
    ("sarah works at the hospital", "incoming"),

    # ── Preferences ──
    ("i like pizza", "incoming"),
    ("i love coffee", "incoming"),
    ("my favorite color is blue", "incoming"),
    ("i hate running", "incoming"),
    ("i enjoy reading before bed", "incoming"),
    ("sarah likes sushi", "incoming"),
    ("jake loves minecraft", "incoming"),
    ("emma likes drawing", "incoming"),

    # ── Work & daily life ──
    ("i work as a software engineer", "incoming"),
    ("i work at a company called glyphh", "incoming"),
    ("i usually wake up at six", "incoming"),
    ("i drive a honda civic", "incoming"),
    ("we live in austin texas", "incoming"),
    ("our house has a blue door", "incoming"),

    # ── Facts & knowledge ──
    ("the earth goes around the sun", "incoming"),
    ("water freezes at zero degrees", "incoming"),
    ("python is a programming language", "incoming"),
    ("dogs are mammals", "incoming"),
    ("the sky is blue", "incoming"),

    # ── Emotions & states ──
    ("i am tired today", "incoming"),
    ("i feel happy when i code", "incoming"),
    ("jake gets scared at night", "incoming"),
    ("emma is always excited about school", "incoming"),
    ("i was angry yesterday at work", "incoming"),

    # ── Temporal / stories ──
    ("yesterday i went to the store", "incoming"),
    ("last week we visited grandma", "incoming"),
    ("tomorrow i have a meeting", "incoming"),
    ("next summer we are going to the beach", "incoming"),
    ("i learned python ten years ago", "incoming"),

    # ── Ada's own statements (things she said) ──
    ("i remember your name is chris", "outgoing"),
    ("you told me you like pizza", "outgoing"),
    ("your wife is sarah", "outgoing"),
    ("you have two children jake and emma", "outgoing"),
    ("i do not know what you had for lunch", "outgoing"),
]


# ── Test queries with expected top-1 content ─────────────────────────────

TEST_QUERIES = [
    # (query, expected substring in answer, description)
    ("who am i?", "chris", "identity - user"),
    ("who are you?", "ada", "identity - ada"),
    ("what is my name?", "chris", "name - user"),
    ("what is your name?", "ada", "name - ada"),
    ("what do i like?", "pizza", "preference - user"),
    ("what is my favorite color?", "blue", "preference - color"),
    ("what do i do for work?", "software", "occupation"),
    ("where do i live?", "austin", "location"),
    ("what is my wife's name?", "sarah", "family - wife"),
    ("how old is jake?", "seven", "family - son age"),
    ("how old is emma?", "four", "family - daughter age"),
    ("what kind of dog do i have?", "golden", "pet"),
    ("what is my dog's name?", "max", "pet name"),
    ("what does sarah do?", "hospital", "family - wife work"),
    ("what does jake like?", "minecraft", "family - son preference"),
    ("what does emma like?", "drawing", "family - daughter preference"),
    ("what car do i drive?", "honda", "possession"),
    ("when do i wake up?", "six", "routine"),
    ("where did i go yesterday?", "store", "temporal - past"),
    ("what happens tomorrow?", "meeting", "temporal - future"),
    ("how am i feeling?", "tired", "emotion - current"),
    ("what makes me happy?", "code", "emotion - trigger"),
]

# ── Cognitive routing tests ──────────────────────────────────────────────

ROUTING_TESTS = [
    # (text, expected_action)
    ("my name is chris", Action.STORE),
    ("i like pizza", Action.STORE),
    ("jake loves minecraft", Action.STORE),
    ("what is my name?", Action.RECALL),
    ("who am i?", Action.RECALL),
    ("where do i live?", Action.RECALL),
    ("i am so happy right now", Action.FEEL),
    ("i love you", Action.FEEL),
    ("that makes me worried", Action.FEEL),
    ("no that is wrong", Action.CONTRADICT),
    ("you are mistaken", Action.CONTRADICT),
    ("i never said that", Action.CONTRADICT),
    ("tell me more about that", Action.WONDER),
    ("that is interesting", Action.WONDER),
    ("i wonder why", Action.WONDER),
]


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def run_hdc_test(space: ThoughtGlyphSpace, label: str) -> tuple[int, int]:
    """Run test queries with pure HDC recall (no LLM)."""
    passed = 0
    failed = []

    for query, expected, desc in TEST_QUERIES:
        results = space.recall(query, top_k=3, speaker="incoming")
        if results:
            top = results[0].thought.content.lower()
            if expected.lower() in top:
                passed += 1
            else:
                failed.append((query, expected, top, results[0].global_similarity, desc))
        else:
            failed.append((query, expected, "(no results)", 0.0, desc))

    total = len(TEST_QUERIES)
    print(f"  {label}: {passed}/{total} ({passed / total * 100:.0f}%)")

    if failed:
        print()
        for query, expected, got, sim, desc in failed:
            print(f"    MISS [{desc}]: \"{query}\"")
            print(f"          expected \"{expected}\", got [{sim:.3f}] \"{got}\"")

    return passed, total


def run_llm_test(space, cog, engine, label: str) -> tuple[int, int]:
    """Run test queries with LLM synthesis + HDC recall."""
    # Lazy import to avoid circular
    from glyphh.cli.commands.ada import _build_llm_prompt, _recall_context

    passed = 0
    failed = []

    for query, expected, desc in TEST_QUERIES:
        state = cog.process(query)
        recall = _recall_context(query)
        prompt = _build_llm_prompt(query, state, recall)
        resp = engine.generate(prompt, max_tokens=128, temperature=0.3, raw=True)
        resp = re.sub(r"</?think>\s*", "", resp).strip()

        if expected.lower() in resp.lower():
            passed += 1
        else:
            failed.append((query, expected, resp[:80], state.winner, desc))

    total = len(TEST_QUERIES)
    print(f"  {label}: {passed}/{total} ({passed / total * 100:.0f}%)")

    if failed:
        print()
        for query, expected, got, winner, desc in failed:
            print(f"    MISS [{desc}]: \"{query}\" (route: {winner})")
            print(f"          expected \"{expected}\", got \"{got}\"")

    return passed, total


def run_routing_test(cog: CognitiveGlyph, label: str) -> tuple[int, int]:
    """Test cognitive routing accuracy."""
    passed = 0
    failed = []

    for text, expected_action in ROUTING_TESTS:
        state = cog.process(text)
        if state.action == expected_action:
            passed += 1
        else:
            failed.append((text, expected_action.value, state.winner, state.confidence))

    total = len(ROUTING_TESTS)
    print(f"  {label}: {passed}/{total} ({passed / total * 100:.0f}%)")

    if failed:
        print()
        for text, expected, winner, conf in failed:
            print(f"    MISS: \"{text}\" → {winner} ({conf:.2f}), expected {expected}")

    return passed, total


def main() -> None:
    print_section("EXPERIMENT: Ada Full Pipeline")

    # ── Initialize ──
    space = ThoughtGlyphSpace()
    loop = GlyphCognitiveLoop(thought_space=space)
    cog = CognitiveGlyph(thought_space=space)

    stats = space.primitives.stats()
    print(f"  Primitives: {stats['words']} words, {stats['roles']} roles")
    print(f"  Cognitive agents: {len(cog.agents)} ({', '.join(cog.agents.keys())})")

    # Check LLM
    engine = None
    try:
        from glyphh.llm import LLMEngine
        engine = LLMEngine(system_prompt="You are Ada.")
        engine._ensure_loaded()
        print(f"  LLM: {engine.backend_name}")
    except Exception as e:
        print(f"  LLM: not available ({e})")

    # ── Phase 1: Absorb training data ──
    print_section("PHASE 1: Absorbing training data")
    for text, speaker in TRAINING_DATA:
        space.absorb(text, speaker=speaker)
    print(f"  Absorbed {space.count} thoughts")

    # ── Phase 2: Cognitive Routing ──
    print_section("PHASE 2: Cognitive Routing")
    routing_passed, routing_total = run_routing_test(cog, "Routing accuracy")

    # ── Phase 3: HDC Recall (baseline) ──
    print_section("PHASE 3: HDC Recall (no LLM)")
    hdc_passed, hdc_total = run_hdc_test(space, "HDC recall")

    # ── Phase 4: LLM + HDC ──
    if engine is not None:
        # Need to wire up the ada module singletons
        import glyphh.cli.commands.ada as ada_mod
        ada_mod._thought_space = space
        ada_mod._cognitive = cog

        print_section("PHASE 4: LLM + HDC (full pipeline)")
        llm_passed, llm_total = run_llm_test(space, cog, engine, "LLM + HDC")
    else:
        print_section("PHASE 4: SKIPPED (no LLM)")
        llm_passed, llm_total = 0, len(TEST_QUERIES)

    # ── Phase 5: Dream ──
    print_section("PHASE 5: Dreaming (40s)")
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
    for i in range(8):
        time.sleep(5)
        insights = dream.drain_insights()
        all_insights.extend(insights)
        loc = dream.stats.get("localized_cycles", 0)
        deep = dream.stats.get("deep_cycles", 0)
        cands = dream.stats.get("crystallization_candidates", 0)
        print(f"  [{i*5:2d}s] localized={loc}, deep={deep}, "
              f"candidates={cands}, insights={len(insights)}")

    dream.stop()

    by_kind = {}
    for ins in all_insights:
        by_kind[ins.kind.value] = by_kind.get(ins.kind.value, 0) + 1
    print(f"\n  Total insights: {len(all_insights)}")
    for kind, count in sorted(by_kind.items()):
        print(f"    {kind}: {count}")

    # ── Phase 6: Post-dream recall ──
    print_section("PHASE 6: HDC Recall AFTER dreaming")
    hdc_after, _ = run_hdc_test(space, "HDC recall (post-dream)")

    if engine is not None:
        print_section("PHASE 7: LLM + HDC AFTER dreaming")
        llm_after, _ = run_llm_test(space, cog, engine, "LLM + HDC (post-dream)")
    else:
        llm_after = 0

    # ── Summary ──
    print_section("SUMMARY")

    print(f"  Cognitive routing:   {routing_passed}/{routing_total} ({routing_passed/routing_total*100:.0f}%)")
    print(f"  HDC recall:          {hdc_passed}/{hdc_total} ({hdc_passed/hdc_total*100:.0f}%)")
    if engine:
        print(f"  LLM + HDC:           {llm_passed}/{llm_total} ({llm_passed/llm_total*100:.0f}%)")
    print(f"  HDC after dream:     {hdc_after}/{hdc_total} ({hdc_after/hdc_total*100:.0f}%)")
    if engine:
        print(f"  LLM + HDC post-dream:{llm_after}/{llm_total} ({llm_after/llm_total*100:.0f}%)")
    print(f"  Thoughts:            {space.count}")
    print(f"  Insights:            {len(all_insights)}")
    print()


if __name__ == "__main__":
    main()
