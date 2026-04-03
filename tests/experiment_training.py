"""
Experiment: Training Ada at scale.

Feed Ada diverse natural conversation — identity, preferences, facts,
emotions, stories, questions — and measure how well the HDC recall
differentiates after DreamLoop crystallization.

This is the test bed for the hypothesis: primitives provide structural
axes, content words provide semantic signal, and the DreamLoop derives
compound concepts through crystallization. No LLM needed.

Run:
    cd glyphh-runtime
    PYTHONPATH=. /opt/homebrew/anaconda3/bin/python tests/experiment_training.py
"""

import logging
import time

from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop
from glyphh.memory.glyph_dream import GlyphDreamLoop, InsightKind

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)


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
    # (query, expected substring in top-1 result, description)
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


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def run_test(space: ThoughtGlyphSpace, label: str) -> tuple[int, int]:
    """Run test queries and return (passed, total)."""
    passed = 0
    failed_details = []

    for query, expected, desc in TEST_QUERIES:
        results = space.recall(query, top_k=3, speaker="incoming")
        if results:
            top = results[0].thought.content.lower()
            if expected.lower() in top:
                passed += 1
            else:
                failed_details.append((query, expected, top, results[0].global_similarity))
        else:
            failed_details.append((query, expected, "(no results)", 0.0))

    total = len(TEST_QUERIES)
    pct = passed / total * 100

    print(f"  {label}: {passed}/{total} ({pct:.0f}%)")

    if failed_details:
        print()
        for query, expected, got, sim in failed_details:
            print(f"    MISS: \"{query}\"")
            print(f"          expected \"{expected}\" in top-1, got [{sim:.3f}] \"{got}\"")

    return passed, total


def main() -> None:
    print_section("EXPERIMENT: Training Ada at Scale")

    # ── Initialize ──
    space = ThoughtGlyphSpace()
    loop = GlyphCognitiveLoop(thought_space=space)

    stats = space.primitives.stats()
    print(f"Primitives: {stats['words']} words, {stats['roles']} roles")

    # ── Phase 1: Absorb training data ──
    print_section("PHASE 1: Absorbing training data")

    for text, speaker in TRAINING_DATA:
        space.absorb(text, speaker=speaker)

    print(f"  Absorbed {space.count} thoughts")

    # Show content word extraction for a few
    print("\n  Sample encodings:")
    for text in ["my name is chris", "my favorite color is blue", "jake loves minecraft"]:
        t = space._thoughts
        for stored in t.values():
            if stored.content == text:
                attrs = stored.glyph.metadata.get("_activated_attrs", [])
                print(f"    \"{text}\"")
                print(f"      segments: {attrs}")
                break

    # ── Phase 2: Test before dreaming ──
    print_section("PHASE 2: Recall BEFORE dreaming")
    before_passed, before_total = run_test(space, "Before dreaming")

    # ── Phase 3: Dream ──
    print_section("PHASE 3: Dreaming")

    dream = GlyphDreamLoop(
        glyph_loop=loop,
        localized_interval=1.0,
        deep_interval=5.0,
        localized_size=100,
        deep_size=200,
    )
    dream.start()

    all_insights = []
    crystallizations = 0
    for i in range(8):
        time.sleep(5)
        insights = dream.drain_insights()
        all_insights.extend(insights)

        new_crystal = sum(1 for ins in insights if ins.kind == InsightKind.CRYSTALLIZATION)
        crystallizations += new_crystal

        loc = dream.stats.get("localized_cycles", 0)
        deep = dream.stats.get("deep_cycles", 0)
        cands = dream.stats.get("crystallization_candidates", 0)

        print(f"  [{i*5:2d}s] localized={loc}, deep={deep}, "
              f"candidates={cands}, insights={len(insights)}, "
              f"crystallized={new_crystal}")

    dream.stop()

    # Insight summary
    by_kind = {}
    for ins in all_insights:
        by_kind[ins.kind.value] = by_kind.get(ins.kind.value, 0) + 1

    print(f"\n  Total insights: {len(all_insights)}")
    for kind, count in sorted(by_kind.items()):
        print(f"    {kind}: {count}")

    # ── Phase 4: Test after dreaming ──
    print_section("PHASE 4: Recall AFTER dreaming")
    after_passed, after_total = run_test(space, "After dreaming")

    # ── Phase 5: Compound primitives ──
    print_section("PHASE 5: Compound primitives")

    stats_after = space.primitives.stats()
    new_words = stats_after["words"] - stats["words"]
    new_roles = stats_after["roles"] - stats["roles"]

    print(f"  Before: {stats['words']} words, {stats['roles']} roles")
    print(f"  After:  {stats_after['words']} words, {stats_after['roles']} roles")
    print(f"  New:    {new_words} words, {new_roles} roles")

    if crystallizations > 0:
        print(f"\n  {crystallizations} crystallization events")

    # ── Phase 6: Detailed recall inspection ──
    print_section("PHASE 6: Detailed recall (top 3)")

    for query, expected, desc in TEST_QUERIES[:10]:
        results = space.recall(query, top_k=3, speaker="incoming")
        print(f"  \"{query}\" ({desc}):")
        for r in results:
            who = "you said" if r.thought.speaker == "incoming" else "I said"
            layers = ", ".join(f"{k}={v:.2f}" for k, v in r.layer_similarities.items())
            marker = " <--" if expected.lower() in r.thought.content.lower() else ""
            print(f"    [{r.global_similarity:.3f}] {who}: \"{r.thought.content}\"{marker}")
        if not results:
            print("    (no results)")
        print()

    # ── Summary ──
    print_section("SUMMARY")

    delta = after_passed - before_passed
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    print(f"  Before dreaming:  {before_passed}/{before_total} ({before_passed/before_total*100:.0f}%)")
    print(f"  After dreaming:   {after_passed}/{after_total} ({after_passed/after_total*100:.0f}%)")
    print(f"  Delta:            {delta_str}")
    print(f"  Crystallizations: {crystallizations}")
    print(f"  New primitives:   {new_words} words, {new_roles} roles")
    print(f"  Total thoughts:   {space.count}")
    print()


if __name__ == "__main__":
    main()
