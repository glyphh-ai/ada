"""End-to-end experiment: CognitiveGlyph routing + ThoughtGlyph recall.

Simulates a full conversation with Ada. Tests that:
  1. Cognitive routing correctly classifies input types
  2. Statements get absorbed into thought space
  3. Questions trigger recall and find the right answers
  4. Emotional input routes to FEEL
  5. Contradictions route to CONTRADICT
  6. Idle time shifts weight toward DREAM
"""

import sys
import time

sys.path.insert(0, "glyphh-runtime")

from glyphh.memory.cognitive_glyph import CognitiveGlyph, Action
from glyphh.memory.thought_space import ThoughtGlyphSpace


def main():
    space = ThoughtGlyphSpace()
    cog = CognitiveGlyph(thought_space=space)

    print("=" * 70)
    print("  COGNITIVE ROUTING + RECALL — End-to-End Experiment")
    print("=" * 70)

    # ── Phase 1: Absorb statements ──────────────────────────────────────
    print("\n  PHASE 1: Teaching Ada (statements)")
    print("  " + "-" * 50)

    statements = [
        "my name is chris",
        "my wife is named sarah",
        "my son is named jake",
        "my daughter is named emma",
        "i work as a software engineer",
        "we live in austin texas",
        "jake loves minecraft",
        "emma likes drawing",
        "i like pizza",
        "sarah works at the hospital",
        "i am tired today",
        "i feel happy when i code",
    ]

    store_correct = 0
    for text in statements:
        state = cog.process(text)
        space.absorb(text, speaker="incoming")
        is_store = state.action == Action.STORE
        if is_store:
            store_correct += 1
        mark = "✓" if is_store else "✗"
        print(f"    {mark} \"{text}\" → {state.winner} ({state.confidence:.2f})")

    print(f"\n    Statement routing: {store_correct}/{len(statements)}")

    # ── Phase 2: Ask questions ──────────────────────────────────────────
    print("\n  PHASE 2: Questions → Cognitive Routing + Recall")
    print("  " + "-" * 50)

    questions = [
        ("what is my name?", "chris"),
        ("who am i?", "chris"),
        ("what is my wife's name?", "sarah"),
        ("what does jake like?", "minecraft"),
        ("where do i live?", "austin"),
        ("what do i do for work?", "software"),
        ("who works at the hospital?", "sarah"),
        ("what does emma like?", "drawing"),
        ("do i like pizza?", "pizza"),
    ]

    recall_correct = 0
    route_correct = 0
    for query, expected_keyword in questions:
        state = cog.process(query)
        is_recall = state.action == Action.RECALL
        if is_recall:
            route_correct += 1

        # Do actual recall
        results = space.recall(query, top_k=3, speaker="incoming")
        top_answer = results[0].thought.content if results else "(nothing)"
        found = expected_keyword.lower() in top_answer.lower()
        if found:
            recall_correct += 1

        route_mark = "?" if is_recall else "✗"
        recall_mark = "✓" if found else "✗"
        sim = f"{results[0].global_similarity:.3f}" if results else "0.000"
        print(f"    {route_mark}{recall_mark} \"{query}\"")
        print(f"        route: {state.winner} | recall: \"{top_answer}\" ({sim})")

    print(f"\n    Question routing: {route_correct}/{len(questions)}")
    print(f"    Recall accuracy: {recall_correct}/{len(questions)}")

    # ── Phase 3: Emotional input ────────────────────────────────────────
    print("\n  PHASE 3: Emotional Input")
    print("  " + "-" * 50)

    emotions = [
        ("i am so happy right now", Action.FEEL),
        ("that makes me really worried", Action.FEEL),
        ("i love you", Action.FEEL),
        ("i am scared", Action.FEEL),
    ]

    emotion_correct = 0
    for text, expected_action in emotions:
        state = cog.process(text)
        correct = state.action == expected_action
        if correct:
            emotion_correct += 1
        mark = "✓" if correct else "✗"
        print(f"    {mark} \"{text}\" → {state.winner} ({state.confidence:.2f})")

    print(f"\n    Emotion routing: {emotion_correct}/{len(emotions)}")

    # ── Phase 4: Contradictions ─────────────────────────────────────────
    print("\n  PHASE 4: Contradictions")
    print("  " + "-" * 50)

    contradictions = [
        ("no that is wrong", Action.CONTRADICT),
        ("you are mistaken", Action.CONTRADICT),
        ("i never said that", Action.CONTRADICT),
        ("actually it is different", Action.CONTRADICT),
    ]

    contra_correct = 0
    for text, expected_action in contradictions:
        state = cog.process(text)
        correct = state.action == expected_action
        if correct:
            contra_correct += 1
        mark = "✓" if correct else "✗"
        print(f"    {mark} \"{text}\" → {state.winner} ({state.confidence:.2f})")

    print(f"\n    Contradiction routing: {contra_correct}/{len(contradictions)}")

    # ── Phase 5: Curiosity ──────────────────────────────────────────────
    print("\n  PHASE 5: Curiosity")
    print("  " + "-" * 50)

    curiosity = [
        ("tell me more about that", Action.WONDER),
        ("that is interesting", Action.WONDER),
        ("i wonder why", Action.WONDER),
        ("how does that work", Action.WONDER),
    ]

    curiosity_correct = 0
    for text, expected_action in curiosity:
        state = cog.process(text)
        correct = state.action == expected_action
        if correct:
            curiosity_correct += 1
        mark = "✓" if correct else "✗"
        print(f"    {mark} \"{text}\" → {state.winner} ({state.confidence:.2f})")

    print(f"\n    Curiosity routing: {curiosity_correct}/{len(curiosity)}")

    # ── Phase 6: Idle → Dream transition ────────────────────────────────
    print("\n  PHASE 6: Idle → Dream Transition")
    print("  " + "-" * 50)

    idle_times = [1, 5, 10, 15, 20, 30]
    for seconds in idle_times:
        cog._idle_since = time.time() - seconds
        idle_state = cog.idle_tick()
        if idle_state:
            print(f"    {seconds:2d}s idle → {idle_state.winner} ({idle_state.confidence:.3f})")
        else:
            print(f"    {seconds:2d}s idle → (no activation)")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total_routing = store_correct + route_correct + emotion_correct + contra_correct + curiosity_correct
    total_tests = len(statements) + len(questions) + len(emotions) + len(contradictions) + len(curiosity)

    print(f"    Statement routing:     {store_correct}/{len(statements)}")
    print(f"    Question routing:      {route_correct}/{len(questions)}")
    print(f"    Emotion routing:       {emotion_correct}/{len(emotions)}")
    print(f"    Contradiction routing: {contra_correct}/{len(contradictions)}")
    print(f"    Curiosity routing:     {curiosity_correct}/{len(curiosity)}")
    print(f"    ─────────────────────────────────")
    print(f"    Total routing:         {total_routing}/{total_tests} ({total_routing/total_tests*100:.0f}%)")
    print(f"    Recall accuracy:       {recall_correct}/{len(questions)} ({recall_correct/len(questions)*100:.0f}%)")
    print(f"    Thoughts stored:       {space.count}")

    # Cognitive agent stats
    print(f"\n    Cognitive agents:")
    stats = cog.stats()
    for name, info in stats["agents"].items():
        print(f"      {name:12s} {info['exemplars']:2d} exemplars, activation={info['activation']:.3f}")


if __name__ == "__main__":
    main()
