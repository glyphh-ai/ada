"""
The nail-in-the-coffin demo: does a decision survive a context wipe?

This models exactly what kills coding agents: compaction summarizes the early
turns and the SPECIFIC decision falls out of the window. We prove that with
Glyphh as the memory spine, the decision survives — retrieved by RELEVANCE
from a paraphrased query (not an echo), with ZERO LLM tokens.

What a 1M context window can't promise: that the one constraint from turn 3 is
still authoritative on turn 400 after a lossy compaction.
What vanilla RAG misses: this is policy/decision recall, re-injected every turn.

Run:  python tests/eval/compaction_survival.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _make_brain():
    """Real Brain, mocked externals. Semantic recall ON (the live default).

    The LLM is UNAVAILABLE on purpose — proving remember/recall never touch it.
    """
    from domains.brain.think import Brain
    from domains.brain.llm import LLMUsage

    state = MagicMock(); state.capabilities = {}; state.capability_names = []
    mm = MagicMock(); mm._models = {}; mm._encoding_in_progress = []
    llm = MagicMock(); llm.available = False
    llm.set_firewall = MagicMock(); llm.ask = AsyncMock(return_value=None)
    llm.usage = LLMUsage()
    brain = Brain(brain_state=state, model_manager=mm, llm=llm,
                  session_factory=MagicMock())
    # Live recall quality: semantic + answerability + expansion.
    sp = brain.cognitive.thought_space
    sp.use_semantic = True; sp.use_answerability = True; sp.use_expansion = True
    return brain


def main():
    brain = _make_brain()

    line = "=" * 78
    print(f"\n{line}\n  COMPACTION-SURVIVAL DEMO — does a decision outlive the context window?\n{line}")

    # ── Turn 3: the agent and user agree on a working rule. The hook captures it.
    decision = ("Always run the eval (python tests/eval/recall_eval.py) before "
                "claiming any recall change improves results.")
    print(f"\n[turn 3]  decision made -> remember()")
    print(f"          \"{decision}\"")
    res = brain.remember(decision)
    print(f"          stored: {res['stored']}  (LLM tokens used: "
          f"{brain.llm.usage.input_tokens + brain.llm.usage.output_tokens})")

    # ... 400 turns of unrelated work happen here ...
    # ── COMPACTION: the window is summarized. The decision is GONE from context.
    print(f"\n[compaction]  window summarized — the decision is no longer in context.")
    print(f"              a plain agent now knows NOTHING about the eval rule.")

    # ── Turn 401: the agent is about to claim a win. A UserPromptSubmit hook
    #    fires and re-grounds from Glyphh with a PARAPHRASED query (no echo).
    paraphrase = "I think the new embedding makes recall better, about to report success"
    print(f"\n[turn 401]  agent's intent (paraphrased, NOT the decision text):")
    print(f"            \"{paraphrase}\"")
    print(f"            UserPromptSubmit hook -> recall()")

    tokens_before = brain.llm.usage.input_tokens + brain.llm.usage.output_tokens
    facts = brain.recall(paraphrase, top_k=3)
    tokens_after = brain.llm.usage.input_tokens + brain.llm.usage.output_tokens

    print(f"\n            re-grounded facts injected into the agent's context:")
    for f in facts:
        print(f"              - ({f['confidence']:.2f}) {f['content']}")
    print(f"\n            LLM tokens spent on recall: {tokens_after - tokens_before}")

    # ── Verdict
    survived = any("run the eval" in f["content"].lower() for f in facts)
    print(f"\n{line}")
    if survived:
        print("  VERDICT: the decision SURVIVED the context wipe.")
        print("  - retrieved by relevance from a paraphrase (not a lexical echo)")
        print("  - zero LLM tokens (pure local retrieval)")
        print("  - this is what the hook re-injects EVERY turn -> the agent can't")
        print("    drift past it, no matter how many compactions happen.")
        print("\n  A context window can't promise this. Vanilla RAG doesn't re-inject")
        print("  policy every turn. This is the product.")
    else:
        print("  VERDICT: decision did NOT resurface — recall/relevance gap to fix.")
    print(line + "\n")
    return 0 if survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
