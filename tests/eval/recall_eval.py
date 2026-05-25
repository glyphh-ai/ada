"""
Recall eval harness — the scoreboard for Ada's memory quality.

Run:
    python tests/eval/recall_eval.py
    python tests/eval/recall_eval.py --top-k 5 --gate 0.5

Measures, per category and overall:
  - recall@1 / recall@3   : did the correct fact come back?
  - gate_pass             : correct fact returned AND confidence >= gate
  - false_negative        : correct fact returned BUT confidence < gate
                            (Ada HOLDS the answer but would say "I don't know")
  - correct_refusal       : unanswerable query stayed below gate (good)
  - dangerous             : a WRONG fact came back above gate (confidently wrong)

The point: paraphrase + multi_hop are where a lexical-overlap recall breaks.
Baseline these now, then re-run after each fix (Qwen3 semantic signal, wired
reason(), chunking). Every fix must move these numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from glyphh.memory.thought_space import ThoughtGlyphSpace


# ── Knowledge base: a small graph Ada is taught once ────────────────────────
KB = [
    # Family graph (for multi-hop)
    "The user's name is Chris.",
    "Chris is married to Brandi.",
    "Chris and Brandi have two sons, James and Traceton.",
    "Elaine is Brandi's mother.",
    "Jim is Elaine's husband.",
    # Company / work
    "Chris works at Glyphh AI.",
    "Glyphh AI builds a hyperdimensional computing runtime.",
    "Brandi is a nurse at Mercy Hospital.",
    # World facts (distractors / exact)
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    # Ada self-facts
    "Ada persists memories in a SQLite database.",
    "Ada has a dream loop with REM and Slow-Wave phases.",
    "Ada's firewall checks every input for prompt injection.",
]


@dataclass
class Case:
    query: str
    category: str                       # exact | paraphrase | multi_hop | unanswerable
    answer: str | None = None           # substring that must appear in the recalled fact
    support: list[str] = field(default_factory=list)  # multi_hop: all must appear in top_k
    unanswerable: bool = False


CASES = [
    # ── exact: query shares words with the stored fact ──
    Case("what is the capital of France?", "exact", answer="Paris"),
    Case("where does Chris work?", "exact", answer="Glyphh AI"),
    Case("at what temperature does water boil?", "exact", answer="100 degrees"),
    Case("how does Ada persist memories?", "exact", answer="SQLite"),
    Case("what is the user's name?", "exact", answer="Chris"),

    # ── paraphrase: synonyms / reordering, low lexical overlap ──
    Case("which company employs Chris?", "paraphrase", answer="Glyphh AI"),
    Case("who am I?", "paraphrase", answer="Chris"),
    Case("what city is the French capital?", "paraphrase", answer="Paris"),
    Case("where are Ada's thoughts stored?", "paraphrase", answer="SQLite"),
    Case("what does Brandi do for a living?", "paraphrase", answer="nurse"),
    Case("who is Chris's wife?", "paraphrase", answer="Brandi"),
    Case("what does Glyphh AI make?", "paraphrase", answer="hyperdimensional"),

    # ── multi_hop: answer requires combining facts; we test that retrieval
    #    surfaces ALL supporting facts (the prerequisite for an LLM to reason) ──
    Case("who are Jim's grandchildren?", "multi_hop",
         support=["Jim is Elaine's husband", "Elaine is Brandi's mother",
                  "James and Traceton"]),
    Case("what is the name of Brandi's father?", "multi_hop",
         support=["Elaine is Brandi's mother", "Jim is Elaine's husband"]),
    Case("who is Traceton's mother?", "multi_hop",
         support=["James and Traceton", "married to Brandi"]),
    Case("where does Chris's wife work?", "multi_hop",
         support=["married to Brandi", "Brandi is a nurse"]),

    # ── unanswerable: nothing in KB; must stay below gate ──
    Case("what is the weather in Tokyo today?", "unanswerable", unanswerable=True),
    Case("what is Chris's favorite color?", "unanswerable", unanswerable=True),
    Case("how tall is the Eiffel Tower?", "unanswerable", unanswerable=True),
]


def build_space(semantic: bool, weight: float, answerability: bool,
                expand: bool = False) -> ThoughtGlyphSpace:
    space = ThoughtGlyphSpace()
    space.use_semantic = semantic
    space.semantic_weight = weight
    space.use_answerability = answerability
    space.use_expansion = expand
    for fact in KB:
        space.absorb(fact, speaker="incoming")
    return space


def recall(space: ThoughtGlyphSpace, query: str, top_k: int):
    """Return [(content, sim), ...] sorted by sim desc."""
    results = space.recall(query, top_k=top_k)
    out = []
    for r in results:
        content = getattr(r.thought, "content", str(r.thought))
        out.append((content, float(r.global_similarity)))
    return out


def contains(content: str, needle: str) -> bool:
    return needle.lower() in content.lower()


def evaluate(top_k: int, gate: float, semantic: bool, weight: float,
             answerability: bool, expand: bool = False):
    space = build_space(semantic, weight, answerability, expand)
    mode = f"HDC+Qwen3 (α={weight})" if semantic else "HDC-only (baseline)"
    if answerability:
        mode += " +answerability"
    if expand:
        mode += " +expansion"
    print(f"\n  MODE: {mode}")

    cats: dict[str, dict] = {}
    rows = []

    for c in CASES:
        hits = recall(space, c.query, top_k)
        top1 = hits[0] if hits else ("<none>", 0.0)

        rec1 = rec3 = gate_pass = false_neg = correct_refusal = dangerous = support_cov = 0

        if c.unanswerable:
            # Good: top1 below gate (Ada correctly refuses)
            correct_refusal = 1 if top1[1] < gate else 0
            dangerous = 1 - correct_refusal
            verdict = "refuse✓" if correct_refusal else f"LEAK({top1[1]:.2f})"

        elif c.category == "multi_hop":
            present = [s for s in c.support if any(contains(h[0], s) for h in hits)]
            support_cov = len(present) / len(c.support)
            # passable only if every support fact is also above the gate
            above = all(any(contains(h[0], s) and h[1] >= gate for h in hits)
                        for s in c.support)
            gate_pass = 1 if (support_cov == 1.0 and above) else 0
            verdict = f"support {len(present)}/{len(c.support)}" + (" gated" if support_cov == 1.0 and not above else "")

        else:  # exact / paraphrase, single-fact
            rec1 = 1 if contains(top1[0], c.answer) else 0
            rec3 = 1 if any(contains(h[0], c.answer) for h in hits) else 0
            # find the correct fact's sim if present in top_k
            correct_sim = next((h[1] for h in hits if contains(h[0], c.answer)), None)
            if rec1 and top1[1] >= gate:
                gate_pass = 1
                verdict = f"ok {top1[1]:.2f}"
            elif correct_sim is not None and correct_sim < gate:
                false_neg = 1
                verdict = f"FALSE-NEG {correct_sim:.2f}"
            elif not rec3:
                # wrong fact; is it confidently wrong?
                dangerous = 1 if top1[1] >= gate else 0
                verdict = f"MISS{'(danger)' if dangerous else ''} {top1[1]:.2f}"
            else:
                verdict = f"rank>1 {top1[1]:.2f}"

        d = cats.setdefault(c.category, dict(n=0, rec1=0, rec3=0, gate_pass=0,
                                             false_neg=0, refuse=0, danger=0, cov=0.0))
        d["n"] += 1
        d["rec1"] += rec1; d["rec3"] += rec3; d["gate_pass"] += gate_pass
        d["false_neg"] += false_neg; d["refuse"] += correct_refusal
        d["danger"] += dangerous; d["cov"] += support_cov

        rows.append((c.category, c.query, top1[0][:38], f"{top1[1]:.2f}", verdict))

    # ── print ──
    print(f"\n  RECALL EVAL  (top_k={top_k}, gate={gate})  — {len(CASES)} cases, {len(KB)} facts\n")
    print(f"  {'CATEGORY':<13} {'QUERY':<36} {'TOP-1 RECALL':<40} {'SIM':<5} VERDICT")
    print("  " + "-"*118)
    for cat, q, top, sim, verdict in rows:
        print(f"  {cat:<13} {q[:35]:<36} {top:<40} {sim:<5} {verdict}")

    print("\n  ── Scoreboard by category ──")
    print(f"  {'category':<13} {'n':>3} {'gate_pass':>10} {'false_neg':>10} {'refuse':>7} {'danger':>7} {'mh_cov':>7}")
    tot = dict(n=0, gate_pass=0, false_neg=0, refuse=0, danger=0)
    for cat, d in cats.items():
        mh = f"{d['cov']/d['n']:.0%}" if cat == "multi_hop" else "-"
        print(f"  {cat:<13} {d['n']:>3} {d['gate_pass']:>10} {d['false_neg']:>10} "
              f"{d['refuse']:>7} {d['danger']:>7} {mh:>7}")
        for k in tot: tot[k] += d.get(k, 0)
    print("  " + "-"*60)
    answerable = tot["n"] - cats.get("unanswerable", {}).get("n", 0)
    print(f"  {'TOTAL':<13} {tot['n']:>3} {tot['gate_pass']:>10} {tot['false_neg']:>10} "
          f"{tot['refuse']:>7} {tot['danger']:>7}")
    print(f"\n  Answerable correctly served (gate_pass): {tot['gate_pass']}/{answerable}")
    print(f"  Held-but-refused (false_neg, the silent failure): {tot['false_neg']}/{answerable}")
    print(f"  Confidently wrong (danger): {tot['danger']}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--semantic", action="store_true", help="enable Qwen3 semantic signal")
    ap.add_argument("--weight", type=float, default=0.6, help="semantic blend weight (0..1)")
    ap.add_argument("--answerability", action="store_true", help="enable slot/type answerability gate")
    ap.add_argument("--expand", action="store_true", help="enable entity-expansion (multi-hop coverage)")
    args = ap.parse_args()
    evaluate(args.top_k, args.gate, args.semantic, args.weight,
             args.answerability, args.expand)
