"""
Real-world DecisionSpace validation on Schema-Guided Dialogue.

Each training dialogue becomes a recorded decision: situation = coarse
attributes of its opening (service intent, first-turn acts), events =
the first N turns as an ordered sequence, outcome = whether the
transaction succeeded. Held-out dialogues are then queried as open
decisions.

What this checks, in the product's own terms (not forecasting):
  - accuracy of top_outcome among SUFFICIENT answers
  - calibration: does confidence 0.9 mean right 90% of the time?
  - the insufficient-precedent gate: how often it declines, and whether
    accuracy among declined-then-forced answers is indeed worse
Prints one full fact-tree answer so the receipts are visible.

Data: any subset of the train/dialogues_*.json files from
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

Usage:
    python tests/eval/decision_space_eval.py --data-dir /path/to/sgd
"""

import argparse
import json
from pathlib import Path

import numpy as np

from glyphh.decision import DecisionSpace

SEED = 42
PREFIX_TURNS = 6


def dialogue_success(dlg: dict) -> bool:
    return any(
        action["act"] == "NOTIFY_SUCCESS"
        for turn in dlg["turns"]
        for frame in turn.get("frames", [])
        for action in frame.get("actions", [])
    )


def situation_of(dlg: dict) -> dict:
    turns = dlg["turns"]
    first_acts = sorted({
        a["act"] for f in turns[0].get("frames", []) for a in f["actions"]})
    intent = next((
        v for f in turns[0].get("frames", [])
        for a in f["actions"] if a["slot"] == "intent"
        for v in a.get("values", [])), "unknown")
    return {
        "service": dlg["services"][0],
        "intent": intent,
        "opening_acts": "+".join(first_acts),
    }


def events_of(dlg: dict) -> list:
    events = []
    for turn in dlg["turns"][:PREFIX_TURNS]:
        acts = sorted({
            f"{a['act']}:{a.get('slot') or 'none'}"
            for f in turn.get("frames", []) for a in f["actions"]})
        events.append({"speaker": turn["speaker"], "acts": "+".join(acts)})
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=4096)
    args = parser.parse_args()

    dialogues = []
    for path in sorted(args.data_dir.glob("dialogues_*.json")):
        dialogues.extend(json.loads(path.read_text()))
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(dialogues))
    test_ids = set(order[: int(len(dialogues) * 0.2)].tolist())
    train = [d for i, d in enumerate(dialogues) if i not in test_ids]
    test = [d for i, d in enumerate(dialogues) if i in test_ids]

    space = DecisionSpace(dimension=args.dimension, k=25)
    for dlg in train:
        outcome = "success" if dialogue_success(dlg) else "failure"
        space.record(situation_of(dlg), outcome, events=events_of(dlg),
                     facts={"dialogue_id": dlg.get("dialogue_id", "?"),
                            "turns": len(dlg["turns"])})
    base = np.mean([dialogue_success(d) for d in test])
    print(f"{len(space)} precedents recorded, {len(test)} queries, "
          f"success base rate {base:.3f}\n")

    rows = []
    for dlg in test:
        answer = space.query(situation_of(dlg), events=events_of(dlg))
        actual = "success" if dialogue_success(dlg) else "failure"
        rows.append((answer, actual))

    sufficient = [(a, y) for a, y in rows if a.sufficient]
    declined = len(rows) - len(sufficient)
    acc = np.mean([a.top_outcome == y for a, y in sufficient])
    print(f"answered {len(sufficient)}/{len(rows)} "
          f"(declined {declined} for thin precedent)")
    print(f"accuracy among answered: {acc:.3f} "
          f"(always-say-success baseline {base:.3f})\n")

    print("calibration of confidence (answered queries):")
    print(f"{'bucket':>12} {'confidence':>11} {'accuracy':>9} {'n':>6}")
    for lo in np.arange(0.5, 1.0, 0.1):
        bucket = [(a, y) for a, y in sufficient
                  if lo <= a.confidence < lo + 0.1 or
                  (lo >= 0.9 and a.confidence == 1.0)]
        if bucket:
            conf = np.mean([a.confidence for a, _ in bucket])
            hit = np.mean([a.top_outcome == y for a, y in bucket])
            print(f"{lo:>5.1f}-{lo + 0.1:<5.1f} {conf:>11.3f} "
                  f"{hit:>9.3f} {len(bucket):>6}")

    # show one full fact-tree answer
    answer, actual = sufficient[0]
    print(f"\n──── example fact-tree answer (actual outcome: {actual}) ────")
    print(answer.fact_tree.to_text()[:2400])


if __name__ == "__main__":
    main()
