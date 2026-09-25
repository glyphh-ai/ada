"""
Real-world strand evaluation #3: how far ahead can the substrate see?

Two horizon curves, measured on Schema-Guided Dialogue:

A. FINE level — multi-step rollout. From a true prefix, predict the next
   turn's label, synthesize its codon, extend the strand with its own
   prediction, and repeat. Accuracy at horizon k shows how fast chained
   step-prediction decays toward the majority baseline: the substrate's
   detail-horizon.

B. COARSE level — destination prediction. At every prefix length,
   predict the dialogue's terminal outcome (transaction success =
   NOTIFY_SUCCESS ever occurs) as the success rate among the k nearest
   training prefixes, and score AUC per prefix length, plus calibration
   (do 70% predictions come true 70% of the time?). Destination
   prediction does not chain, so it can stay accurate at ranges where
   rollout is noise.

Labels here are fully codon-reconstructible — (speaker, sorted
(act, slot) pairs) — so a predicted label can be re-encoded and fed
back into the strand during rollout.

Data: any subset of the train/dialogues_*.json files from
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

Usage:
    python tests/eval/strand_horizon_eval.py --data-dir /path/to/sgd
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from glyphh.core.ops import bundle, generate_symbol
from glyphh.strand import Strand

SEED = 42


class SymbolTable:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._cache = {}

    def __call__(self, key: str):
        if key not in self._cache:
            self._cache[key] = generate_symbol(SEED, key, self.dimension)
        return self._cache[key]


def turn_label(turn: dict) -> tuple:
    """(speaker, sorted (act, slot) pairs) — reconstructible into a codon."""
    pairs = set()
    for frame in turn.get("frames", []):
        for action in frame.get("actions", []):
            pairs.add((action["act"], action.get("slot") or "none"))
    return (turn["speaker"], tuple(sorted(pairs)))


def label_codon(label: tuple, sym: SymbolTable):
    """Canonical codon for a label; identical for real and predicted turns."""
    speaker, pairs = label
    parts = [sym("speaker") * sym(f"spk:{speaker}")]
    for act, slot in pairs:
        parts.append((sym(f"act:{act}") * sym(f"slot:{slot}")).astype(np.int8))
    return bundle(parts)


def dialogue_success(dlg: dict) -> bool:
    return any(
        action["act"] == "NOTIFY_SUCCESS"
        for turn in dlg["turns"]
        for frame in turn.get("frames", [])
        for action in frame.get("actions", [])
    )


class MatrixPredictor:
    """Nearest-class-mean over strand states, vectorized for rollout."""

    def __init__(self):
        self._acc = {}

    def observe(self, state, label):
        acc = self._acc.setdefault(label, np.zeros(state.shape[0]))
        acc += state

    def freeze(self):
        self.labels = list(self._acc)
        protos = np.stack([self._acc[l] for l in self.labels]).astype(np.float32)
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        self._protos = protos / np.maximum(norms, 1e-9)

    def predict(self, state) -> tuple:
        scores = self._protos @ state.astype(np.float32)
        return self.labels[int(np.argmax(scores))]


def encode(dlg, sym):
    labels = [turn_label(t) for t in dlg["turns"]]
    return [label_codon(l, sym) for l in labels], labels


def rollout_curve(train, test, sym, decay, max_h, anchor_stride):
    predictor = MatrixPredictor()
    speaker_majority = {}
    counts = {"USER": Counter(), "SYSTEM": Counter()}
    for dlg in train:
        codons, labels = encode(dlg, sym)
        strand = Strand()
        for t in range(len(labels) - 1):
            strand.append(codons[t])
            predictor.observe(strand.state(decay), labels[t + 1])
            counts[labels[t + 1][0]][labels[t + 1]] += 1
    predictor.freeze()
    for spk, c in counts.items():
        speaker_majority[spk] = c.most_common(1)[0][0]

    hits = np.zeros(max_h)
    base_hits = np.zeros(max_h)
    totals = np.zeros(max_h)
    for dlg in test:
        codons, labels = encode(dlg, sym)
        for t in range(1, len(labels) - 1, anchor_stride):
            strand = Strand(codons[:t])
            rolled = Strand(codons[:t])
            for k in range(1, max_h + 1):
                target_idx = t + k - 1
                if target_idx >= len(labels):
                    break
                pred = predictor.predict(rolled.state(decay))
                rolled.append(label_codon(pred, sym))
                truth = labels[target_idx]
                hits[k - 1] += pred == truth
                base_hits[k - 1] += speaker_majority[truth[0]] == truth
                totals[k - 1] += 1
    return hits / totals, base_hits / totals, totals


def outcome_curve(train, test, sym, decay, max_prefix, knn):
    train_states, train_outcomes = [], []
    for dlg in train:
        codons, _ = encode(dlg, sym)
        outcome = dialogue_success(dlg)
        strand = Strand()
        for t in range(min(len(codons), max_prefix)):
            strand.append(codons[t])
            train_states.append(strand.state(decay))
            train_outcomes.append(outcome)
    train_m = np.stack(train_states).astype(np.float32)
    train_y = np.array(train_outcomes)

    rows = []          # (prefix_len, p_success, actual)
    for dlg in test:
        codons, _ = encode(dlg, sym)
        outcome = dialogue_success(dlg)
        strand = Strand()
        for t in range(min(len(codons), max_prefix)):
            strand.append(codons[t])
            sims = train_m @ strand.state(decay).astype(np.float32)
            top = np.argpartition(sims, -knn)[-knn:]
            rows.append((t + 1, float(train_y[top].mean()), outcome))
    return rows


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.asarray(pos), np.asarray(neg)
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--anchor-stride", type=int, default=3)
    parser.add_argument("--max-prefix", type=int, default=12)
    parser.add_argument("--knn", type=int, default=50)
    args = parser.parse_args()

    dialogues = []
    for path in sorted(args.data_dir.glob("dialogues_*.json")):
        dialogues.extend(json.loads(path.read_text()))
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(dialogues))
    test_ids = set(order[: int(len(dialogues) * 0.2)].tolist())
    train = [d for i, d in enumerate(dialogues) if i not in test_ids]
    test = [d for i, d in enumerate(dialogues) if i in test_ids]
    base_rate = np.mean([dialogue_success(d) for d in test])
    print(f"train {len(train)} / test {len(test)}, "
          f"success base rate {base_rate:.3f}, dim={args.dimension}\n")

    sym = SymbolTable(args.dimension)

    print("A. FINE level — rollout accuracy by horizon (strand feeds on its own predictions)")
    acc, base, totals = rollout_curve(
        train, test, sym, args.decay, args.max_horizon, args.anchor_stride)
    print(f"{'horizon':>8} {'strand':>8} {'majority':>9} {'edge':>7} {'n':>6}")
    for k in range(args.max_horizon):
        print(f"{k + 1:>8} {acc[k]:>8.3f} {base[k]:>9.3f} "
              f"{acc[k] - base[k]:>+7.3f} {int(totals[k]):>6}")

    print("\nB. COARSE level — outcome AUC by prefix length (destination, not path)")
    rows = outcome_curve(train, test, sym, args.decay, args.max_prefix, args.knn)
    print(f"{'prefix':>8} {'AUC':>7} {'n':>6}")
    for t in range(1, args.max_prefix + 1):
        preds = [(p, o) for pl, p, o in rows if pl == t]
        pos = [p for p, o in preds if o]
        neg = [p for p, o in preds if not o]
        print(f"{t:>8} {auc(pos, neg):>7.3f} {len(preds):>6}")

    print("\nCalibration (prefixes 4-8): predicted P(success) vs observed")
    mid = [(p, o) for pl, p, o in rows if 4 <= pl <= 8]
    print(f"{'bucket':>12} {'predicted':>10} {'observed':>9} {'n':>6}")
    for lo in np.arange(0.0, 1.0, 0.2):
        bucket = [(p, o) for p, o in mid if lo <= p < lo + 0.2 or (lo == 0.8 and p == 1.0)]
        if bucket:
            print(f"{lo:>5.1f}-{lo + 0.2:<5.1f} "
                  f"{np.mean([p for p, _ in bucket]):>10.3f} "
                  f"{np.mean([o for _, o in bucket]):>9.3f} {len(bucket):>6}")


if __name__ == "__main__":
    main()
