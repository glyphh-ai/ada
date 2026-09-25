"""
Real-world strand evaluation: next-dialogue-act prediction on the
Google Schema-Guided Dialogue (DSTC8) corpus.

The question under test: does the ORDER a strand preserves carry real
predictive signal that the lattice (unordered bundle) loses?

Task: given the conversation so far, predict the set of dialogue acts in
the next turn (e.g. {REQUEST}, {OFFER, INFORM_COUNT}). Real task-oriented
dialogues, real annotation, directly the shape of ada's tool-calling loop
(PERCEIVE -> DECIDE what to do next).

Encoders compared, identical codons throughout (one glyph per turn:
speaker + act/slot bindings), differing only in how history is combined:

  strand       permute-bind chain state, tail-anchored, exponential decay
  bundle       majority bundle of the same turn glyphs (the lattice way)
  last-turn    the previous turn's glyph alone (no history)

Prediction is 1-NN over training histories in each vector space: the
predicted next act-set is whatever followed the nearest training history.

Symbolic baselines: global majority act-set, and a bigram table
(most frequent act-set following the previous turn's act-set).

Data: any subset of the train/dialogues_*.json files from
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

Usage:
    python tests/eval/strand_dialog_eval.py --data-dir /path/to/sgd \
        [--dimension 4096] [--decay 0.7] [--max-train 20000]
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from glyphh.core.ops import bundle, generate_symbol
from glyphh.strand import Strand

SEED = 42


class SymbolTable:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._cache = {}

    def __call__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = generate_symbol(SEED, key, self.dimension)
        return self._cache[key]


def turn_glyph(turn: dict, sym: SymbolTable) -> np.ndarray:
    """One codon per turn: speaker plus every (act, slot) binding."""
    parts = [sym("speaker") * sym(f"spk:{turn['speaker']}")]
    for frame in turn.get("frames", []):
        for action in frame.get("actions", []):
            act = sym(f"act:{action['act']}")
            slot = sym(f"slot:{action.get('slot') or 'none'}")
            parts.append((act * slot).astype(np.int8))
    return bundle(parts)


def act_set(turn: dict) -> tuple:
    acts = set()
    for frame in turn.get("frames", []):
        for action in frame.get("actions", []):
            acts.add(action["act"])
    return tuple(sorted(acts))


def load_dialogues(data_dir: Path) -> list:
    dialogues = []
    for path in sorted(data_dir.glob("dialogues_*.json")):
        dialogues.extend(json.loads(path.read_text()))
    return dialogues


def build_examples(dialogues: list, sym: SymbolTable, decay: float):
    """Per (dialogue, t>=1): encode turns[0..t-1] three ways, target = acts of turn t."""
    enc = {"strand": [], "bundle": [], "last-turn": []}
    targets, prev_sets = [], []
    for dlg in dialogues:
        turns = dlg["turns"]
        codons = [turn_glyph(t, sym) for t in turns]
        sets = [act_set(t) for t in turns]
        strand = Strand()
        acc = None
        for t in range(len(turns) - 1):
            strand.append(codons[t])
            acc = codons[t].astype(np.int32) if acc is None else acc + codons[t]
            enc["strand"].append(strand.state(decay))
            enc["bundle"].append(np.where(acc >= 0, 1, -1).astype(np.int8))
            enc["last-turn"].append(codons[t])
            targets.append(sets[t + 1])
            prev_sets.append(sets[t])
    return enc, targets, prev_sets


def jaccard(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def knn_eval(train_vecs, train_targets, test_vecs, test_targets, block=512):
    """1-NN by dot product (bipolar vectors share one norm)."""
    train_m = np.stack(train_vecs).astype(np.float32)
    exact, jac = 0, 0.0
    for start in range(0, len(test_vecs), block):
        chunk = np.stack(test_vecs[start:start + block]).astype(np.float32)
        nearest = np.argmax(chunk @ train_m.T, axis=1)
        for i, n in enumerate(nearest):
            pred, true = train_targets[n], test_targets[start + i]
            exact += pred == true
            jac += jaccard(pred, true)
    n = len(test_vecs)
    return exact / n, jac / n


def symbolic_baselines(train_targets, train_prev, test_targets, test_prev):
    majority = Counter(train_targets).most_common(1)[0][0]
    bigram = defaultdict(Counter)
    for prev, nxt in zip(train_prev, train_targets):
        bigram[prev][nxt] += 1
    results = {}
    for name, predict in [
        ("majority", lambda prev: majority),
        ("bigram", lambda prev: bigram[prev].most_common(1)[0][0]
         if prev in bigram else majority),
    ]:
        exact, jac = 0, 0.0
        for prev, true in zip(test_prev, test_targets):
            pred = predict(prev)
            exact += pred == true
            jac += jaccard(pred, true)
        results[name] = (exact / len(test_targets), jac / len(test_targets))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    dialogues = load_dialogues(args.data_dir)
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(dialogues))
    n_test = int(len(dialogues) * args.test_frac)
    test_ids = set(order[:n_test].tolist())
    train_dlgs = [d for i, d in enumerate(dialogues) if i not in test_ids]
    test_dlgs = [d for i, d in enumerate(dialogues) if i in test_ids]
    print(f"dialogues: {len(train_dlgs)} train / {len(test_dlgs)} test, "
          f"dim={args.dimension}, decay={args.decay}")

    sym = SymbolTable(args.dimension)
    t0 = time.time()
    train_enc, train_targets, train_prev = build_examples(train_dlgs, sym, args.decay)
    test_enc, test_targets, test_prev = build_examples(test_dlgs, sym, args.decay)
    print(f"examples: {len(train_targets)} train / {len(test_targets)} test "
          f"({len(set(train_targets))} distinct act-sets) "
          f"[encoded in {time.time() - t0:.1f}s]")

    if len(train_targets) > args.max_train:
        keep = rng.choice(len(train_targets), args.max_train, replace=False)
        for k in train_enc:
            train_enc[k] = [train_enc[k][i] for i in keep]
        train_prev = [train_prev[i] for i in keep]
        train_targets = [train_targets[i] for i in keep]

    print(f"\n{'encoder':<12} {'exact':>8} {'jaccard':>8}")
    for name, (exact, jac) in symbolic_baselines(
            train_targets, train_prev, test_targets, test_prev).items():
        print(f"{name:<12} {exact:>8.3f} {jac:>8.3f}")
    for name in ["last-turn", "bundle", "strand"]:
        t0 = time.time()
        exact, jac = knn_eval(train_enc[name], train_targets,
                              test_enc[name], test_targets)
        print(f"{name:<12} {exact:>8.3f} {jac:>8.3f}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
