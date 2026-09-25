"""
Real-world strand evaluation #2: trajectory anomaly detection.

Scenario: a session that stops behaving like itself — account takeover,
prompt injection steering an agent off-course, a workflow going off the
rails. Train a StrandPredictor on normal sessions, then score each
incoming session by how far its actual continuations rank below the
expected ones (the margin score).

Test construction, from Schema-Guided Dialogue:
  normal    = held-out real dialogues
  hijacked  = franken-sessions spliced from the front half of one real
              dialogue and the back half of another (different service) —
              every turn is individually plausible; only the TRAJECTORY
              is wrong.

Labels are (act, slot) pairs per turn: slots are service-specific, so
they carry the content signal a bare act name lacks.

Findings this eval encodes (and re-checks): the two views of the same
codons split the work. The strand state (ordered, decayed) wins at
next-act PREDICTION (see strand_dialog_eval.py) but its decay forgets
the session's past and adapts to a hijack within a few turns; the bundle
state (unordered, undecayed) keeps the whole session's identity and wins
at DRIFT DETECTION. Genotype and phenotype — keep both.

Metric: ROC AUC of mean session margin separating hijacked from normal.

Data: any subset of the train/dialogues_*.json files from
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

Usage:
    python tests/eval/strand_anomaly_eval.py --data-dir /path/to/sgd \
        [--dimension 4096] [--decay 0.7]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from glyphh.strand import Strand
from glyphh.strand.predictor import StrandPredictor

sys.path.insert(0, str(Path(__file__).parent))
from strand_dialog_eval import SymbolTable, turn_glyph  # noqa: E402

SEED = 42


def service_of(dlg: dict) -> str:
    return dlg["services"][0]


def act_slot_set(turn: dict) -> tuple:
    out = set()
    for frame in turn.get("frames", []):
        for action in frame.get("actions", []):
            out.add((action["act"], action.get("slot") or "none"))
    return tuple(sorted(out))


def encode_dialogue(dlg: dict, sym: SymbolTable):
    turns = dlg["turns"]
    codons = [turn_glyph(t, sym) for t in turns]
    labels = [act_slot_set(t) for t in turns]
    return codons, labels


def session_steps(codons, labels, decay: float, encoder: str):
    """(state-before-turn-t, label-of-turn-t) pairs for one session."""
    states, next_labels = [], []
    strand = Strand()
    acc = None
    for t in range(len(codons) - 1):
        strand.append(codons[t])
        if encoder == "strand":
            states.append(strand.state(decay))
        else:
            acc = codons[t].astype(np.int32) if acc is None else acc + codons[t]
            states.append(np.where(acc >= 0, 1, -1).astype(np.int8))
        next_labels.append(labels[t + 1])
    return states, next_labels


def make_hijacked(dialogues, rng, n: int):
    """Splice front of one dialogue onto back of another, different service."""
    franken = []
    attempts = 0
    while len(franken) < n and attempts < n * 20:
        attempts += 1
        a, b = rng.choice(len(dialogues), 2, replace=False)
        da, db = dialogues[a], dialogues[b]
        if service_of(da) == service_of(db):
            continue
        ta, tb = da["turns"], db["turns"]
        cut_a, cut_b = len(ta) // 2, len(tb) // 2
        # keep speaker alternation intact across the splice
        if ta[cut_a - 1]["speaker"] == tb[cut_b]["speaker"]:
            cut_b += 1
        if cut_b >= len(tb) - 1:
            continue
        franken.append({
            "services": [service_of(da)],
            "turns": ta[:cut_a] + tb[cut_b:],
        })
    return franken


def auc(pos_scores, neg_scores) -> float:
    """Rank-based ROC AUC: P(hijacked score > normal score)."""
    pos, neg = np.asarray(pos_scores), np.asarray(neg_scores)
    order = np.concatenate([pos, neg]).argsort().argsort() + 1
    rank_sum = order[: len(pos)].sum()
    return float(
        (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    dialogues = []
    for path in sorted(args.data_dir.glob("dialogues_*.json")):
        dialogues.extend(json.loads(path.read_text()))
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(dialogues))
    n_test = int(len(dialogues) * args.test_frac)
    test_ids = set(order[:n_test].tolist())
    train = [d for i, d in enumerate(dialogues) if i not in test_ids]
    normal = [d for i, d in enumerate(dialogues) if i in test_ids]
    hijacked = make_hijacked(normal, rng, len(normal))
    print(f"train {len(train)}, normal test {len(normal)}, "
          f"hijacked test {len(hijacked)}, dim={args.dimension}")

    sym = SymbolTable(args.dimension)
    for encoder in ["strand", "bundle"]:
        predictor = StrandPredictor(args.dimension)
        for dlg in train:
            codons, labels = encode_dialogue(dlg, sym)
            for state, label in zip(*session_steps(
                    codons, labels, args.decay, encoder)):
                predictor.observe(state, label)

        def score(dlg):
            codons, labels = encode_dialogue(dlg, sym)
            states, next_labels = session_steps(
                codons, labels, args.decay, encoder)
            return float(np.mean([
                predictor.margin(s, lab)
                for s, lab in zip(states, next_labels)
            ]))

        normal_scores = [score(d) for d in normal]
        hijack_scores = [score(d) for d in hijacked]
        print(f"{encoder:<8} AUC {auc(hijack_scores, normal_scores):.3f}   "
              f"mean margin: normal {np.mean(normal_scores):.3f}, "
              f"hijacked {np.mean(hijack_scores):.3f}")


if __name__ == "__main__":
    main()
