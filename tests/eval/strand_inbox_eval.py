"""
Real-world strand evaluation #4: predict tomorrow's inbox.

Walk-forward test on a real mailbox: each day is a codon (weekday +
volume bucket + sender-domain bindings), the mailbox is a strand of
days, and for every day D the model sees only days before D, finds the
k most similar historical states, and predicts day D's traffic from
what followed those states.

Predictions scored per day:
  senders   which sender domains appear tomorrow (Jaccard + precision@3)
  volume    tomorrow's volume tercile (low/med/high, cut on history only)

Baselines: yesterday-again (persistence), same-weekday-last-week, and
the global top-k domains by frequency so far.

Input: JSONL of {"date": ISO8601, "sender": address}, one per message.

Usage:
    python tests/eval/strand_inbox_eval.py --events /path/to/events.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from glyphh.core.ops import bundle, generate_symbol
from glyphh.strand import Strand

SEED = 42
WARMUP_DAYS = 7


def domain_of(sender: str) -> str:
    host = sender.rsplit("@", 1)[-1].lower()
    return ".".join(host.split(".")[-2:])


def load_days(path: Path):
    """Ordered list of (date, Counter of domains), empty days included."""
    per_day = defaultdict(Counter)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        day = date.fromisoformat(row["date"][:10])
        per_day[day][domain_of(row["sender"])] += 1
    first, last = min(per_day), max(per_day)
    days = []
    d = first
    while d <= last:
        days.append((d, per_day.get(d, Counter())))
        d += timedelta(days=1)
    return days


class InboxEncoder:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._cache = {}

    def sym(self, key: str):
        if key not in self._cache:
            self._cache[key] = generate_symbol(SEED, key, self.dimension)
        return self._cache[key]

    def day_codon(self, day: date, domains: Counter, volume_bucket: int):
        parts = [
            self.sym("dow") * self.sym(f"dow:{day.weekday()}"),
            self.sym("vol") * self.sym(f"vol:{volume_bucket}"),
        ]
        for dom in domains or {"(empty)": 1}:
            parts.append(
                (self.sym("from") * self.sym(f"dom:{dom}")).astype(np.int8))
        return bundle(parts)


def bucket(volume: int, history_volumes) -> int:
    cuts = np.percentile(history_volumes, [33, 66]) if history_volumes else [1, 3]
    return int(volume > cuts[0]) + int(volume > cuts[1])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def precision_at(predicted: list, actual: set, k: int) -> float:
    if not predicted:
        return 1.0 if not actual else 0.0
    top = predicted[:k]
    return sum(p in actual for p in top) / len(top)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--knn", type=int, default=3)
    args = parser.parse_args()

    days = load_days(args.events)
    enc = InboxEncoder(args.dimension)
    print(f"{len(days)} days, {sum(sum(c.values()) for _, c in days)} messages, "
          f"{days[0][0]} .. {days[-1][0]}\n")

    scores = defaultdict(list)
    strand = Strand()
    past_states, volumes = [], []

    for i, (day, domains) in enumerate(days):
        actual_set = set(domains)
        actual_vol = sum(domains.values())

        if i >= WARMUP_DAYS:
            state = strand.state(args.decay)
            sims = np.array([
                float(np.dot(s.astype(np.int32), state.astype(np.int32)))
                for s in past_states
            ])
            # neighbors among past states whose NEXT day is known (< i-1)
            valid = len(past_states) - 1
            top = np.argsort(sims[:valid])[-args.knn:]
            dom_votes, vol_votes = Counter(), []
            for j in top:
                nxt_domains = days[j + 1][1]
                dom_votes.update(set(nxt_domains))
                vol_votes.append(sum(nxt_domains.values()))
            ranked = [d for d, _ in dom_votes.most_common()]
            majority_n = max(1, int(np.median([len(set(days[j + 1][1])) for j in top])))
            pred_set = set(ranked[:majority_n])

            hist_vols = volumes[:-1] if len(volumes) > 1 else volumes
            pred_bucket = int(np.round(np.mean(
                [bucket(v, hist_vols) for v in vol_votes])))
            actual_bucket = bucket(actual_vol, hist_vols)

            scores["strand jaccard"].append(jaccard(pred_set, actual_set))
            scores["strand p@3"].append(precision_at(ranked, actual_set, 3))
            scores["strand vol-acc"].append(pred_bucket == actual_bucket)

            yesterday = set(days[i - 1][1])
            scores["persist jaccard"].append(jaccard(yesterday, actual_set))
            scores["persist p@3"].append(
                precision_at(sorted(yesterday), actual_set, 3))
            last_week = set(days[i - 7][1]) if i >= 7 else set()
            scores["weekday jaccard"].append(jaccard(last_week, actual_set))
            freq = Counter()
            for _, c in days[:i]:
                freq.update(set(c))
            global_ranked = [d for d, _ in freq.most_common()]
            scores["global p@3"].append(
                precision_at(global_ranked, actual_set, 3))
            scores["global jaccard"].append(
                jaccard(set(global_ranked[:majority_n]), actual_set))
            scores["persist vol-acc"].append(
                bucket(sum(days[i - 1][1].values()), hist_vols) == actual_bucket)

        volumes.append(actual_vol)
        vol_b = bucket(actual_vol, volumes[:-1] if len(volumes) > 1 else volumes)
        codon = enc.day_codon(day, domains, vol_b)
        strand.append(codon)
        past_states.append(strand.state(args.decay))

    n = len(scores["strand jaccard"])
    print(f"walk-forward over {n} predicted days "
          f"(first {WARMUP_DAYS} days are warmup)\n")
    print(f"{'metric':<18} {'strand':>8} {'persist':>8} {'weekday':>8} {'global':>8}")
    print(f"{'sender jaccard':<18} {np.mean(scores['strand jaccard']):>8.3f} "
          f"{np.mean(scores['persist jaccard']):>8.3f} "
          f"{np.mean(scores['weekday jaccard']):>8.3f} "
          f"{np.mean(scores['global jaccard']):>8.3f}")
    print(f"{'sender p@3':<18} {np.mean(scores['strand p@3']):>8.3f} "
          f"{np.mean(scores['persist p@3']):>8.3f} {'-':>8} "
          f"{np.mean(scores['global p@3']):>8.3f}")
    print(f"{'volume bucket acc':<18} {np.mean(scores['strand vol-acc']):>8.3f} "
          f"{np.mean(scores['persist vol-acc']):>8.3f} {'-':>8} {'-':>8}")


if __name__ == "__main__":
    main()
