"""
Benchmark runner for glyphh.intent.IntentExtractor.

Measures extraction accuracy across query categories:
  - clear: Unambiguous queries
  - synonym: Queries using uncommon synonyms
  - phrase: Multi-word phrase matching
  - domain: Domain inference accuracy
  - ambiguous: Competing action/target signals
  - adversarial: Misleading phrasing and edge cases

Usage:
    python tests/intent/benchmark/run.py
    python tests/intent/benchmark/run.py --verbose
    python tests/intent/benchmark/run.py --output tests/intent/benchmark/results/
    python tests/intent/benchmark/run.py --packs filesystem trading
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from glyphh.intent import IntentExtractor


def load_queries() -> dict:
    path = Path(__file__).parent / "queries.json"
    with open(path) as f:
        return json.load(f)


def run_benchmark(
    verbose: bool = False,
    output_dir: str | None = None,
    packs: list[str] | None = None,
):
    print("=" * 70)
    print("glyphh.intent — IntentExtractor Benchmark")
    print("=" * 70)
    if packs:
        print(f"Packs: {', '.join(packs)}")
    print()

    t0 = time.perf_counter()
    extractor = IntentExtractor(packs=packs)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"Build time: {build_time:.1f}ms")
    print()

    data = load_queries()
    queries = data["queries"]

    results = []
    stats = defaultdict(lambda: {"total": 0, "action": 0, "target": 0, "domain": 0, "all": 0})
    total_time = 0.0

    for q in queries:
        t0 = time.perf_counter()
        result = extractor.extract(q["query"])
        elapsed = (time.perf_counter() - t0) * 1000
        total_time += elapsed

        action_ok = result["action"] == q["expected_action"]
        target_ok = result["target"] == q["expected_target"]
        domain_ok = result["domain"] == q["expected_domain"]
        all_ok = action_ok and target_ok and domain_ok

        cat = q["category"]
        stats[cat]["total"] += 1
        if action_ok: stats[cat]["action"] += 1
        if target_ok: stats[cat]["target"] += 1
        if domain_ok: stats[cat]["domain"] += 1
        if all_ok:    stats[cat]["all"] += 1

        results.append({
            "id": q["id"], "category": cat, "query": q["query"],
            "expected": {"action": q["expected_action"], "target": q["expected_target"], "domain": q["expected_domain"]},
            "actual": {"action": result["action"], "target": result["target"], "domain": result["domain"], "keywords": result["keywords"]},
            "correct": {"action": action_ok, "target": target_ok, "domain": domain_ok, "all": all_ok},
            "latency_ms": round(elapsed, 3),
        })

        if verbose and not all_ok:
            print(f"  FAIL [{q['id']}] {q['query']}")
            if not action_ok: print(f"    action: {result['action']} (expected {q['expected_action']})")
            if not target_ok: print(f"    target: {result['target']} (expected {q['expected_target']})")
            if not domain_ok: print(f"    domain: {result['domain']} (expected {q['expected_domain']})")

    total = len(queries)
    a_total = sum(s["action"] for s in stats.values())
    t_total = sum(s["target"] for s in stats.values())
    d_total = sum(s["domain"] for s in stats.values())
    all_total = sum(s["all"] for s in stats.values())

    print()
    print("=" * 70)
    print(f"{'Category':<15} {'N':>5} {'Action':>8} {'Target':>8} {'Domain':>8} {'All':>8}")
    print("-" * 55)
    for cat in sorted(stats.keys()):
        s = stats[cat]
        n = s["total"]
        print(f"{cat:<15} {n:>5} {s['action']:>5}/{n:<2} {s['target']:>5}/{n:<2} {s['domain']:>5}/{n:<2} {s['all']:>5}/{n:<2}")
    print("-" * 55)
    print(f"{'TOTAL':<15} {total:>5} {a_total:>5}/{total:<2} {t_total:>5}/{total:<2} {d_total:>5}/{total:<2} {all_total:>5}/{total:<2}")
    print()
    print(f"Action accuracy:  {a_total / total * 100:.1f}%")
    print(f"Target accuracy:  {t_total / total * 100:.1f}%")
    print(f"Domain accuracy:  {d_total / total * 100:.1f}%")
    print(f"Overall accuracy: {all_total / total * 100:.1f}%")
    print(f"Avg latency:      {total_time / total:.3f}ms/query")
    print(f"Total time:       {total_time:.1f}ms")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_path = out / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "metadata": {"total_queries": total, "build_time_ms": round(build_time, 1),
                             "total_time_ms": round(total_time, 1), "avg_latency_ms": round(total_time / total, 3),
                             "packs": packs or []},
                "accuracy": {"action": round(a_total / total * 100, 1), "target": round(t_total / total * 100, 1),
                             "domain": round(d_total / total * 100, 1), "overall": round(all_total / total * 100, 1)},
                "category_stats": dict(stats),
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run glyphh.intent benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show failures inline")
    parser.add_argument("--output", "-o", type=str, help="Directory to save JSON results")
    parser.add_argument("--packs", nargs="*", help="Domain packs to load (e.g. filesystem trading)")
    args = parser.parse_args()

    run_benchmark(verbose=args.verbose, output_dir=args.output, packs=args.packs or None)
