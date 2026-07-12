"""Unit test for the enrich front half (pure Python — no glyphh/DB needed).

Run:  python api/enrich/test_enrich.py     (from the glyphh-runtime root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))     # import enricher/pipe standalone
from pipe import enrich_to_concept                 # noqa: E402  (the real function)


if __name__ == "__main__":
    cases = [
        ("Erin lives in Boston.", "erin", "location", "boston"),
        ("Bob is 42 years old.", "bob", "age", "42"),
        ("Alice's favorite color is green.", "alice", "color", "green"),
        ("The capital of France is Paris.", "france", "location", "paris"),
    ]
    print("enrich -> ConceptInput-shaped payload (flat role attributes):")
    ok = 0
    for text, want_name, want_role, want_val in cases:
        c = enrich_to_concept(text)
        good = c and c["name"] == want_name and c["attributes"].get(want_role) == want_val
        ok += bool(good)
        print(f"  {'ok ' if good else 'MISS'} {text:34} -> name={c['name'] if c else None!r:8} "
              f"attrs={c['attributes'] if c else {}}")
    none_case = enrich_to_concept("hello there, how are you doing today")   # non-fact -> nothing to ground
    print(f"  {'ok ' if none_case is None else 'MISS'} non-fact text -> {none_case}")
    assert ok == len(cases) and none_case is None
    print(f"\n{ok}/{len(cases)} facts -> flat {{role: value}} attributes; non-fact -> None. PASS")
