#!/usr/bin/env python3
"""
SessionStart hook — load standing decisions when a session begins/resumes.

Injects the project's standing constraints so the agent starts grounded, even
on a fresh session (memory survives across sessions, not just compaction).
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from glyphh_client import recall  # noqa: E402


def main() -> None:
    facts = recall(
        "project decisions conventions constraints rules to follow",
        top_k=8, min_confidence=0.35, exclude_speakers=["ada"],
    )
    ctx = ""
    if facts:
        lines = ["Standing decisions/constraints from Glyphh memory:"]
        for f in facts:
            lines.append(f"  - {f['content']}")
        ctx = "\n".join(lines)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": ctx,
        }
    }))


if __name__ == "__main__":
    main()
