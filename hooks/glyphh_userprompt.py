#!/usr/bin/env python3
"""
UserPromptSubmit hook — the re-grounding star.

Fires before Claude sees each prompt. Two jobs, both deterministic, LLM-free:
  1. CAPTURE: if the prompt is a durable instruction ("always X", "never Y",
     "from now on Z"), remember() it so it outlives compaction.
  2. RE-GROUND: recall() decisions relevant to this prompt and inject them via
     hookSpecificOutput.additionalContext — so the model re-sees them THIS turn,
     no matter how many compactions have wiped them from the window.

Best-effort: if the Glyphh runtime is down, emits empty context and never
blocks the prompt.
"""
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from glyphh_client import recall, remember  # noqa: E402

# Prompts that look like durable instructions get captured as decisions.
_DECISION_RE = re.compile(
    r"\b(always|never|from now on|going forward|make sure|you must|"
    r"don'?t|do not|remember to|whenever|before you|each time)\b",
    re.IGNORECASE,
)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    prompt = (data.get("prompt") or "").strip()

    ctx = ""
    if prompt:
        # 1. capture durable instructions
        if _DECISION_RE.search(prompt):
            remember(prompt, speaker="incoming")
        # 2. re-ground from memory (exclude Ada's own seed identity — we want
        #    project decisions/user facts, not brain trivia)
        facts = recall(prompt, top_k=5, min_confidence=0.35,
                       exclude_speakers=["ada"])
        if facts:
            lines = [
                "Relevant memory from Glyphh (persists across compaction — "
                "treat as authoritative decisions/constraints):"
            ]
            for f in facts:
                lines.append(f"  - {f['content']}")
            ctx = "\n".join(lines)

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": ctx,
        }
    }))


if __name__ == "__main__":
    main()
