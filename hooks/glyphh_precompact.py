#!/usr/bin/env python3
"""
PreCompact hook — flush decisions before the window collapses.

Fires right before compaction. Side-effect only (it cannot change what gets
compacted), which is exactly what we want: scan the about-to-be-summarized
transcript for durable instructions and remember() them, so the lossy summary
can't drop them. Belt-and-suspenders alongside the UserPromptSubmit capture.

Best-effort and silent: never blocks compaction.
"""
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from glyphh_client import remember  # noqa: E402

_DECISION_RE = re.compile(
    r"\b(always|never|from now on|going forward|make sure|you must|"
    r"don'?t|do not|remember to|whenever|before you|each time)\b",
    re.IGNORECASE,
)


def _text_of(message: dict) -> str:
    """Extract plain text from a transcript message (string or block list)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    path = data.get("transcript_path")
    if not path:
        return
    try:
        with open(path, "r") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                msg = ev.get("message", {})
                if msg.get("role") != "user":
                    continue
                text = _text_of(msg).strip()
                if text and _DECISION_RE.search(text):
                    remember(text, speaker="incoming")
    except Exception:
        pass  # never block compaction


if __name__ == "__main__":
    main()
