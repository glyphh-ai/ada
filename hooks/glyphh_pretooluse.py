#!/usr/bin/env python3
"""
PreToolUse hook — enforcement. The "can't get around it" teeth.

Fires before a tool runs. Checks the pending action against Glyphh's
deterministic guards; if it violates one, returns permissionDecision "deny"
so Claude Code blocks it. This is what turns a remembered decision from a
reminder into an unbreakable constraint.

Fails OPEN: if the runtime is down or the action is clean, it stays silent and
normal permission flow proceeds — a memory outage never wedges the agent.
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from glyphh_client import check_action  # noqa: E402


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return  # silent → allow
    tool = data.get("tool_name") or data.get("tool") or ""
    tool_input = data.get("tool_input") or {}
    if not tool:
        return

    result = check_action(tool, tool_input)
    if result.get("decision") == "deny":
        reason = result.get("reason", "Blocked by a Glyphh guard.")
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Glyphh guard: {reason}",
            }
        }))
    # allow / warn → stay silent (exit 0): defer to normal permission flow.


if __name__ == "__main__":
    main()
