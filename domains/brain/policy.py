"""
Policy store — deterministic enforcement rules for agent actions.

This is the "can't get around it" half. Re-grounding (recall) *reminds* the
agent of a decision; a policy rule *blocks* the action that violates it. For a
real guarantee the match must be deterministic — a tool filter + a regex over
the tool input — not fuzzy similarity. Fuzzy recall can surface a candidate
rule, but the deny itself is precise and reproducible.

Rules persist to JSON (GLYPHH_POLICY_FILE, default ~/.glyphh/policy.json) so
they survive restarts.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class Rule:
    """A single enforcement rule.

    tool:    tool name to match ("Bash", "Edit", ...) or "*" for any.
    pattern: regex matched (case-insensitive) against json.dumps(tool_input).
    action:  "deny" (block) or "warn" (allow but flag).
    reason:  shown to the agent when the rule fires.
    """
    pattern: str
    reason: str
    tool: str = "*"
    action: str = "deny"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


def _default_path() -> str:
    return os.environ.get(
        "GLYPHH_POLICY_FILE",
        os.path.join(os.path.expanduser("~"), ".glyphh", "policy.json"),
    )


class PolicyStore:
    def __init__(self, path: str | None = None):
        self._path = path or _default_path()
        self._rules: list[Rule] = []
        self._load()

    # ── persistence ──
    def _load(self) -> None:
        try:
            with open(self._path) as fh:
                self._rules = [Rule(**r) for r in json.load(fh)]
        except Exception:
            self._rules = []

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as fh:
                json.dump([asdict(r) for r in self._rules], fh, indent=2)
        except Exception:
            pass  # best-effort; in-memory rules still enforce this session

    # ── api ──
    def add(self, pattern: str, reason: str, tool: str = "*",
            action: str = "deny") -> Rule:
        rule = Rule(pattern=pattern, reason=reason, tool=tool, action=action)
        self._rules.append(rule)
        self._save()
        return rule

    def rules(self) -> list[Rule]:
        return list(self._rules)

    def clear(self) -> None:
        self._rules = []
        self._save()

    def evaluate(self, tool: str, tool_input: dict) -> dict | None:
        """Return the first matching rule as a decision, or None (allow).

        {"decision": "deny"|"warn", "reason": ..., "rule_id": ...}
        """
        try:
            blob = json.dumps(tool_input or {}, ensure_ascii=False)
        except Exception:
            blob = str(tool_input)
        for rule in self._rules:
            if rule.tool not in ("*", tool):
                continue
            try:
                if re.search(rule.pattern, blob, re.IGNORECASE):
                    return {"decision": rule.action, "reason": rule.reason,
                            "rule_id": rule.id}
            except re.error:
                # bad regex → fall back to substring match
                if rule.pattern.lower() in blob.lower():
                    return {"decision": rule.action, "reason": rule.reason,
                            "rule_id": rule.id}
        return None
