"""
Convert action JSONL entries into SDK Concepts for the actions model.

Each entry becomes a Concept with:
- Intent attributes (verb, object, domain, keywords) -> encoded for similarity
- Action attributes (action_type, action_name) -> encoded to distinguish actions
- Metadata (endpoint, slots, auth, destructive, template) -> NOT encoded, read from match
"""

import re
from glyphh.core.types import Concept


_VERB_MAP = {
    "reset": "manage", "forgot": "manage", "change": "manage",
    "delete": "manage", "remove": "manage", "cancel": "manage",
    "create": "build", "make": "build", "generate": "build", "new": "build",
    "show": "query", "list": "query", "get": "query", "check": "query",
    "view": "query", "see": "query", "what": "query", "who": "query",
    "upgrade": "manage", "downgrade": "manage", "switch": "manage",
    "invite": "manage", "add": "manage", "rename": "manage",
    "deploy": "deploy", "launch": "deploy", "start": "deploy",
    "stop": "manage", "shut": "manage", "kill": "manage", "pause": "manage",
    "open": "navigate", "go": "navigate",
    "log": "manage", "logout": "manage", "sign": "manage",
}

_KNOWN_OBJECTS = [
    "password", "account", "plan", "billing", "invoice", "subscription",
    "member", "org", "organization", "token", "key",
    "model", "runtime", "logs", "portal",
]

_STOP_WORDS = {
    "my", "me", "i", "a", "the", "to", "is", "can", "you", "please",
    "do", "am", "on", "in", "out", "up", "of", "for",
}


def _extract_verb(text: str) -> str:
    for w in text.split():
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _VERB_MAP:
            return _VERB_MAP[clean]
    return "manage"


def _extract_object(text: str) -> str:
    words = text.lower().split()
    for obj in _KNOWN_OBJECTS:
        if obj in words:
            return obj
    for w in reversed(words):
        clean = re.sub(r"[^a-z]", "", w)
        if clean and clean not in _STOP_WORDS:
            return clean
    return "general"


def action_entry_to_concept(entry: dict) -> Concept:
    """Convert a JSONL action entry to a Concept."""
    trigger = entry.get("trigger", "").lower()
    action_type = entry.get("action_type", "")
    action_name = entry.get("action_name", "")
    keywords = entry.get("keywords", [])

    verb = _extract_verb(trigger)
    obj = _extract_object(trigger)
    kw_str = " ".join(keywords) if isinstance(keywords, list) else str(keywords)

    attributes = {
        "verb": verb,
        "object": obj,
        "domain": action_type,
        "keywords": kw_str,
        "action_type": action_type,
        "action_name": action_name,
    }

    metadata = {
        "action_name": action_name,
        "action_type": action_type,
        "endpoint": entry.get("endpoint", ""),
        "required_slots": entry.get("required_slots", ""),
        "slot_defaults": entry.get("slot_defaults", {}),
        "auth_required": entry.get("auth_required", True),
        "destructive": entry.get("destructive", False),
        "response_template": entry.get("response_template", ""),
        "original_trigger": entry.get("trigger", ""),
    }

    name = f"action_{action_name}"
    return Concept(name=name, attributes=attributes, metadata=metadata)
