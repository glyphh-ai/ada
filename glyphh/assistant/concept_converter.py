"""
Convert JSONL knowledge entries and procedures into SDK Concepts
with router-shaped attributes for the assistant model.

Each entry becomes a Concept with:
- Intent attributes (verb, object, domain, keywords) -> encoded for similarity
- Action attributes (action_type, action_id) -> encoded to distinguish actions
- Metadata (response, command, code, etc.) -> NOT encoded, read from match
"""

import re
from typing import Optional
from glyphh.core.types import Concept
from glyphh.assistant.encoder_config import CONTEXT_TYPE_MAP


# Verb extraction: map question-leading words to canonical verbs
_VERB_MAP = {
    "how": "help", "what": "explain", "show": "help", "tell": "explain",
    "explain": "explain", "describe": "explain", "why": "explain",
    "create": "build", "build": "build", "init": "build", "initialize": "build",
    "add": "build", "make": "build", "setup": "build", "set": "build",
    "find": "query", "search": "query", "look": "query", "filter": "query",
    "query": "query", "list": "query", "get": "query",
    "deploy": "deploy", "publish": "deploy", "release": "deploy",
    "run": "deploy", "start": "deploy", "launch": "deploy",
    "delete": "manage", "remove": "manage", "update": "manage",
    "navigate": "navigate", "go": "navigate", "open": "navigate",
    "type": "navigate",
}

# Object extraction: known nouns in the Glyphh domain
_KNOWN_OBJECTS = [
    "model", "query", "data", "runtime", "encoder", "config",
    "gql", "hdc", "vector", "glyph", "glyphh", "concept", "role", "segment",
    "layer", "procedure", "pattern", "account", "package", "results",
    "threshold", "similarity", "filter", "limit", "schema",
]

# Content type -> action_type mapping
_ACTION_TYPE_MAP = {
    "command": "command",
    "concept": "response",
    "gql": "query",
    "quick_action": "quick_action",
    "workflow": "response",
}


def extract_intent_attributes(entry: dict) -> dict:
    """Extract verb, object, domain, keywords from a JSONL entry."""
    question = entry.get("question", "").lower()
    intent = entry.get("intent", "")
    keywords = entry.get("keywords", [])
    content_type = entry.get("content_type", "")

    verb = _extract_verb(question)
    obj = _extract_object(question)
    domain = intent if intent else _infer_domain(question)
    kw_str = " ".join(keywords) if isinstance(keywords, list) else str(keywords)

    return {
        "verb": verb,
        "object": obj,
        "domain": domain,
        "keywords": kw_str,
    }


def entry_to_concept(entry: dict) -> Concept:
    """Convert a JSONL entry to a Concept with router attributes + metadata."""
    intent_attrs = extract_intent_attributes(entry)
    content_type = entry.get("content_type", "")
    intent = entry.get("intent", "")

    action_type = _ACTION_TYPE_MAP.get(content_type, "response")
    action_id = _make_action_id(content_type, intent, entry.get("question", ""))
    context_type_str = entry.get("context_type", "standalone")
    context_type = CONTEXT_TYPE_MAP.get(context_type_str, 1.0)  # numeric for binning

    attributes = {
        **intent_attrs,
        "action_type": action_type,
        "action_id": action_id,
        "context_type": context_type,
    }

    metadata = {
        "response": entry.get("response", ""),
        "command": entry.get("command"),
        "code": entry.get("code"),
        "content_type": content_type,
        "original_question": entry.get("question", ""),
    }

    name = f"{content_type}_{action_id}"
    return Concept(name=name, attributes=attributes, metadata=metadata)


def procedure_to_concepts(procedures: list[dict]) -> list[Concept]:
    """Convert procedure definitions to Concepts.

    Each procedure dict should have: name, description, triggers, category,
    response_template, and optionally slots/examples.
    """
    concepts = []
    for proc in procedures:
        name = proc.get("name", "")
        triggers = proc.get("triggers", [])
        category = proc.get("category", "general")
        template = proc.get("response_template", "")
        description = proc.get("description", "")

        # Use first trigger as the question proxy
        question = triggers[0] if triggers else description
        kw = " ".join(triggers)

        verb = _extract_verb(question.lower())
        obj = _extract_object(question.lower())

        attributes = {
            "verb": verb,
            "object": obj,
            "domain": category,
            "keywords": kw,
            "action_type": "response",
            "action_id": f"proc_{name}",
            "context_type": CONTEXT_TYPE_MAP["standalone"],
        }
        metadata = {
            "response": template if template else description,
            "command": None,
            "code": None,
            "content_type": "procedure",
            "original_question": question,
        }
        concepts.append(Concept(name=f"procedure_{name}", attributes=attributes, metadata=metadata))
    return concepts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_verb(question: str) -> str:
    """Extract canonical verb from question text."""
    words = question.split()
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _VERB_MAP:
            return _VERB_MAP[clean]
    return "help"


def _extract_object(question: str) -> str:
    """Extract primary object noun from question text."""
    words = question.lower().split()
    for obj in _KNOWN_OBJECTS:
        if obj in words:
            return obj
    # Fallback: last meaningful word
    stop = {"how", "do", "i", "a", "the", "to", "is", "what", "my", "an", "can", "does", "it", "in", "on", "for", "with"}
    for w in reversed(words):
        clean = re.sub(r"[^a-z]", "", w)
        if clean and clean not in stop:
            return clean
    return "general"


def _infer_domain(question: str) -> str:
    """Infer domain from question when intent field is missing."""
    q = question.lower()
    if any(w in q for w in ["build", "create", "init", "add"]):
        return "build"
    if any(w in q for w in ["find", "search", "query", "similar"]):
        return "query"
    if any(w in q for w in ["deploy", "publish", "release"]):
        return "deploy"
    if any(w in q for w in ["what", "explain", "how does"]):
        return "explain"
    return "help"


def _make_action_id(content_type: str, intent: str, question: str) -> str:
    """Create a unique action_id from entry fields."""
    # Slugify the question for uniqueness
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    if len(slug) > 40:
        slug = slug[:40].rstrip("_")
    base = f"{content_type}_{intent}" if intent else content_type
    return f"{base}_{slug}"
