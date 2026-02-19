"""
Router-style encoder config for the Glyphh assistant model.

This is the canonical source for the assistant encoder configuration.
Previously imported from models/assistant/encoder.py, now self-contained.
"""

import hashlib
import re
from glyphh.core.config import EncoderConfig, Layer, Segment, Role, NumericConfig, EncodingStrategy, TemporalConfig
from glyphh.core.types import Concept

# ---------------------------------------------------------------------------
# Encoder config
# ---------------------------------------------------------------------------

CONTEXT_TYPE_NUMERIC = NumericConfig(
    bin_width=1.0,
    encoding_strategy=EncodingStrategy.THERMOMETER,
    min_value=0.0,
    max_value=2.0,
)

CONTEXT_TYPE_MAP = {"standalone": 1.0}

ASSISTANT_ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    temporal_source="router.context.context_type",
    temporal_config=TemporalConfig(signal_type="sequence"),
    layers=[
        Layer(
            name="router",
            similarity_weight=1.0,
            segments=[
                Segment(
                    name="intent",
                    roles=[
                        Role(name="verb", similarity_weight=1.0),
                        Role(name="object", similarity_weight=0.9),
                        Role(name="domain", similarity_weight=1.0),
                        Role(name="keywords", similarity_weight=0.7),
                    ],
                ),
                Segment(
                    name="action",
                    roles=[
                        Role(name="action_type", similarity_weight=0.8),
                        Role(name="action_id", similarity_weight=0.6),
                    ],
                ),
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="context_type",
                            similarity_weight=1.0,
                            numeric_config=CONTEXT_TYPE_NUMERIC,
                        ),
                    ],
                ),
            ],
        )
    ],
)

# Alias for backwards compatibility
ENCODER_CONFIG = ASSISTANT_ENCODER_CONFIG


# ---------------------------------------------------------------------------
# Query encoding — converts raw NL text to a Concept for this model
# ---------------------------------------------------------------------------

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
    "hello": "greeting", "hi": "greeting", "hey": "greeting",
    "howdy": "greeting", "yo": "greeting", "sup": "greeting",
    "greetings": "greeting", "morning": "greeting",
}

_KNOWN_OBJECTS = [
    "model", "query", "data", "runtime", "encoder", "config",
    "gql", "hdc", "vector", "glyph", "glyphh", "concept", "role", "segment",
    "layer", "procedure", "pattern", "account", "package", "results",
    "threshold", "similarity", "filter", "limit", "schema",
]

_STOP_WORDS = {
    "how", "do", "i", "a", "the", "to", "is", "what", "my", "an",
    "can", "does", "it", "in", "on", "for", "with", "me", "about",
    "hello", "hi", "hey", "howdy", "yo", "sup", "greetings", "morning",
    "good", "afternoon", "evening",
}

_ACTION_TYPE_MAP = {
    "command": "command",
    "concept": "response",
    "gql": "query",
    "quick_action": "quick_action",
    "workflow": "response",
}


def _extract_verb(words):
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _VERB_MAP:
            return _VERB_MAP[clean]
    return "help"


def _extract_object(words):
    for obj in _KNOWN_OBJECTS:
        if obj in words:
            return obj
    for w in reversed(words):
        clean = re.sub(r"[^a-z]", "", w)
        if clean and clean not in _STOP_WORDS:
            return clean
    return "general"


def _infer_domain(words):
    text = " ".join(words)
    if any(w in text for w in ["hello", "hi", "hey", "howdy", "yo", "sup", "greetings", "morning"]):
        return "greeting"
    if any(w in text for w in ["build", "create", "init", "add"]):
        return "build"
    if any(w in text for w in ["find", "search", "query", "similar"]):
        return "query"
    if any(w in text for w in ["deploy", "publish", "release"]):
        return "deploy"
    if any(w in text for w in ["what", "explain", "how does"]):
        return "explain"
    return "help"


def encode_query(query: str) -> Concept:
    """Convert a raw NL query string into a Concept for this model."""
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    words = cleaned.split()
    verb = _extract_verb(words)
    obj = _extract_object(words)
    domain = _infer_domain(words)
    keywords = " ".join(words)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return Concept(
        name=f"query_{stable_id:08d}",
        attributes={
            "verb": verb,
            "object": obj,
            "domain": domain,
            "keywords": keywords,
            "action_type": "",
            "action_id": "",
            "context_type": 1.0,
        },
    )


def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL entry to an encodable record + metadata."""
    question = entry.get("question", "").lower()
    words = question.split()
    intent = entry.get("intent", "")
    kw_list = entry.get("keywords", [])
    content_type = entry.get("content_type", "")
    context_type_str = entry.get("context_type", "standalone")

    verb = _extract_verb(words)
    obj = _extract_object(words)
    domain = intent if intent else _infer_domain(words)
    kw_str = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)
    action_type = _ACTION_TYPE_MAP.get(content_type, "response")

    slug = re.sub(r"[^a-z0-9]+", "_", question).strip("_")[:40]
    action_id = f"{content_type}_{intent}_{slug}" if intent else f"{content_type}_{slug}"

    return {
        "concept_text": entry.get("question", ""),
        "attributes": {
            "verb": verb,
            "object": obj,
            "domain": domain,
            "keywords": kw_str,
            "action_type": action_type,
            "action_id": action_id,
            "context_type": CONTEXT_TYPE_MAP.get(context_type_str, 1.0),
        },
        "metadata": {
            "response": entry.get("response", ""),
            "command": entry.get("command"),
            "code": entry.get("code"),
            "content_type": content_type,
            "original_question": entry.get("question", ""),
        },
    }
