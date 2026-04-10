"""
Encoder for the Cognitive Router capability.

Routes natural language to the right brain capability using HDC similarity.
Two-layer analysis: cognitive intent + capability domain, plus semantic BoW.

Exports:
  ENCODER_CONFIG — EncoderConfig with intent + domain + semantic layers
  encode_query(query) — converts NL text to a Concept dict
  entry_to_record(entry) — converts a JSONL exemplar to a build record

Architecture:
  Lightweight extraction — no external packs, no heavy IntentExtractor.
  Uses simple keyword/pattern matching for cognitive intent and domain
  classification, with BoW fallback for fuzzy matching.

  Main model (seed=42, dim=2000):
  - Intent layer (0.40): cognitive_action (lexicon) + intent_signals (BoW)
  - Domain layer (0.35): capability_domain (lexicon) + domain_signals (BoW)
  - Semantic layer (0.25): description (BoW) + keywords (BoW)
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Role,
    Segment,
)

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.40,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="cognitive_action",
                            similarity_weight=1.0,
                            lexicons=[
                                "detect", "analyze", "identify", "protect",
                                "scan", "check", "verify",
                                "remember", "recall", "forget", "learn",
                                "listen", "recognize",
                                "watch", "observe",
                                "route", "classify",
                                "none",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="intent_signals",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="domain",
            similarity_weight=0.35,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="capability_domain",
                            similarity_weight=1.0,
                            lexicons=[
                                "security", "voice", "identity", "memory",
                                "vision", "routing", "general", "none",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="domain_signals",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="semantic",
            similarity_weight=0.25,
            segments=[
                Segment(
                    name="text",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="keywords",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Cognitive extraction — lightweight, no external deps
# ---------------------------------------------------------------------------

# Cognitive action patterns: verb → action
_ACTION_MAP = {
    # Security
    "safe": "detect", "unsafe": "detect", "secure": "protect",
    "inject": "detect", "injection": "detect", "jailbreak": "detect",
    "attack": "detect", "threat": "detect", "malicious": "detect",
    "harmful": "detect", "dangerous": "detect", "block": "protect",
    "filter": "protect", "firewall": "protect", "guard": "protect",
    "scan": "scan", "check": "check", "verify": "verify",
    "validate": "verify", "test": "check", "audit": "analyze",

    # Voice / Identity
    "speaker": "identify", "speaking": "identify", "voice": "identify",
    "who": "identify", "identify": "identify", "recognize": "recognize",
    "listen": "listen", "hear": "listen", "audio": "listen",
    "sound": "listen", "liveness": "verify",

    # Memory
    "remember": "remember", "recall": "recall", "forget": "forget",
    "memorize": "remember", "store": "remember", "save": "remember",
    "know": "recall", "knew": "recall", "learned": "recall",
    "told": "recall",

    # Vision
    "see": "watch", "look": "watch", "image": "observe",
    "picture": "observe", "photo": "observe", "visual": "observe",
    "watch": "watch", "observe": "observe",

    # Code / Search
    "find": "analyze", "search": "analyze", "index": "learn",
    "codebase": "analyze", "files": "analyze", "imports": "analyze",
    "drift": "analyze", "blast": "analyze", "risk": "analyze",
    "related": "analyze", "defined": "analyze", "handles": "analyze",

    # Analysis
    "analyze": "analyze", "classify": "classify", "categorize": "classify",
    "route": "route", "detect": "detect",
}

# Domain patterns: keyword → domain
_DOMAIN_MAP = {
    # Security
    "safe": "security", "unsafe": "security", "injection": "security",
    "jailbreak": "security", "attack": "security", "threat": "security",
    "malicious": "security", "harmful": "security", "firewall": "security",
    "prompt": "security", "exploit": "security", "hack": "security",
    "security": "security", "dangerous": "security", "override": "security",
    "ignore": "security", "disregard": "security",

    # Voice / Identity
    "voice": "voice", "speaker": "voice", "speaking": "voice",
    "audio": "voice", "sound": "voice", "hear": "voice",
    "listen": "voice", "microphone": "voice", "recording": "voice",
    "identity": "identity", "biometric": "identity", "liveness": "identity",
    "who": "identity",

    # Memory
    "remember": "memory", "recall": "memory", "forget": "memory",
    "memory": "memory", "memorize": "memory", "name": "memory",
    "told": "memory", "said": "memory", "mentioned": "memory",

    # Vision
    "image": "vision", "picture": "vision", "photo": "vision",
    "visual": "vision", "see": "vision", "camera": "vision",
    "screenshot": "vision",

    # Code
    "code": "general", "codebase": "general", "file": "general",
    "files": "general", "function": "general", "class": "general",
    "import": "general", "module": "general", "search": "general",
    "index": "general", "drift": "general", "risk": "general",
    "blast": "general", "radius": "general", "repository": "general",

}


def _extract_cognitive(text: str) -> dict:
    """Extract cognitive intent and domain from text.

    Returns dict with keys matching ENCODER_CONFIG roles:
      cognitive_action, intent_signals,
      capability_domain, domain_signals,
      description, keywords
    """
    words = re.findall(r'[a-z]+', text.lower())
    word_set = set(words)
    text_lower = text.lower()

    # Extract cognitive action
    action = "none"
    for word in words:
        if word in _ACTION_MAP:
            action = _ACTION_MAP[word]
            break

    # Extract domain
    domain = "none"
    domain_scores: dict[str, int] = {}
    for word in words:
        if word in _DOMAIN_MAP:
            d = _DOMAIN_MAP[word]
            domain_scores[d] = domain_scores.get(d, 0) + 1
    if domain_scores:
        domain = max(domain_scores, key=domain_scores.get)

    # Build BoW signals
    intent_signals = " ".join(words[:10])
    domain_signals = " ".join(words[:10])
    description = text_lower
    keywords = " ".join(words)

    return {
        "cognitive_action": action,
        "intent_signals": intent_signals,
        "capability_domain": domain,
        "domain_signals": domain_signals,
        "description": description,
        "keywords": keywords,
    }


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert NL text into a Concept dict for cognitive routing.

    Returns a dict with 'name' and 'attributes' matching ENCODER_CONFIG roles.
    """
    features = _extract_cognitive(query)
    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    return {
        "name": f"route_{stable_id:08d}",
        "attributes": features,
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL exemplar → build record
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL exemplar into a record for encoding.

    Expected exemplar format:
      {"text": "...", "capability": "firewall", "cognitive_action": "detect", "capability_domain": "security"}

    Or minimal:
      {"text": "is this prompt safe?", "capability": "firewall"}
    """
    text = entry.get("text", "")

    if "cognitive_action" in entry:
        # Pre-extracted features
        attributes = {
            "cognitive_action": entry["cognitive_action"],
            "intent_signals": entry.get("intent_signals", text),
            "capability_domain": entry.get("capability_domain", "none"),
            "domain_signals": entry.get("domain_signals", text),
            "description": text,
            "keywords": entry.get("keywords", text),
        }
    else:
        # Auto-extract
        attributes = _extract_cognitive(text)

    return {
        "concept_text": entry.get("capability", "unknown"),
        "attributes": attributes,
        "metadata": {
            "capability": entry.get("capability", "unknown"),
            "text": text[:200],
        },
    }
