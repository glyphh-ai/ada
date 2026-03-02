"""
glyphh.intent — Generic NL intent extraction for Glyphh models.

Provides rule-based NL parsing (action, target, domain, keywords) with HDC
similarity fallback. Works with any downstream model — toolrouter, BFCL,
or custom domain models.

Quick start:
    from glyphh.intent import IntentExtractor

    extractor = IntentExtractor()
    result = extractor.extract("Send a message to #general on Slack")
    # {"action": "send", "target": "channel", "domain": "messaging", "keywords": "..."}

    # With domain packs:
    extractor = IntentExtractor(packs=["filesystem"])
    result = extractor.extract("List files in the home directory")
    # {"action": "ls", "target": "directory", "domain": "filesystem", "keywords": "..."}

Available packs: filesystem, trading, travel, social, math, vehicle
"""

from glyphh.intent.extractor import (
    VERB_CONFIG,
    NOUN_CONFIG,
    IntentExtractor,
    get_extractor,
    extract_intent,
    extract_action,
    extract_target,
    infer_domain,
)

__all__ = [
    "VERB_CONFIG",
    "NOUN_CONFIG",
    "IntentExtractor",
    "get_extractor",
    "extract_intent",
    "extract_action",
    "extract_target",
    "infer_domain",
]
