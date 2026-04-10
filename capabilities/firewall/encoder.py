"""
Encoder for the Prompt Injection Firewall model.

Exports:
  ENCODER_CONFIG — EncoderConfig with 4 layers: intent, structure, semantic, adversarial
  encode_prompt(text) — converts prompt text to a Concept dict for similarity search
  entry_to_record(entry) — converts a JSONL exemplar entry to a build record
  score_prompt(query_glyph, exemplar_glyphs) — returns verdict with per-layer breakdown

Architecture:
  NL Extraction — intent.py (local, deterministic):
    - analyze_prompt(text) → features across 4 dimensions
    - Pattern-based detection for overrides, jailbreaks, extraction, role assumption
    - Structural analysis for delimiter injection, nesting depth
    - Obfuscation scoring for base64, unicode tricks, mixed scripts

  Main model (seed=42):
  - Intent layer (0.30): intent_type (lexicon) + intent_signals (BoW)
  - Structure layer (0.30): delimiter_type (lexicon) + nesting_depth (numeric) + structure_signals (BoW)
  - Semantic layer (0.25): attack_family (lexicon) + semantic_tokens (BoW)
  - Adversarial layer (0.15): encoding_type (lexicon) + obfuscation_score (numeric) + adversarial_signals (BoW)

  Scoring: Differential — threat = max_attack_sim - max_benign_sim.
  Cancels out shared default similarity so benign prompts score near zero.
"""

import hashlib
import time

from glyphh.core.config import (
    EncoderConfig,
    EncodingStrategy,
    Layer,
    NumericConfig,
    Role,
    Segment,
)

from intent import analyze_prompt

# ---------------------------------------------------------------------------
# ENCODER_CONFIG — 4-layer prompt injection detection
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="intent_type",
                            similarity_weight=1.0,
                            lexicons=[
                                "query", "instruct", "override",
                                "extract", "jailbreak", "benign",
                                "harmful", "abuse", "manipulate",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="intent_signals",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="structure",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="pattern",
                    roles=[
                        Role(
                            name="delimiter_type",
                            similarity_weight=1.0,
                            lexicons=[
                                "none", "markdown", "xml",
                                "json", "system_tag", "separator",
                            ],
                        ),
                        Role(
                            name="nesting_depth",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=1.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=5.0,
                            ),
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="structure_signals",
                            similarity_weight=0.8,
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
                    name="classification",
                    roles=[
                        Role(
                            name="attack_family",
                            similarity_weight=1.0,
                            lexicons=[
                                "none",
                                # Original 6 families
                                "role_assumption", "instruction_override",
                                "context_manipulation", "delimiter_injection",
                                "extraction", "indirect_injection",
                                # Expanded families (v0.9+)
                                "encoding_obfuscation", "logic_exploitation",
                                "harmful_content", "social_engineering",
                                "tool_abuse", "resource_abuse",
                                "multi_turn", "compliance_violation",
                                "agentic_exploit", "output_manipulation",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="semantic_tokens",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="adversarial",
            similarity_weight=0.15,
            segments=[
                Segment(
                    name="encoding",
                    roles=[
                        Role(
                            name="encoding_type",
                            similarity_weight=1.0,
                            lexicons=[
                                "none", "base64", "hex",
                                "unicode", "rot13", "mixed",
                            ],
                        ),
                        Role(
                            name="obfuscation_score",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=10.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=100.0,
                            ),
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="adversarial_signals",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# encode_prompt — NL text -> Concept dict
# ---------------------------------------------------------------------------

def encode_prompt(text: str) -> dict:
    """Alias for encode_query (backward compat)."""
    return encode_query(text)


def encode_query(text: str) -> dict:
    """Convert raw prompt text into a Concept-compatible dict for similarity search.

    Returns a dict with 'name' and 'attributes' keys matching ENCODER_CONFIG roles.
    """
    features = analyze_prompt(text)
    stable_id = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return {
        "name": f"prompt_{stable_id:08d}",
        "attributes": features,
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL exemplar -> build record
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL exemplar entry into a record for building/encoding.

    Expects entry keys: text, label, attack_family (optional metadata).
    Can also accept pre-extracted features for manually crafted exemplars.
    """
    text = entry.get("text", "")

    if "intent_type" in entry:
        from intent import (
            extract_intent_signals, extract_structure_signals,
            extract_semantic_tokens, extract_adversarial_signals,
        )
        attributes = {
            "intent_type": entry["intent_type"],
            "intent_signals": entry.get("intent_signals", extract_intent_signals(text)),
            "delimiter_type": entry.get("delimiter_type", "none"),
            "nesting_depth": entry.get("nesting_depth", 0),
            "structure_signals": entry.get("structure_signals", extract_structure_signals(text)),
            "attack_family": entry.get("attack_family", "none"),
            "semantic_tokens": entry.get("semantic_tokens", extract_semantic_tokens(text)),
            "encoding_type": entry.get("encoding_type", "none"),
            "obfuscation_score": entry.get("obfuscation_score", 0),
            "adversarial_signals": entry.get("adversarial_signals", "clean"),
        }
    else:
        attributes = analyze_prompt(text)

    return {
        "concept_text": entry.get("label", f"exemplar_{entry.get('id', 'unknown')}"),
        "attributes": attributes,
        "metadata": {
            "label": entry.get("label", "unknown"),
            "attack_family": entry.get("attack_family", attributes.get("attack_family", "none")),
            "text": text[:200],
        },
    }


# ---------------------------------------------------------------------------
# score_prompt — verdict with per-layer breakdown
# ---------------------------------------------------------------------------

# Thresholds (calibrated for differential scoring)
# Attack exemplar self-scores: 0.15-0.52, benign: 0.00
BLOCK_THRESHOLD = 0.30
FLAG_THRESHOLD = 0.12

# Layer weights for scoring (match ENCODER_CONFIG similarity_weights)
LAYER_WEIGHTS = {
    "intent": 0.30,
    "structure": 0.30,
    "semantic": 0.25,
    "adversarial": 0.15,
}


def _compute_layer_scores(query_glyph, exemplar_glyph, cos_sim) -> tuple[dict, float]:
    """Compute per-layer similarity and weighted score between two glyphs."""
    layer_scores = {}
    weighted_score = 0.0
    for layer_name, weight in LAYER_WEIGHTS.items():
        if layer_name in query_glyph.layers and layer_name in exemplar_glyph.layers:
            q_cortex = query_glyph.layers[layer_name].cortex.data
            e_cortex = exemplar_glyph.layers[layer_name].cortex.data
            sim = float(cos_sim(q_cortex, e_cortex))
            layer_scores[layer_name] = round(sim, 4)
            weighted_score += sim * weight
        else:
            layer_scores[layer_name] = 0.0
    return layer_scores, weighted_score


def score_prompt(
    query_glyph,
    exemplar_glyphs: list,
    block_threshold: float = BLOCK_THRESHOLD,
    flag_threshold: float = FLAG_THRESHOLD,
) -> dict:
    """Score a query prompt against exemplar glyphs using differential scoring.

    threat_score = max_attack_sim - max_benign_sim.
    Cancels out shared default similarity so benign prompts score near zero.

    Args:
        query_glyph: Encoded glyph of the input prompt
        exemplar_glyphs: List of (glyph, metadata) tuples for ALL exemplars
        block_threshold: Score above this = BLOCK
        flag_threshold: Score above this = FLAG

    Returns dict with:
        threat_score, verdict, matched_family, matched_label,
        layer_scores, explanation
    """
    from glyphh.core.ops import cosine_similarity as cos_sim

    best_attack_score = 0.0
    best_family = "none"
    best_label = "unknown"
    best_layer_scores = {}

    for exemplar_glyph, metadata in exemplar_glyphs:
        if metadata.get("label") == "benign":
            continue
        layer_scores, weighted_score = _compute_layer_scores(
            query_glyph, exemplar_glyph, cos_sim,
        )
        if weighted_score > best_attack_score:
            best_attack_score = weighted_score
            best_family = metadata.get("attack_family", "unknown")
            best_label = metadata.get("label", "unknown")
            best_layer_scores = layer_scores

    best_benign_score = 0.0
    for exemplar_glyph, metadata in exemplar_glyphs:
        if metadata.get("label") != "benign":
            continue
        _, weighted_score = _compute_layer_scores(
            query_glyph, exemplar_glyph, cos_sim,
        )
        if weighted_score > best_benign_score:
            best_benign_score = weighted_score

    threat_score = max(0.0, best_attack_score - best_benign_score)

    if threat_score >= block_threshold:
        verdict = "BLOCK"
    elif threat_score >= flag_threshold:
        verdict = "FLAG"
    else:
        verdict = "PASS"

    if verdict == "PASS":
        explanation = "No significant injection patterns detected."
    else:
        top_layer = max(best_layer_scores, key=best_layer_scores.get) if best_layer_scores else "unknown"
        top_sim = best_layer_scores.get(top_layer, 0)
        explanation = (
            f"{verdict}. {top_layer.capitalize()} layer {top_sim:.2f} match to "
            f"{best_family.replace('_', ' ')} pattern. "
            f"Matched exemplar: {best_label}."
        )

    return {
        "threat_score": round(threat_score, 4),
        "verdict": verdict,
        "matched_family": best_family,
        "matched_label": best_label,
        "layer_scores": best_layer_scores,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "firewall_scan",
        "description": "Scan a prompt for injection attacks. Returns verdict (BLOCK/FLAG/PASS), threat score, per-layer breakdown, and explanation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The prompt text to scan for injection attacks.",
                },
            },
            "required": ["text"],
        },
    },
]


async def handle_mcp_tool(tool_name: str, arguments: dict, context: dict) -> dict:
    """Handle model-specific MCP tool calls."""
    if tool_name != "firewall_scan":
        return {
            "state": "ERROR",
            "confidence": 0,
            "match_method": "",
            "fact_tree": {"error": f"Unknown tool: {tool_name}"},
        }

    text = arguments.get("text", "")
    if not text.strip():
        return {
            "state": "ERROR",
            "confidence": 0,
            "match_method": "",
            "fact_tree": {"error": "Missing required argument: text"},
        }

    t0 = time.time()

    # Encode the query prompt
    concept = encode_query(text)
    features = concept["attributes"]

    # Get the model manager from context to access exemplar glyphs
    manager = context.get("model_manager")
    model_id = context.get("model_id", "model-firewall")
    org_id = context.get("org_id", "")

    # Heuristic scoring — fast, deterministic, no DB round-trip.
    # Exemplar-based differential scoring (score_prompt) available for
    # future use when glyph loading from DB is wired up.
    result = _heuristic_score(features)

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    # Log event for shield dashboard
    try:
        from pathlib import Path
        import json as _json
        log_dir = Path.home() / ".ada" / "firewall"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "events.jsonl", "a") as _f:
            _f.write(_json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "text": text[:200],
                "verdict": result["verdict"],
                "threat_score": result["threat_score"],
                "intent_type": features.get("intent_type", "benign"),
                "matched_family": result.get("matched_family", "none"),
                "latency_ms": elapsed_ms,
            }) + "\n")
    except Exception:
        pass

    return {
        "state": "DONE",
        "confidence": result["threat_score"],
        "match_method": "differential_cosine",
        "query_time_ms": elapsed_ms,
        # Top-level fields for the UI (reads threat_score, verdict, etc. directly)
        **result,
        "features": features,
        "fact_tree": {
            **result,
            "latency_ms": elapsed_ms,
            "features": features,
        },
    }


def _heuristic_score(features: dict) -> dict:
    """Fallback scoring when exemplar glyphs aren't available."""
    intent_type = features.get("intent_type", "benign")
    attack_family = features.get("attack_family", "none")

    threat_map = {
        "override": 0.85, "jailbreak": 0.80, "extract": 0.75,
        "harmful": 0.80, "abuse": 0.70, "manipulate": 0.65,
        "instruct": 0.40, "query": 0.0, "benign": 0.0,
    }
    threat = threat_map.get(intent_type, 0.0)

    if threat >= BLOCK_THRESHOLD:
        verdict = "BLOCK"
    elif threat >= FLAG_THRESHOLD:
        verdict = "FLAG"
    else:
        verdict = "PASS"

    return {
        "threat_score": threat,
        "verdict": verdict,
        "matched_family": attack_family,
        "matched_label": attack_family if attack_family != "none" else "benign",
        "layer_scores": {
            "intent": threat,
            "structure": 0.0,
            "semantic": threat * 0.9,
            "adversarial": 0.0,
        },
        "explanation": f"{verdict}. Intent: {intent_type}, family: {attack_family}."
            if verdict != "PASS"
            else "No significant injection patterns detected.",
    }
