"""
Encoder for the Ada Voice model — voice identity and state encoding.

Exports:
    ENCODER_CONFIG      — Four-layer HDC encoder (identity + emotional + cognitive + cadence)
    encode_features     — Audio feature dict → Concept dict for encoding
    entry_to_record     — Enrollment record → encodable record + metadata
    IDENTITY_ROLE_WEIGHTS — Per-role weights for identity scoring
    score_identity_roles — Role-level identity scoring (speaker recognition)
    score_state_drift   — Per-layer state comparison against baseline
    MCP_TOOLS           — Model-specific MCP tool schemas
    handle_mcp_tool     — Async handler for voice_enroll / voice_identify

Architecture:
    Identity layer (0.55):       MFCCs 1-13, deltas 1-4, formants F1-F3, HNR — THERMOMETER
        → Physical vocal tract geometry. 13 MFCCs capture fine resonance detail that
        → is nearly impossible to consciously control. Deltas add articulatory dynamics.
        → Formants demoted (LPC noisy). Wide bins absorb librosa extraction noise.

    Emotional state layer (0.15): F0 mean/std/p20/p80, loudness mean/std, spectral slope/flux — THERMOMETER
        → Prosodic contour. Changes per utterance. Encodes arousal, valence, affect.
        → F0 percentiles detect pitch range compression/exaggeration (deception marker).
        → Spectral flux detects monotone speech (overcorrection marker).

    Cognitive load layer (0.10):  Jitter, shimmer, voiced std, pause std, alpha ratio,
                                   hammarberg index, F1 bandwidth, MFCC1 CV — THERMOMETER
        → Voice stability markers. Stress, fatigue, deception, cognitive strain.
        → Alpha ratio + hammarberg = laryngeal tension (involuntary, can't be faked).
        → F1 bandwidth widens under stress. MFCC1 CV = within-utterance instability.

    Cadence layer (0.20):        Loudness peaks/sec, voiced/unvoiced segment lengths — THERMOMETER
        → Speech rhythm and phrasing. Cognitive style, personality signature.

    Dimension: 2000 (matches runtime pgvector HNSW limit, same as pipedream)
    Temporal: G-number (speaker_id) as key_part with auto timestamps for drift tracking.

    Matching pipeline (follows SDK patterns):
        1. pgvector cosine on identity layer cortex → top-5 candidates (rough filter)
        2. Fetch per-role vectors from glyph_vectors for each candidate
        3. SDK cosine_similarity role-by-role with weights → final score (Pattern A)
        This is how toolrouter/pipedream work — pgvector is the index, SDK is the scorer.
"""

import base64
import hashlib
import io
import json
import logging
import math
import tempfile
import time

from glyphh.core.config import (
    EncoderConfig,
    EncodingStrategy,
    Layer,
    NumericConfig,
    Role,
    Segment,
    TemporalConfig,
)
from glyphh.core.ops import cosine_similarity

from features import extract_features as _extract_opensmile
from features import extract_features_librosa, compute_liveness_score


# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    include_temporal=True,
    temporal_config=TemporalConfig(signal_type="auto"),
    temporal_source="auto",
    apply_weights_during_encoding=False,
    layers=[
        # --- Identity layer: vocal tract geometry (speaker fingerprint) ---
        Layer(
            name="identity",
            similarity_weight=0.55,
            segments=[
                Segment(
                    name="vocal_tract",
                    roles=[
                        Role(
                            name="mfcc_1",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=8.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-20.0,
                                max_value=60.0,
                            ),
                        ),
                        Role(
                            name="mfcc_2",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=6.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-30.0,
                                max_value=30.0,
                            ),
                        ),
                        Role(
                            name="mfcc_3",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=5.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-20.0,
                                max_value=20.0,
                            ),
                        ),
                        Role(
                            name="mfcc_4",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=4.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-15.0,
                                max_value=15.0,
                            ),
                        ),
                        Role(
                            name="mfcc_5",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=3.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-15.0,
                                max_value=15.0,
                            ),
                        ),
                        Role(
                            name="mfcc_6",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=3.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-15.0,
                                max_value=15.0,
                            ),
                        ),
                        Role(
                            name="mfcc_7",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=2.5,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-12.0,
                                max_value=12.0,
                            ),
                        ),
                        Role(
                            name="mfcc_8",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=2.5,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-12.0,
                                max_value=12.0,
                            ),
                        ),
                        Role(
                            name="mfcc_9",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-10.0,
                                max_value=10.0,
                            ),
                        ),
                        Role(
                            name="mfcc_10",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-10.0,
                                max_value=10.0,
                            ),
                        ),
                        Role(
                            name="mfcc_11",
                            similarity_weight=0.5,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-10.0,
                                max_value=10.0,
                            ),
                        ),
                        Role(
                            name="mfcc_12",
                            similarity_weight=0.5,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-10.0,
                                max_value=10.0,
                            ),
                        ),
                        Role(
                            name="mfcc_13",
                            similarity_weight=0.5,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-10.0,
                                max_value=10.0,
                            ),
                        ),
                        # MFCC deltas — articulatory dynamics, speaker-specific
                        Role(
                            name="mfcc_delta_1",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=0.5,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-5.0,
                                max_value=5.0,
                            ),
                        ),
                        Role(
                            name="mfcc_delta_2",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=0.5,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-5.0,
                                max_value=5.0,
                            ),
                        ),
                        Role(
                            name="mfcc_delta_3",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=0.4,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-4.0,
                                max_value=4.0,
                            ),
                        ),
                        Role(
                            name="mfcc_delta_4",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=0.4,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-4.0,
                                max_value=4.0,
                            ),
                        ),
                        # Formants via LPC — noisy, wide bins, low weight
                        Role(
                            name="formant_f1",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=80.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=200.0,
                                max_value=1000.0,
                            ),
                        ),
                        Role(
                            name="formant_f2",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=200.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=500.0,
                                max_value=3000.0,
                            ),
                        ),
                        Role(
                            name="formant_f3",
                            similarity_weight=0.5,
                            numeric_config=NumericConfig(
                                bin_width=250.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=1500.0,
                                max_value=4000.0,
                            ),
                        ),
                        Role(
                            name="hnr",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=4.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=40.0,
                            ),
                        ),
                    ],
                ),
            ],
        ),
        # --- Emotional state layer: prosodic contour ---
        Layer(
            name="emotional_state",
            similarity_weight=0.15,
            segments=[
                Segment(
                    name="prosody",
                    roles=[
                        Role(
                            name="f0_mean",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=1.0,
                                max_value=40.0,
                            ),
                        ),
                        Role(
                            name="f0_std",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                        Role(
                            name="loudness_mean",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=0.1,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=2.0,
                            ),
                        ),
                        Role(
                            name="loudness_std",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                        Role(
                            name="spectral_slope",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=0.005,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-0.05,
                                max_value=0.05,
                            ),
                        ),
                        # F0 pitch range — liars compress or exaggerate
                        Role(
                            name="f0_p20",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=1.0,
                                max_value=40.0,
                            ),
                        ),
                        Role(
                            name="f0_p80",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=1.0,
                                max_value=40.0,
                            ),
                        ),
                        # Spectral flux — rate of spectral change, monotone = low flux
                        Role(
                            name="spectral_flux",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=0.01,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=0.5,
                            ),
                        ),
                    ],
                ),
            ],
        ),
        # --- Cognitive load layer: voice stability markers ---
        Layer(
            name="cognitive_load",
            similarity_weight=0.10,
            segments=[
                Segment(
                    name="stability",
                    roles=[
                        Role(
                            name="jitter",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=0.005,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=0.10,
                            ),
                        ),
                        Role(
                            name="shimmer",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=0.15,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=3.0,
                            ),
                        ),
                        Role(
                            name="voiced_std",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                        Role(
                            name="pause_std",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                        # Alpha ratio — energy below/above 1kHz, laryngeal tension
                        Role(
                            name="alpha_ratio",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=1.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-10.0,
                                max_value=10.0,
                            ),
                        ),
                        # Hammarberg index — spectral peak ratio, throat tension
                        Role(
                            name="hammarberg_index",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=2.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=40.0,
                            ),
                        ),
                        # F1 bandwidth — widens under stress, involuntary
                        Role(
                            name="formant_f1_bw",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=30.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=50.0,
                                max_value=500.0,
                            ),
                        ),
                        # MFCC1 coefficient of variation — within-utterance instability
                        Role(
                            name="mfcc_1_cv",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=0.1,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=2.0,
                            ),
                        ),
                    ],
                ),
            ],
        ),
        # --- Cadence layer: speech rhythm (cognitive style) ---
        Layer(
            name="cadence",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="rhythm",
                    roles=[
                        Role(
                            name="loudness_peaks_per_sec",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=0.5,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=8.0,
                            ),
                        ),
                        Role(
                            name="voiced_segment_mean",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                        Role(
                            name="unvoiced_segment_mean",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                        Role(
                            name="voiced_segments_per_sec",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=0.5,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=10.0,
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Identity role weights — used for Pattern A role-level scoring
# ---------------------------------------------------------------------------

# Build from config so weights stay in sync
IDENTITY_ROLE_WEIGHTS = {}
ALL_ROLE_WEIGHTS = {}  # {layer.segment.role: (layer_weight * role_weight)}
for _layer in ENCODER_CONFIG.layers:
    for _seg in _layer.segments:
        for _role in _seg.roles:
            path = f"{_layer.name}.{_seg.name}.{_role.name}"
            combined_weight = _layer.similarity_weight * _role.similarity_weight
            ALL_ROLE_WEIGHTS[path] = combined_weight
            if _layer.name == "identity":
                IDENTITY_ROLE_WEIGHTS[_role.name] = _role.similarity_weight


# ---------------------------------------------------------------------------
# encode_features — audio feature dict → Concept-compatible dict
# ---------------------------------------------------------------------------

def encode_features(features: dict, speaker_id: str = "") -> dict:
    """Convert extracted audio features to a Concept-compatible dict.

    Args:
        features: Dict from features.extract_features() — role_name → float.
        speaker_id: G-number (e.g. "G-00001") or empty for new enrollment.

    Returns:
        Dict with 'name' and 'attributes' ready for Encoder.encode().
    """
    if speaker_id:
        name = speaker_id
    else:
        fp = "|".join(f"{k}:{v:.4f}" for k, v in sorted(features.items()))
        h = int(hashlib.md5(fp.encode()).hexdigest()[:8], 16)
        name = f"voice_{h:08d}"

    # Sanitize NaN/Inf — these fall through to symbolic encoding,
    # making all speakers look identical.
    clean = {}
    for k, v in features.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = 0.0
        else:
            clean[k] = v

    return {"name": name, "attributes": clean}


# ---------------------------------------------------------------------------
# entry_to_record — enrollment record → encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert an enrollment/exemplar record to an encodable record."""
    speaker_id = entry.get("speaker_id", "")
    attrs = {}
    feature_keys = [
        "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
        "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9",
        "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13",
        "mfcc_delta_1", "mfcc_delta_2", "mfcc_delta_3", "mfcc_delta_4",
        "formant_f1", "formant_f2", "formant_f3", "hnr",
        "f0_mean", "f0_std", "loudness_mean", "loudness_std", "spectral_slope",
        "f0_p20", "f0_p80", "spectral_flux",
        "jitter", "shimmer", "voiced_std", "pause_std",
        "alpha_ratio", "hammarberg_index", "formant_f1_bw", "mfcc_1_cv",
        "loudness_peaks_per_sec", "voiced_segment_mean",
        "unvoiced_segment_mean", "voiced_segments_per_sec",
    ]
    for key in feature_keys:
        attrs[key] = entry.get(key, 0.0)

    return {
        "concept_text": speaker_id or "unknown",
        "attributes": attrs,
        "metadata": {
            "speaker_id": speaker_id,
            "recording_type": entry.get("recording_type", "authentic"),
            "language": entry.get("language", ""),
            "session_id": entry.get("session_id", ""),
        },
    }


# ---------------------------------------------------------------------------
# Scoring functions — SDK Pattern A (role-level weighted average)
# ---------------------------------------------------------------------------

def score_identity_roles(query_glyph, candidate_glyph) -> float:
    """Score identity match using per-role cosine with weights.

    This is the primary matching function. Each MFCC, formant, HNR role
    is compared individually via SDK cosine_similarity, then weighted.
    A faker who matches MFCCs 1-2 but misses 5-13 will score low on
    those roles, dragging the weighted average down.

    Follows SDK Pattern A (flat role-level weighted average).
    """
    q_roles = {}
    c_roles = {}

    # Collect identity layer roles from both glyphs
    q_identity = query_glyph.layers.get("identity")
    c_identity = candidate_glyph.layers.get("identity")
    if not q_identity or not c_identity:
        return 0.0

    for seg in q_identity.segments.values():
        q_roles.update(seg.roles)
    for seg in c_identity.segments.values():
        c_roles.update(seg.roles)

    weighted_sum = 0.0
    weight_total = 0.0
    for role_name, weight in IDENTITY_ROLE_WEIGHTS.items():
        if role_name in q_roles and role_name in c_roles:
            sim = cosine_similarity(q_roles[role_name].data, c_roles[role_name].data)
            weighted_sum += sim * weight
            weight_total += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.0


def score_state_drift(current_glyph, baseline_glyph) -> dict:
    """Compare current voice state against personal baseline.

    Returns per-layer similarity scores. Low similarity in a state layer
    means the person's voice has shifted from their baseline.
    """
    result = {}
    for layer_name in ("identity", "emotional_state", "cognitive_load", "cadence"):
        if layer_name in current_glyph.layers and layer_name in baseline_glyph.layers:
            result[layer_name] = cosine_similarity(
                current_glyph.layers[layer_name].cortex.data,
                baseline_glyph.layers[layer_name].cortex.data,
            )
        else:
            result[layer_name] = 0.0
    return result


def score_authenticity(real_glyph, faked_glyph) -> dict:
    """Score the authenticity delta between a real and faked recording."""
    drift = score_state_drift(faked_glyph, real_glyph)
    identity_match = drift["identity"]
    state_layers = ["emotional_state", "cognitive_load", "cadence"]
    state_sims = [drift[l] for l in state_layers]
    avg_state_sim = sum(state_sims) / len(state_sims) if state_sims else 0.0
    total_shift = 1.0 - avg_state_sim
    return {
        "identity_match": identity_match,
        "total_shift": total_shift,
        "per_layer": drift,
    }


# ---------------------------------------------------------------------------
# MCP Tools — voice_enroll, voice_identify
# ---------------------------------------------------------------------------

log = logging.getLogger("ada.voice")

MCP_TOOLS = [
    {
        "name": "voice_enroll",
        "description": (
            "Enroll a new voice identity. Accepts base64-encoded WAV audio, "
            "extracts vocal features, encodes as an HDC glyph, and assigns a "
            "G-number. Pass g_number and phrase_number for multi-phrase enrollment "
            "(3 phrases required for complete enrollment)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_base64": {
                    "type": "string",
                    "description": "Base64-encoded WAV audio data",
                },
                "speaker_name": {
                    "type": "string",
                    "description": "Speaker's stated name (stored in metadata only)",
                },
                "g_number": {
                    "type": "string",
                    "description": "Existing G-number for additional enrollment phrases",
                },
                "phrase_number": {
                    "type": "integer",
                    "description": "Which enrollment phrase (1, 2, or 3)",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for temporary enrollments (e.g. demo sessions)",
                },
                "recording_label": {
                    "type": "string",
                    "description": "Label for this recording (e.g. 'baseline', 'statement_a', 'statement_b')",
                },
                "eye_features": {
                    "type": "object",
                    "description": (
                        "Eye tracking features from MediaPipe FaceMesh. "
                        "Optional — enriches lie detection when video is available."
                    ),
                    "properties": {
                        "pupil_dilation_rel": {
                            "type": "number",
                            "description": "Relative pupil dilation vs baseline (ratio, ~1.0 = no change)",
                        },
                        "saccade_rate": {
                            "type": "number",
                            "description": "Saccades per second (rapid eye movements)",
                        },
                        "saccade_velocity_mean": {
                            "type": "number",
                            "description": "Mean saccade velocity in pixels/frame",
                        },
                        "microsaccade_rate": {
                            "type": "number",
                            "description": "Microsaccades per second (fixational instability)",
                        },
                        "blink_rate": {
                            "type": "number",
                            "description": "Blinks per minute",
                        },
                        "blink_interval_std": {
                            "type": "number",
                            "description": "Standard deviation of inter-blink intervals (seconds)",
                        },
                        "fixation_stability": {
                            "type": "number",
                            "description": "Gaze stability 0-1 (1 = perfectly steady)",
                        },
                        "pupil_oscillation": {
                            "type": "number",
                            "description": "Hippus frequency — involuntary pupil oscillation (Hz)",
                        },
                        "eye_quality": {
                            "type": "number",
                            "description": "Tracking quality 0-1 (fraction of frames with good iris detection)",
                        },
                    },
                },
            },
            "required": ["audio_base64"],
        },
    },
    {
        "name": "voice_identify",
        "description": (
            "Identify a speaker from voice audio. Accepts base64-encoded WAV, "
            "extracts features, and matches against enrolled voice glyphs. "
            "Returns match confidence, G-number, and state drift analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_base64": {
                    "type": "string",
                    "description": "Base64-encoded WAV audio data",
                },
            },
            "required": ["audio_base64"],
        },
    },
    {
        "name": "voice_list",
        "description": (
            "List enrolled voice identities. Returns count and recent G-numbers. "
            "Use limit to control how many entries are returned (default 100)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default 100)",
                },
            },
        },
    },
    {
        "name": "voice_compare",
        "description": (
            "Compare two statement recordings against a baseline recording. "
            "All three must already be enrolled. Returns per-layer divergence "
            "scores and picks which statement diverges more from baseline on "
            "emotional/cognitive layers (likely the lie). Optionally accepts "
            "eye_features for each statement — when available, adds a 6th "
            "ocular deception signal weighted at 0.40."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "baseline_g": {
                    "type": "string",
                    "description": "G-number of the baseline recording",
                },
                "statement_a_g": {
                    "type": "string",
                    "description": "G-number of statement A",
                },
                "statement_b_g": {
                    "type": "string",
                    "description": "G-number of statement B",
                },
                "statement_a_eye": {
                    "type": "object",
                    "description": "Eye features for statement A (optional, from client-side MediaPipe)",
                },
                "statement_b_eye": {
                    "type": "object",
                    "description": "Eye features for statement B (optional, from client-side MediaPipe)",
                },
                "baseline_eye": {
                    "type": "object",
                    "description": "Eye features for baseline recording (optional)",
                },
            },
            "required": ["baseline_g", "statement_a_g", "statement_b_g"],
        },
    },
    {
        "name": "voice_cleanup",
        "description": (
            "Delete all enrolled glyphs for a given session ID. "
            "Used to clean up temporary demo data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID whose glyphs should be deleted",
                },
            },
            "required": ["session_id"],
        },
    },
]


def _cortex_to_position(embedding, spread: float = 8.0) -> list:
    """Derive 3D position from cortex embedding via chunk-means.

    Splits the vector into 3 equal chunks and takes the mean of each.
    Similar cortices produce similar means → natural clustering in 3D.
    """
    import numpy as np

    arr = np.array(embedding, dtype=np.float32)
    n = len(arr)
    third = n // 3
    x = float(np.mean(arr[:third])) * spread
    y = float(np.mean(arr[third:2 * third])) * spread
    z = float(np.mean(arr[2 * third:])) * spread
    return [round(x, 4), round(y, 4), round(z, 4)]


async def _handle_list(arguments: dict, context: dict) -> dict:
    """List enrolled voices — count + recent G-numbers with 3D positions."""
    session_factory = context["session_factory"]
    org_id = context["org_id"]
    model_id = context["model_id"]
    limit = min(arguments.get("limit", 100), 1000)

    async with session_factory() as session:
        from sqlalchemy import text

        count_row = await session.execute(
            text("SELECT COUNT(*) FROM glyphs WHERE org_id = :org AND model_id = :model"),
            {"org": org_id, "model": model_id},
        )
        total = count_row.scalar() or 0

        rows = await session.execute(
            text("""
                SELECT concept_text, metadata, embedding, created_at
                FROM glyphs
                WHERE org_id = :org AND model_id = :model
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            {"org": org_id, "model": model_id, "lim": limit},
        )
        entries = rows.fetchall()

    glyphs = []
    for row in entries:
        concept = row.concept_text or ""
        g_number = concept.split("@")[0] if "@" in concept else concept
        meta = row.metadata or {}

        # Derive 3D position from cortex embedding
        pos = [0.0, 0.0, 0.0]
        if row.embedding is not None:
            try:
                emb = row.embedding
                if isinstance(emb, str):
                    emb = json.loads(emb)
                pos = _cortex_to_position(emb)
            except Exception:
                pass

        glyphs.append({
            "g_number": g_number,
            "speaker_name": meta.get("speaker_name", ""),
            "record_type": meta.get("record_type", "enrollment"),
            "position": pos,
        })

    return {
        "state": "DONE",
        "confidence": 1.0,
        "match_method": "list",
        "fact_tree": {
            "total": total,
            "returned": len(glyphs),
            "glyphs": glyphs,
        },
    }


def _check_audio_energy(wav_path: str, min_rms: float = 0.015) -> bool:
    """Check if WAV file contains enough energy to be speech (not just noise).

    Returns True if RMS exceeds min_rms. Typical speech is 0.02–0.5,
    ambient mic noise is 0.001–0.01.
    """
    try:
        import numpy as np
        import soundfile as sf
        samples, _ = sf.read(wav_path, dtype="float32")
        if len(samples) == 0:
            return False
        # If stereo, take first channel
        if samples.ndim > 1:
            samples = samples[:, 0]
        rms = float(np.sqrt(np.mean(samples ** 2)))
        log.info("Audio RMS: %.6f (threshold: %.4f)", rms, min_rms)
        return rms >= min_rms
    except Exception as exc:
        log.warning("Could not check audio energy: %s", exc)
        return True  # fail open — let feature validation catch it


def _decode_audio_to_file(audio_b64: str) -> str:
    """Decode base64 audio to a WAV file, converting from WebM/Opus if needed."""
    import os
    import subprocess

    audio_bytes = base64.b64decode(audio_b64)

    is_wav = (len(audio_bytes) > 12
              and audio_bytes[:4] == b"RIFF"
              and audio_bytes[8:12] == b"WAVE")

    if is_wav:
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(audio_bytes)
        wav.close()
        log.info("Audio is WAV (%d bytes), skipping ffmpeg", len(audio_bytes))
        return wav.name

    raw = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    raw.write(audio_bytes)
    raw.close()

    wav_path = raw.name.replace(".webm", ".wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.warning("ffmpeg failed (rc=%d): %s", result.returncode, result.stderr[-500:] if result.stderr else "")
            raise subprocess.CalledProcessError(result.returncode, "ffmpeg")
        os.unlink(raw.name)
        return wav_path
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("ffmpeg conversion failed (%s), trying raw file", exc)
        os.rename(raw.name, wav_path)
        return wav_path


def _has_valid_features(features: dict) -> bool:
    """Check if extracted features contain real signal (not all NaN/zero).

    Requires at least 6 of 8 core MFCCs to be non-zero AND at least one
    cadence/prosody feature to be non-zero (proves actual speech occurred).
    Without this, silence or ambient noise produces default-bin vectors
    that match everyone.
    """
    identity_keys = ["mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8"]
    valid_mfcc = 0
    for k in identity_keys:
        v = features.get(k, 0.0)
        if isinstance(v, float) and math.isfinite(v) and v != 0.0:
            valid_mfcc += 1

    # Need speech evidence — voiced segments proves actual phonation occurred.
    # Ambient noise produces MFCCs but no voiced segments or meaningful F0.
    vps = features.get("voiced_segments_per_sec", 0.0)
    f0 = features.get("f0_mean", 0.0)
    loudness = features.get("loudness_mean", 0.0)

    has_speech = (
        (isinstance(vps, (int, float)) and math.isfinite(vps) and vps > 0.5)
        or (isinstance(f0, (int, float)) and math.isfinite(f0) and f0 > 5.0)
    )
    has_energy = isinstance(loudness, (int, float)) and math.isfinite(loudness) and loudness > 0.01

    return valid_mfcc >= 6 and has_speech and has_energy


def _extract_features_safe(audio_path: str) -> dict:
    """Extract features using openSMILE, falling back to librosa if NaN."""
    try:
        features = _extract_opensmile(audio_path)
        if _has_valid_features(features):
            return features
        log.warning("openSMILE returned NaN/empty features, falling back to librosa")
    except (ImportError, Exception) as exc:
        log.warning("openSMILE unavailable (%s), falling back to librosa", exc)
    return extract_features_librosa(audio_path)


async def _handle_enroll(arguments: dict, context: dict) -> dict:
    """Enroll a new voice — extract features, encode glyph, store via SDK patterns.

    Supports multi-phrase enrollment: pass g_number + phrase_number (1-3) to add
    additional recordings to an existing speaker. Each phrase exercises different
    vocal features for a more robust voiceprint.
    """
    t0 = time.time()
    audio_b64 = arguments["audio_base64"]
    speaker_name = arguments.get("speaker_name", "")
    existing_g = arguments.get("g_number", "")
    phrase_number = arguments.get("phrase_number", 1)
    session_id = arguments.get("session_id", "")
    recording_label = arguments.get("recording_label", "")
    eye_features = arguments.get("eye_features")  # Optional eye tracking data

    audio_path = _decode_audio_to_file(audio_b64)

    if not _check_audio_energy(audio_path):
        import os
        os.unlink(audio_path)
        return {
            "state": "ASK",
            "confidence": 0.0,
            "match_method": "",
            "query_time_ms": round((time.time() - t0) * 1000, 1),
            "fact_tree": {"message": "No speech detected. Please speak clearly into the microphone."},
            "ask": {"question": "Audio energy too low — ambient noise or silence. Please try again."},
        }

    try:
        features = _extract_features_safe(audio_path)
    finally:
        import os
        os.unlink(audio_path)

    if not _has_valid_features(features):
        return {
            "state": "ASK",
            "confidence": 0.0,
            "match_method": "",
            "query_time_ms": round((time.time() - t0) * 1000, 1),
            "fact_tree": {"message": "Audio too short or silent. Speak for at least one second."},
            "ask": {"question": "Could not extract voice features. Please record again with more speech."},
        }

    encoder = context["encoder"]
    session_factory = context["session_factory"]
    model_manager = context["model_manager"]
    org_id = context["org_id"]
    model_id = context["model_id"]

    from glyphh.core.types import Concept

    # If adding to existing enrollment, reuse G-number
    if existing_g:
        g_number = existing_g
    else:
        async with session_factory() as session:
            from sqlalchemy import text
            # Count unique speakers, not total rows
            row = await session.execute(
                text("""
                    SELECT COUNT(DISTINCT concept_text) FROM glyphs
                    WHERE org_id = :org AND model_id = :model
                """),
                {"org": org_id, "model": model_id},
            )
            count = row.scalar() or 0
        # Base-36 alphanumeric ID — supports billions of users
        def _to_base36(n: int) -> str:
            chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if n == 0:
                return "0"
            result = []
            while n:
                result.append(chars[n % 36])
                n //= 36
            return "".join(reversed(result))

        g_number = f"G-{_to_base36(count + 1).zfill(5)}"

    # Encode via SDK — produces proper hierarchical glyph
    concept_dict = encode_features(features, speaker_id=g_number)
    concept = Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
    glyph = encoder.encode(concept)

    record = entry_to_record({
        "speaker_id": g_number,
        **features,
        "recording_type": "authentic",
    })
    record["metadata"]["speaker_name"] = speaker_name
    record["metadata"]["phrase_number"] = phrase_number
    if session_id:
        record["metadata"]["session_id"] = session_id
    if recording_label:
        record["metadata"]["recording_label"] = recording_label
    if eye_features:
        record["metadata"]["eye_features"] = eye_features
    # Store raw features for lie detection (HDC vectors lack resolution)
    safe_features = {k: round(v, 4) if math.isfinite(v) else 0.0 for k, v in features.items()}
    record["metadata"]["features"] = safe_features
    metadata = record.get("metadata", {})

    # Store using the STANDARD runtime pattern:
    # 1. Global cortex embedding in glyphs table (for GQL FIND SIMILAR)
    # 2. Per-layer/segment/role vectors in glyph_vectors (for role-level re-scoring)
    embedding = glyph.global_cortex.data.astype(float).tolist()

    from domains.models.storage import GlyphStorage

    async with session_factory() as session:
        storage = GlyphStorage(session)
        glyph_response = await storage.create_glyph(
            org_id=org_id,
            model_id=model_id,
            concept_text=g_number,
            embedding=embedding,
            metadata={**metadata, "record_type": "enrollment"},
        )

        # Store hierarchical vectors — this is what enables role-level re-scoring
        from domains.listeners.async_service import _extract_hierarchical_vectors
        hierarchical = _extract_hierarchical_vectors(glyph)
        if hierarchical:
            await storage.create_glyph_vectors_batch(
                glyph_id=glyph_response.glyph_id,
                org_id=org_id,
                model_id=model_id,
                vectors=hierarchical,
            )

        await session.commit()

    elapsed_ms = (time.time() - t0) * 1000

    return {
        "state": "DONE",
        "confidence": 1.0,
        "match_method": "enrollment",
        "query_time_ms": round(elapsed_ms, 1),
        "fact_tree": {
            "g_number": g_number,
            "speaker_name": speaker_name,
            "phrase_number": phrase_number,
            "features": safe_features,
            "message": f"Voice enrolled as {g_number} (phrase {phrase_number}/3).",
        },
    }


async def _handle_identify(arguments: dict, context: dict) -> dict:
    """Identify a speaker — pgvector rough filter → SDK role-level re-scoring.

    Pipeline:
        1. Extract features, encode query glyph via SDK
        2. pgvector cosine on identity layer cortex → top-5 candidates
        3. Fetch per-role vectors for each candidate from glyph_vectors
        4. SDK cosine_similarity per-role with weights → final score
    """
    t0 = time.time()
    audio_b64 = arguments["audio_base64"]

    audio_path = _decode_audio_to_file(audio_b64)

    if not _check_audio_energy(audio_path):
        import os
        os.unlink(audio_path)
        return {
            "state": "ASK",
            "confidence": 0.0,
            "match_method": "",
            "query_time_ms": round((time.time() - t0) * 1000, 1),
            "fact_tree": {"message": "No speech detected. Please speak clearly into the microphone."},
            "ask": {"question": "Audio energy too low — ambient noise or silence. Please try again."},
        }

    # Liveness check — detect replay/playback attacks.
    # Speakers roll off high frequencies and alter spectral flatness.
    liveness = compute_liveness_score(audio_path)
    log.info(
        "Liveness: hf_ratio=%.4f flatness=%.4f score=%.3f live=%s",
        liveness["hf_energy_ratio"], liveness["spectral_flatness"],
        liveness["liveness_score"], liveness["is_live"],
    )

    try:
        features = _extract_features_safe(audio_path)
    finally:
        import os
        os.unlink(audio_path)

    if not _has_valid_features(features):
        return {
            "state": "ASK",
            "confidence": 0.0,
            "match_method": "",
            "query_time_ms": round((time.time() - t0) * 1000, 1),
            "fact_tree": {"message": "Audio too short or silent. Speak for at least one second."},
            "ask": {"question": "Could not extract voice features. Please record again with more speech."},
        }

    if not liveness["is_live"]:
        return {
            "state": "NO_MATCH",
            "confidence": 0.0,
            "match_method": "liveness_rejected",
            "query_time_ms": round((time.time() - t0) * 1000, 1),
            "fact_tree": {
                "message": "Replay attack detected. This doesn't sound like a live voice.",
                "liveness_score": liveness["liveness_score"],
                "hf_energy_ratio": liveness["hf_energy_ratio"],
                "spectral_flatness": liveness["spectral_flatness"],
            },
        }

    encoder = context["encoder"]
    session_factory = context["session_factory"]
    org_id = context["org_id"]
    model_id = context["model_id"]

    from glyphh.core.types import Concept
    import numpy as np

    concept_dict = encode_features(features)
    concept = Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
    query_glyph = encoder.encode(concept)

    # Step 1: pgvector rough filter — global cortex cosine (uses all layers)
    query_cortex_vec = query_glyph.global_cortex.data.tolist()

    async with session_factory() as session:
        from sqlalchemy import text

        rows = await session.execute(
            text("""
                SELECT id, concept_text, metadata,
                       embedding <=> CAST(:qvec AS vector) AS distance
                FROM glyphs
                WHERE org_id = :org AND model_id = :model
                ORDER BY distance ASC
                LIMIT 10
            """),
            {"org": org_id, "model": model_id, "qvec": str(query_cortex_vec)},
        )
        candidates = rows.fetchall()

    if not candidates:
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "state": "NO_MATCH",
            "confidence": 0.0,
            "match_method": "identity_roles",
            "query_time_ms": round(elapsed_ms, 1),
            "fact_tree": {"message": "No enrolled voices found. Enroll first."},
        }

    # Step 2: Build query role dict — ALL layers (identity + emotional + cognitive + cadence)
    query_roles = {}  # {layer.segment.role: Vector}
    for layer_name, layer_obj in query_glyph.layers.items():
        if layer_name.startswith("_"):
            continue  # skip _temporal
        for seg_name, seg_obj in layer_obj.segments.items():
            for role_name, role_vec in seg_obj.roles.items():
                path = f"{layer_name}.{seg_name}.{role_name}"
                query_roles[path] = role_vec

    scored = []
    async with session_factory() as session:
        from sqlalchemy import text

        for candidate in candidates:
            glyph_id = candidate.id
            meta = candidate.metadata or {}
            g_number = meta.get("speaker_id", candidate.concept_text.split("@")[0] if candidate.concept_text else "G-?????")
            speaker_name = meta.get("speaker_name", "")

            # Fetch ALL role vectors for this candidate (all 4 layers)
            role_rows = await session.execute(
                text("""
                    SELECT path, embedding
                    FROM glyph_vectors
                    WHERE glyph_id = :gid
                      AND level = 'role'
                      AND path NOT LIKE '_temporal.%'
                """),
                {"gid": glyph_id},
            )
            role_data = role_rows.fetchall()

            # Build candidate role dict: path → numpy array
            candidate_roles = {}
            for row in role_data:
                emb = row.embedding
                if isinstance(emb, str):
                    emb = json.loads(emb)
                candidate_roles[row.path] = np.array(emb, dtype=np.int8)

            # Pattern A: weighted role-level scoring across ALL layers
            weighted_sum = 0.0
            weight_total = 0.0
            for path, weight in ALL_ROLE_WEIGHTS.items():
                if path in query_roles and path in candidate_roles:
                    sim = cosine_similarity(
                        query_roles[path].data,
                        candidate_roles[path],
                    )
                    weighted_sum += sim * weight
                    weight_total += weight

            role_score = weighted_sum / weight_total if weight_total > 0 else 0.0
            cortex_score = 1.0 - candidate.distance

            scored.append({
                "g_number": g_number,
                "speaker_name": speaker_name,
                "role_score": role_score,
                "cortex_score": cortex_score,
                "roles_matched": int(weight_total > 0),
            })

    # Group by g_number — take best score per speaker (multi-phrase enrollment)
    best_per_speaker = {}
    for s in scored:
        g = s["g_number"]
        if g not in best_per_speaker or s["role_score"] > best_per_speaker[g]["role_score"]:
            best_per_speaker[g] = s
    scored = sorted(best_per_speaker.values(), key=lambda x: x["role_score"], reverse=True)

    elapsed_ms = (time.time() - t0) * 1000
    best = scored[0]
    best_score = best["role_score"]

    log.info(
        "Identify: best=%s score=%.4f cortex=%.4f candidates=%d | top3: %s",
        best["g_number"], best_score, best["cortex_score"], len(scored),
        ", ".join(f'{s["g_number"]}={s["role_score"]:.3f}' for s in scored[:3]),
    )
    g_number = best["g_number"]
    speaker_name = best["speaker_name"]

    threshold = 0.50
    if best_score >= threshold:
        # Store temporal glyph — same G-number, new timestamp
        # This builds the voice history for drift tracking
        embedding = query_glyph.global_cortex.data.astype(float).tolist()
        from domains.models.storage import GlyphStorage

        async with session_factory() as session:
            storage = GlyphStorage(session)
            glyph_response = await storage.create_glyph(
                org_id=org_id,
                model_id=model_id,
                concept_text=g_number,
                embedding=embedding,
                metadata={
                    "speaker_id": g_number,
                    "speaker_name": speaker_name,
                    "record_type": "identification",
                },
            )

            from domains.listeners.async_service import _extract_hierarchical_vectors
            hierarchical = _extract_hierarchical_vectors(query_glyph)
            if hierarchical:
                await storage.create_glyph_vectors_batch(
                    glyph_id=glyph_response.glyph_id,
                    org_id=org_id,
                    model_id=model_id,
                    vectors=hierarchical,
                )

            await session.commit()

        elapsed_ms = (time.time() - t0) * 1000

        # Sanitize features for response
        safe_features = {}
        for k, v in features.items():
            if isinstance(v, float) and math.isfinite(v):
                safe_features[k] = round(v, 4)
            else:
                safe_features[k] = 0.0

        return {
            "state": "DONE",
            "confidence": round(best_score, 4),
            "match_method": "all_roles",
            "query_time_ms": round(elapsed_ms, 1),
            "fact_tree": {
                "g_number": g_number,
                "speaker_name": speaker_name,
                "similarity": round(best_score, 4),
                "cortex_similarity": round(best["cortex_score"], 4),
                "liveness_score": liveness["liveness_score"],
                "features": safe_features,
                "candidates": [
                    {
                        "g_number": s["g_number"],
                        "similarity": round(s["role_score"], 4),
                        "cortex_similarity": round(s["cortex_score"], 4),
                    }
                    for s in scored[:3]
                ],
                "message": (
                    f"Identified as {g_number}"
                    + (f" ({speaker_name})" if speaker_name else "")
                    + f" with {best_score:.1%} confidence."
                ),
            },
        }
    else:
        return {
            "state": "ASK",
            "confidence": round(best_score, 4),
            "match_method": "all_roles",
            "query_time_ms": round(elapsed_ms, 1),
            "fact_tree": {
                "best_match": g_number,
                "similarity": round(best_score, 4),
                "cortex_similarity": round(best["cortex_score"], 4),
                "candidates": [
                    {
                        "g_number": s["g_number"],
                        "similarity": round(s["role_score"], 4),
                        "cortex_similarity": round(s["cortex_score"], 4),
                    }
                    for s in scored[:3]
                ],
                "message": (
                    f"Low confidence match to {g_number} ({best_score:.1%}). "
                    "This may be a new speaker — enroll?"
                ),
            },
            "ask": {
                "question": "Voice not recognized with high confidence. Enroll as new speaker?",
                "disambiguation_options": [
                    {
                        "label": s["g_number"],
                        "confidence": round(s["role_score"], 4),
                    }
                    for s in scored[:3]
                ],
            },
        }


# ---------------------------------------------------------------------------
# Eye tracking (retina) bypassed in v3 — voice-only signals.
# ---------------------------------------------------------------------------


async def _handle_compare(arguments: dict, context: dict) -> dict:
    """Compare two statements against a baseline — pick which diverges more on
    emotional/cognitive layers (the likely lie).

    v4: Compares RAW acoustic features directly instead of HDC vectors.
    HDC binary vectors lack resolution for within-speaker state comparison —
    different words produce wildly different vectors (cosine ~0.2-0.5),
    drowning any lie signal in phonetic noise.
    """
    t0 = time.time()
    baseline_g = arguments["baseline_g"]
    statement_a_g = arguments["statement_a_g"]
    statement_b_g = arguments["statement_b_g"]

    session_factory = context["session_factory"]
    org_id = context["org_id"]
    model_id = context["model_id"]

    # Fetch raw features from glyph metadata
    async def _fetch_features(g_number: str) -> dict | None:
        async with session_factory() as session:
            from sqlalchemy import text
            row = await session.execute(
                text("""
                    SELECT metadata FROM glyphs
                    WHERE org_id = :org AND model_id = :model
                      AND concept_text = :g
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"org": org_id, "model": model_id, "g": g_number},
            )
            r = row.fetchone()
            if r and r.metadata:
                return r.metadata.get("features")
        return None

    feat_baseline = await _fetch_features(baseline_g)
    feat_a = await _fetch_features(statement_a_g)
    feat_b = await _fetch_features(statement_b_g)

    if not feat_baseline or not feat_a or not feat_b:
        missing = []
        if not feat_baseline: missing.append(f"baseline {baseline_g}")
        if not feat_a: missing.append(f"statement_a {statement_a_g}")
        if not feat_b: missing.append(f"statement_b {statement_b_g}")
        return {
            "state": "ERROR", "confidence": 0, "match_method": "",
            "query_time_ms": round((time.time() - t0) * 1000, 1),
            "fact_tree": {"error": f"Missing features for: {', '.join(missing)}. Re-enroll to store raw features."},
        }

    # -----------------------------------------------------------------------
    # DIRECTIONAL LIE DETECTION (v4.1)
    #
    # Only uses UTTERANCE-INDEPENDENT stress markers — features that reflect
    # vocal quality regardless of what words are spoken. Cadence, loudness,
    # formants, MFCCs are all phrase-dependent and create false signals.
    #
    # Research-backed directional signals:
    #   Lying → jitter ↑, shimmer ↑, f0_std ↑, HNR ↓, alpha_ratio shifts
    #
    # A (calibration lie) teaches us this person's lie direction/magnitude.
    # B is scored by whether it shifts in the same direction as A.
    # -----------------------------------------------------------------------

    # Stress markers — direction is LEARNED from calibration, not hardcoded.
    # Whatever direction A shifts from baseline IS this person's lie direction.
    STRESS_MARKERS = [
        "jitter", "shimmer", "f0_std", "hnr",
        "alpha_ratio", "hammarberg_index", "mfcc_1_cv",
    ]

    def _signed_change(baseline_val: float, test_val: float) -> float:
        """Signed fractional change from baseline."""
        if abs(baseline_val) < 1e-6:
            return test_val * 10 if abs(test_val) > 1e-6 else 0.0
        return (test_val - baseline_val) / abs(baseline_val)

    # For each marker, compute shift from baseline
    marker_scores = []
    marker_details = {}

    for feat_name in STRESS_MARKERS:
        bv = feat_baseline.get(feat_name, 0.0)
        av = feat_a.get(feat_name, 0.0)
        bv_test = feat_b.get(feat_name, 0.0)

        # Raw signed shifts from baseline
        shift_a = _signed_change(bv, av)  # calibration lie shift
        shift_b = _signed_change(bv, bv_test)  # test statement shift

        marker_details[feat_name] = {
            "baseline": round(bv, 4),
            "cal_lie": round(av, 4),
            "test": round(bv_test, 4),
            "shift_a": round(shift_a, 4),
            "shift_b": round(shift_b, 4),
        }

        # Skip dead features (both zero)
        if abs(shift_a) < 0.001 and abs(shift_b) < 0.001:
            score = 0.5
            marker_scores.append(score)
            marker_details[feat_name]["score"] = round(score, 4)
            continue

        # A's shift DEFINES the lie direction for this person.
        # If A shifted meaningfully, check if B shifts the SAME direction.
        if abs(shift_a) > 0.03:
            # Project B's shift onto A's direction
            # Same direction & similar magnitude = lie, opposite = truth
            if shift_a > 0:
                # Lie direction is positive for this feature
                if shift_b <= 0:
                    score = 0.0  # B went opposite → truth
                else:
                    ratio = shift_b / shift_a
                    score = max(0.0, min(1.0, (ratio - 0.3) / 0.4))
            else:
                # Lie direction is negative for this feature
                if shift_b >= 0:
                    score = 0.0  # B went opposite → truth
                else:
                    ratio = shift_b / shift_a  # both negative → positive ratio
                    score = max(0.0, min(1.0, (ratio - 0.3) / 0.4))
        else:
            # A barely shifted — this marker isn't useful for this person
            score = 0.5

        marker_scores.append(score)
        marker_details[feat_name]["score"] = round(score, 4)

    # Final score — equal weight per marker (all are research-validated)
    if marker_scores:
        raw_score_b = sum(marker_scores) / len(marker_scores)
    else:
        raw_score_b = 0.5

    # -------------------------------------------------------------------
    # HDC TIEBREAKER (v4.2)
    #
    # When raw features are inconclusive (0.40-0.60), use HDC role-level
    # cosine similarities as a tiebreaker.  HDC vectors capture holistic
    # cross-feature correlations within each layer — signal that per-
    # feature averaging misses.
    #
    # For state-sensitive layers (emotional, cognitive, cadence), the
    # statement that diverges MORE from baseline is the likely lie.
    # Identity should stay similar (same speaker) — we skip it.
    # -------------------------------------------------------------------
    import numpy as np

    hdc_adjustment = 0.0
    hdc_detail = {}

    # Only fetch HDC when raw features are in the uncertain zone
    if 0.40 <= raw_score_b <= 0.60:
        async def _fetch_role_vectors(g_number: str) -> dict:
            """Fetch HDC role vectors {path: np.array} for a glyph."""
            async with session_factory() as session:
                from sqlalchemy import text
                # Get the glyph_id first
                glyph_row = await session.execute(
                    text("""
                        SELECT id FROM glyphs
                        WHERE org_id = :org AND model_id = :model
                          AND concept_text = :g
                        ORDER BY created_at DESC
                        LIMIT 1
                    """),
                    {"org": org_id, "model": model_id, "g": g_number},
                )
                gr = glyph_row.fetchone()
                if not gr:
                    return {}

                role_rows = await session.execute(
                    text("""
                        SELECT path, embedding
                        FROM glyph_vectors
                        WHERE glyph_id = :gid
                          AND level = 'role'
                          AND path NOT LIKE '_temporal.%%'
                    """),
                    {"gid": gr.id},
                )
                roles = {}
                for row in role_rows.fetchall():
                    emb = row.embedding
                    if isinstance(emb, str):
                        emb = json.loads(emb)
                    roles[row.path] = np.array(emb, dtype=np.int8)
                return roles

        roles_baseline = await _fetch_role_vectors(baseline_g)
        roles_a = await _fetch_role_vectors(statement_a_g)
        roles_b = await _fetch_role_vectors(statement_b_g)

        if roles_baseline and roles_a and roles_b:
            # Compute per-layer average cosine similarity vs baseline
            # State-sensitive layers only — identity is same speaker, skip it
            STATE_LAYERS = {
                "emotional_state": 0.40,   # prosody shifts most under stress
                "cognitive_load":  0.40,   # jitter/shimmer/tension markers
                "cadence":         0.20,   # rhythm changes
            }

            divergence_a = 0.0
            divergence_b = 0.0
            weight_total = 0.0

            for layer_prefix, layer_weight in STATE_LAYERS.items():
                # Collect role paths belonging to this layer
                layer_paths = [p for p in ALL_ROLE_WEIGHTS if p.startswith(layer_prefix + ".")]

                sims_a = []
                sims_b = []
                for path in layer_paths:
                    if path in roles_baseline and path in roles_a and path in roles_b:
                        sim_a = cosine_similarity(roles_baseline[path], roles_a[path])
                        sim_b = cosine_similarity(roles_baseline[path], roles_b[path])
                        sims_a.append(sim_a)
                        sims_b.append(sim_b)

                if sims_a:
                    avg_sim_a = sum(sims_a) / len(sims_a)
                    avg_sim_b = sum(sims_b) / len(sims_b)
                    # Lower similarity = more divergence = more likely lie
                    div_a = 1.0 - avg_sim_a
                    div_b = 1.0 - avg_sim_b
                    divergence_a += div_a * layer_weight
                    divergence_b += div_b * layer_weight
                    weight_total += layer_weight

                    hdc_detail[layer_prefix] = {
                        "sim_a": round(avg_sim_a, 4),
                        "sim_b": round(avg_sim_b, 4),
                        "div_a": round(div_a, 4),
                        "div_b": round(div_b, 4),
                        "roles_matched": len(sims_a),
                    }

            if weight_total > 0:
                divergence_a /= weight_total
                divergence_b /= weight_total

                # HDC says: whichever diverges more is the lie
                # Convert to a 0-1 score where >0.5 means B is the lie
                total_div = divergence_a + divergence_b
                if total_div > 1e-6:
                    hdc_score_b = divergence_b / total_div
                else:
                    hdc_score_b = 0.5

                # Adjustment: push raw score toward HDC opinion
                # Scale: max ±0.15 adjustment (enough to break a 0.50 tie
                # but not enough to override a clear raw signal)
                hdc_adjustment = (hdc_score_b - 0.5) * 0.30

                hdc_detail["hdc_score_b"] = round(hdc_score_b, 4)
                hdc_detail["adjustment"] = round(hdc_adjustment, 4)

                log.info(
                    "HDC tiebreaker: div_a=%.4f div_b=%.4f hdc_score=%.4f adj=%.4f",
                    divergence_a, divergence_b, hdc_score_b, hdc_adjustment,
                )

    # Blend raw + HDC
    lie_score_b = max(0.0, min(1.0, raw_score_b + hdc_adjustment))
    lie_score_a = 1.0 - lie_score_b

    if lie_score_b > 0.55:
        likely_lie = "B"
        lie_confidence = lie_score_b
    elif lie_score_b < 0.45:
        likely_lie = "A"
        lie_confidence = lie_score_a
    else:
        likely_lie = "uncertain"
        lie_confidence = 0.5

    elapsed_ms = (time.time() - t0) * 1000

    # Build signal summary
    signals = {}
    for feat_name in STRESS_MARKERS:
        signals[feat_name] = marker_details[feat_name]

    # Layer-level summary for reporting (raw feature based)
    def _pct_change_abs(bv: float, tv: float) -> float:
        if abs(bv) < 1e-6: return abs(tv) * 100
        return abs(tv - bv) / abs(bv)

    layer_feat_groups = {
        "identity": ["hnr"],
        "emotional_state": ["f0_std"],
        "cognitive_load": ["jitter", "shimmer", "alpha_ratio", "hammarberg_index", "mfcc_1_cv"],
        "cadence": [],
    }

    def _layer_sim(feat_ref: dict, feat_test: dict, keys: list) -> float:
        if not keys: return 1.0
        sims = [max(0.0, 1.0 - _pct_change_abs(feat_ref.get(k, 0), feat_test.get(k, 0))) for k in keys]
        return round(sum(sims) / len(sims), 4)

    layer_names = ["identity", "emotional_state", "cognitive_load", "cadence"]
    scores_a = {ln: _layer_sim(feat_baseline, feat_a, layer_feat_groups[ln]) for ln in layer_names}
    scores_b = {ln: _layer_sim(feat_baseline, feat_b, layer_feat_groups[ln]) for ln in layer_names}

    return {
        "state": "DONE",
        "confidence": round(lie_confidence, 4),
        "match_method": "lie_detection_v4.2",
        "query_time_ms": round(elapsed_ms, 1),
        "fact_tree": {
            "likely_lie": likely_lie,
            "lie_confidence": round(lie_confidence, 4),
            "statement_a_scores": scores_a,
            "statement_b_scores": scores_b,
            "signals": signals,
            "lie_scores": {
                "raw_b": round(raw_score_b, 4),
                "hdc_adjustment": round(hdc_adjustment, 4),
                "final_a": round(lie_score_a, 4),
                "final_b": round(lie_score_b, 4),
            },
            "hdc_tiebreaker": hdc_detail if hdc_detail else "not_needed",
            "detail": {k: v for k, v in marker_details.items()},
            "analysis": {
                "identity_a": scores_a.get("identity", 0),
                "identity_b": scores_b.get("identity", 0),
                "emotional_a": scores_a.get("emotional_state", 0),
                "emotional_b": scores_b.get("emotional_state", 0),
                "cognitive_a": scores_a.get("cognitive_load", 0),
                "cognitive_b": scores_b.get("cognitive_load", 0),
                "cadence_a": scores_a.get("cadence", 0),
                "cadence_b": scores_b.get("cadence", 0),
            },
        },
    }


async def _handle_cleanup(arguments: dict, context: dict) -> dict:
    """Delete all glyphs for a given session ID."""
    t0 = time.time()
    session_id = arguments["session_id"]

    session_factory = context["session_factory"]
    org_id = context["org_id"]
    model_id = context["model_id"]

    async with session_factory() as session:
        from sqlalchemy import text

        # Find glyphs with this session_id in metadata
        rows = await session.execute(
            text("""
                SELECT id FROM glyphs
                WHERE org_id = :org AND model_id = :model
                  AND metadata->>'session_id' = :sid
            """),
            {"org": org_id, "model": model_id, "sid": session_id},
        )
        glyph_ids = [r.id for r in rows.fetchall()]

        if glyph_ids:
            # Delete role vectors first (foreign key)
            await session.execute(
                text("""
                    DELETE FROM glyph_vectors
                    WHERE glyph_id = ANY(:ids)
                """),
                {"ids": glyph_ids},
            )
            # Delete glyphs
            await session.execute(
                text("""
                    DELETE FROM glyphs
                    WHERE id = ANY(:ids)
                """),
                {"ids": glyph_ids},
            )
            await session.commit()

    elapsed_ms = (time.time() - t0) * 1000
    return {
        "state": "DONE",
        "confidence": 1.0,
        "match_method": "cleanup",
        "query_time_ms": round(elapsed_ms, 1),
        "fact_tree": {
            "session_id": session_id,
            "deleted_count": len(glyph_ids),
            "message": f"Deleted {len(glyph_ids)} glyphs for session {session_id}.",
        },
    }


async def handle_mcp_tool(tool_name: str, arguments: dict, context: dict) -> dict:
    """Handle model-specific MCP tool calls."""
    handlers = {
        "voice_enroll": _handle_enroll,
        "voice_identify": _handle_identify,
        "voice_list": _handle_list,
        "voice_compare": _handle_compare,
        "voice_cleanup": _handle_cleanup,
    }
    handler = handlers.get(tool_name)
    if not handler:
        return {
            "state": "ERROR",
            "confidence": 0,
            "match_method": "",
            "fact_tree": {"error": f"Unknown tool: {tool_name}"},
        }
    return await handler(arguments, context)
