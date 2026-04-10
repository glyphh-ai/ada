"""Tests for encoder config validation and feature encoding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ada import Encoder
from glyphh.core.types import Concept

from encoder import ENCODER_CONFIG, encode_features, entry_to_record
from conftest import SPEAKER_A_REAL, SPEAKER_B_REAL


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestEncoderConfig:
    def test_config_creates_encoder(self):
        encoder = Encoder(ENCODER_CONFIG)
        assert encoder is not None

    def test_dimension(self):
        assert ENCODER_CONFIG.dimension == 2000

    def test_seed(self):
        assert ENCODER_CONFIG.seed == 42

    def test_temporal_enabled(self):
        assert ENCODER_CONFIG.include_temporal is True

    def test_four_layers(self):
        layer_names = [l.name for l in ENCODER_CONFIG.layers]
        assert layer_names == ["identity", "emotional_state", "cognitive_load", "cadence"]

    def test_layer_weights_sum_to_one(self):
        total = sum(l.similarity_weight for l in ENCODER_CONFIG.layers)
        assert abs(total - 1.0) < 0.01

    def test_no_key_part_roles(self):
        """speaker_id is metadata only — no key_part roles in the config."""
        for layer in ENCODER_CONFIG.layers:
            for seg in layer.segments:
                for role in seg.roles:
                    assert not role.key_part, (
                        f"Role {role.name} in {layer.name}/{seg.name} "
                        f"should not be key_part (speaker_id is metadata)"
                    )

    def test_all_roles_are_numeric(self):
        """Every role should have numeric_config (thermometer)."""
        for layer in ENCODER_CONFIG.layers:
            for seg in layer.segments:
                for role in seg.roles:
                    assert role.numeric_config is not None, (
                        f"Role {role.name} in {layer.name}/{seg.name} "
                        f"missing numeric_config"
                    )

    def test_identity_roles(self):
        identity = ENCODER_CONFIG.layers[0]
        role_names = [r.name for s in identity.segments for r in s.roles]
        expected = [
            "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
            "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8",
            "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13",
            "mfcc_delta_1", "mfcc_delta_2", "mfcc_delta_3", "mfcc_delta_4",
            "formant_f1", "formant_f2", "formant_f3", "hnr",
        ]
        assert role_names == expected

    def test_emotional_state_roles(self):
        emotional = ENCODER_CONFIG.layers[1]
        role_names = [r.name for s in emotional.segments for r in s.roles]
        assert role_names == [
            "f0_mean", "f0_std", "loudness_mean", "loudness_std", "spectral_slope",
            "f0_p20", "f0_p80", "spectral_flux",
        ]

    def test_cognitive_load_roles(self):
        cognitive = ENCODER_CONFIG.layers[2]
        role_names = [r.name for s in cognitive.segments for r in s.roles]
        assert role_names == [
            "jitter", "shimmer", "voiced_std", "pause_std",
            "alpha_ratio", "hammarberg_index", "formant_f1_bw", "mfcc_1_cv",
        ]

    def test_cadence_roles(self):
        cadence = ENCODER_CONFIG.layers[3]
        role_names = [r.name for s in cadence.segments for r in s.roles]
        assert role_names == [
            "loudness_peaks_per_sec", "voiced_segment_mean",
            "unvoiced_segment_mean", "voiced_segments_per_sec",
        ]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encode_features_returns_dict(self):
        result = encode_features(SPEAKER_A_REAL, speaker_id="G-00001")
        assert "name" in result
        assert "attributes" in result
        # speaker_id should NOT be in attributes (it's metadata only)
        assert "speaker_id" not in result["attributes"]

    def test_encode_features_without_speaker_id(self):
        result = encode_features(SPEAKER_A_REAL)
        assert result["name"].startswith("voice_")
        assert "speaker_id" not in result["attributes"]

    def test_encode_produces_glyph(self, encoder):
        attrs = encode_features(SPEAKER_A_REAL, speaker_id="G-00001")
        glyph = encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))
        assert glyph is not None
        assert glyph.global_cortex is not None
        assert glyph.global_cortex.data.shape[0] == 2000

    def test_glyph_has_all_layers(self, encoder):
        attrs = encode_features(SPEAKER_A_REAL, speaker_id="G-00001")
        glyph = encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))
        for layer_name in ("identity", "emotional_state", "cognitive_load", "cadence"):
            assert layer_name in glyph.layers, f"Missing layer: {layer_name}"

    def test_glyph_has_temporal_layer(self, encoder):
        attrs = encode_features(SPEAKER_A_REAL, speaker_id="G-00001")
        glyph = encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))
        assert "_temporal" in glyph.layers

    def test_deterministic_encoding(self, encoder):
        attrs = encode_features(SPEAKER_A_REAL, speaker_id="G-00001")
        g1 = encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))
        g2 = encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))
        # Same input → same vectors (excluding temporal which uses current time)
        from glyphh.core.ops import cosine_similarity
        sim = cosine_similarity(
            g1.layers["identity"].cortex.data,
            g2.layers["identity"].cortex.data,
        )
        assert sim == 1.0

    def test_different_speakers_produce_different_glyphs(self, encoder):
        a = encode_features(SPEAKER_A_REAL, speaker_id="G-00001")
        b = encode_features(SPEAKER_B_REAL, speaker_id="G-00002")
        g_a = encoder.encode(Concept(name=a["name"], attributes=a["attributes"]))
        g_b = encoder.encode(Concept(name=b["name"], attributes=b["attributes"]))
        from glyphh.core.ops import cosine_similarity
        sim = cosine_similarity(
            g_a.layers["identity"].cortex.data,
            g_b.layers["identity"].cortex.data,
        )
        assert sim < 0.7, f"Different speakers too similar: {sim:.4f}"


# ---------------------------------------------------------------------------
# entry_to_record
# ---------------------------------------------------------------------------

class TestEntryToRecord:
    def test_basic_conversion(self):
        entry = {
            "speaker_id": "G-00001",
            "mfcc_1": 12.5, "mfcc_2": -3.2, "mfcc_3": 1.8, "mfcc_4": -0.5,
            "formant_f1": 450.0, "formant_f2": 1200.0, "formant_f3": 2600.0,
            "hnr": 18.0,
            "f0_mean": 14.0, "f0_std": 0.25,
            "loudness_mean": 0.45, "loudness_std": 0.30, "spectral_slope": -0.01,
            "jitter": 0.015, "shimmer": 0.50, "voiced_std": 0.12, "pause_std": 0.35,
            "loudness_peaks_per_sec": 3.5, "voiced_segment_mean": 0.25,
            "unvoiced_segment_mean": 0.18, "voiced_segments_per_sec": 4.0,
            "recording_type": "authentic",
        }
        record = entry_to_record(entry)
        # speaker_id should be in metadata, not attributes
        assert "speaker_id" not in record["attributes"]
        assert record["metadata"]["speaker_id"] == "G-00001"
        assert record["attributes"]["mfcc_1"] == 12.5
        assert record["metadata"]["recording_type"] == "authentic"

    def test_missing_features_default_to_zero(self):
        entry = {"speaker_id": "G-00001"}
        record = entry_to_record(entry)
        assert record["attributes"]["mfcc_1"] == 0.0
        assert record["attributes"]["f0_mean"] == 0.0
