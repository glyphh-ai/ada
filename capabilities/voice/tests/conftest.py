"""Shared fixtures for Ada Voice tests."""

import sys
from pathlib import Path

import pytest

# Add model root to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ada import Encoder
from glyphh.core.types import Concept

from encoder import ENCODER_CONFIG, encode_features, score_identity_roles as score_identity, score_state_drift


# ---------------------------------------------------------------------------
# Synthetic speaker profiles — realistic eGeMAPS feature ranges
# ---------------------------------------------------------------------------

# Speaker A: adult male, calm baseline
SPEAKER_A_REAL = {
    "mfcc_1": 12.5, "mfcc_2": -3.2, "mfcc_3": 1.8, "mfcc_4": -0.5,
    "formant_f1": 450.0, "formant_f2": 1200.0, "formant_f3": 2600.0,
    "hnr": 18.0,
    "f0_mean": 14.0, "f0_std": 0.25, "loudness_mean": 0.45, "loudness_std": 0.30,
    "spectral_slope": -0.01,
    "f0_p20": 12.0, "f0_p80": 16.0, "spectral_flux": 0.08,
    "jitter": 0.015, "shimmer": 0.50, "voiced_std": 0.12, "pause_std": 0.35,
    "alpha_ratio": 2.5, "hammarberg_index": 15.0, "formant_f1_bw": 120.0, "mfcc_1_cv": 0.4,
    "loudness_peaks_per_sec": 3.5, "voiced_segment_mean": 0.25,
    "unvoiced_segment_mean": 0.18, "voiced_segments_per_sec": 4.0,
}

# Speaker A faking voice: pitched up, exaggerated, but same vocal tract
SPEAKER_A_FAKED = {
    "mfcc_1": 13.0, "mfcc_2": -2.8, "mfcc_3": 2.1, "mfcc_4": -0.3,  # MFCCs barely change
    "formant_f1": 460.0, "formant_f2": 1220.0, "formant_f3": 2620.0,  # formants barely change
    "hnr": 15.0,  # slightly noisier
    "f0_mean": 22.0, "f0_std": 0.45, "loudness_mean": 0.60, "loudness_std": 0.50,  # big pitch/energy shift
    "spectral_slope": -0.005,
    "f0_p20": 18.0, "f0_p80": 26.0, "spectral_flux": 0.12,  # exaggerated range
    "jitter": 0.030, "shimmer": 0.80, "voiced_std": 0.20, "pause_std": 0.50,  # more strain
    "alpha_ratio": 4.0, "hammarberg_index": 18.0, "formant_f1_bw": 180.0, "mfcc_1_cv": 0.7,
    "loudness_peaks_per_sec": 2.8, "voiced_segment_mean": 0.20,
    "unvoiced_segment_mean": 0.25, "voiced_segments_per_sec": 3.2,  # slower cadence
}

# Speaker B: adult female, different vocal tract
SPEAKER_B_REAL = {
    "mfcc_1": 22.0, "mfcc_2": 4.5, "mfcc_3": -2.0, "mfcc_4": 1.2,
    "formant_f1": 600.0, "formant_f2": 1800.0, "formant_f3": 3200.0,
    "hnr": 22.0,
    "f0_mean": 24.0, "f0_std": 0.30, "loudness_mean": 0.50, "loudness_std": 0.25,
    "spectral_slope": -0.008,
    "f0_p20": 21.0, "f0_p80": 27.0, "spectral_flux": 0.10,
    "jitter": 0.012, "shimmer": 0.40, "voiced_std": 0.10, "pause_std": 0.30,
    "alpha_ratio": 3.0, "hammarberg_index": 17.0, "formant_f1_bw": 110.0, "mfcc_1_cv": 0.35,
    "loudness_peaks_per_sec": 4.0, "voiced_segment_mean": 0.22,
    "unvoiced_segment_mean": 0.15, "voiced_segments_per_sec": 4.5,
}

# Speaker C: different adult male (control — should NOT match Speaker A)
SPEAKER_C_REAL = {
    "mfcc_1": 8.0, "mfcc_2": -8.0, "mfcc_3": 5.0, "mfcc_4": -3.0,
    "formant_f1": 380.0, "formant_f2": 1050.0, "formant_f3": 2400.0,
    "hnr": 14.0,
    "f0_mean": 12.0, "f0_std": 0.20, "loudness_mean": 0.40, "loudness_std": 0.35,
    "spectral_slope": -0.015,
    "f0_p20": 10.0, "f0_p80": 14.0, "spectral_flux": 0.06,
    "jitter": 0.020, "shimmer": 0.60, "voiced_std": 0.15, "pause_std": 0.40,
    "alpha_ratio": 1.8, "hammarberg_index": 12.0, "formant_f1_bw": 140.0, "mfcc_1_cv": 0.5,
    "loudness_peaks_per_sec": 3.0, "voiced_segment_mean": 0.28,
    "unvoiced_segment_mean": 0.20, "voiced_segments_per_sec": 3.5,
}

# Speaker A return visit — same person, slightly different state
SPEAKER_A_RETURN = {
    "mfcc_1": 12.3, "mfcc_2": -3.0, "mfcc_3": 1.9, "mfcc_4": -0.6,  # very close to real
    "formant_f1": 448.0, "formant_f2": 1195.0, "formant_f3": 2590.0,
    "hnr": 17.5,
    "f0_mean": 15.0, "f0_std": 0.28, "loudness_mean": 0.48, "loudness_std": 0.32,
    "spectral_slope": -0.012,
    "f0_p20": 12.5, "f0_p80": 17.0, "spectral_flux": 0.09,
    "jitter": 0.016, "shimmer": 0.52, "voiced_std": 0.13, "pause_std": 0.38,
    "alpha_ratio": 2.6, "hammarberg_index": 15.5, "formant_f1_bw": 125.0, "mfcc_1_cv": 0.42,
    "loudness_peaks_per_sec": 3.3, "voiced_segment_mean": 0.24,
    "unvoiced_segment_mean": 0.19, "voiced_segments_per_sec": 3.8,
}


@pytest.fixture
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture
def speaker_a_real_glyph(encoder):
    attrs = encode_features(SPEAKER_A_REAL, speaker_id="G-00001")
    return encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))


@pytest.fixture
def speaker_a_faked_glyph(encoder):
    attrs = encode_features(SPEAKER_A_FAKED, speaker_id="G-00001")
    return encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))


@pytest.fixture
def speaker_b_real_glyph(encoder):
    attrs = encode_features(SPEAKER_B_REAL, speaker_id="G-00002")
    return encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))


@pytest.fixture
def speaker_c_real_glyph(encoder):
    attrs = encode_features(SPEAKER_C_REAL, speaker_id="G-00003")
    return encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))


@pytest.fixture
def speaker_a_return_glyph(encoder):
    attrs = encode_features(SPEAKER_A_RETURN, speaker_id="G-00001")
    return encoder.encode(Concept(name=attrs["name"], attributes=attrs["attributes"]))
