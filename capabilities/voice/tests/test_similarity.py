"""
Tests for voice identity matching and state drift detection.

V1 success criteria: Two recordings from the same person, one authentic
and one faked, produce a high-confidence identity match AND a measurable
authenticity delta via cosine similarity in HDC space.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from glyphh.core.ops import cosine_similarity

from encoder import score_identity_roles as score_identity, score_state_drift, score_authenticity


# ---------------------------------------------------------------------------
# Identity matching — "Ada knows who you are"
# ---------------------------------------------------------------------------

class TestIdentityMatching:
    def test_same_speaker_high_identity(
        self, speaker_a_real_glyph, speaker_a_return_glyph
    ):
        """Same person on two visits → high identity layer similarity."""
        sim = score_identity(speaker_a_real_glyph, speaker_a_return_glyph)
        assert sim > 0.5, f"Same speaker identity too low: {sim:.4f}"

    def test_different_speakers_low_identity(
        self, speaker_a_real_glyph, speaker_b_real_glyph
    ):
        """Different people → low identity layer similarity."""
        sim = score_identity(speaker_a_real_glyph, speaker_b_real_glyph)
        assert sim < 0.5, f"Different speakers identity too high: {sim:.4f}"

    def test_different_male_speakers_distinguishable(
        self, speaker_a_real_glyph, speaker_c_real_glyph
    ):
        """Two males with different vocal tracts → distinguishable."""
        sim = score_identity(speaker_a_real_glyph, speaker_c_real_glyph)
        assert sim < 0.6, f"Different male speakers too similar: {sim:.4f}"

    def test_faked_voice_still_matches_identity(
        self, speaker_a_real_glyph, speaker_a_faked_glyph
    ):
        """Faked voice from same person → identity layer still matches.
        MFCCs and formants encode vocal tract geometry which can't be faked."""
        sim = score_identity(speaker_a_real_glyph, speaker_a_faked_glyph)
        assert sim > 0.4, f"Faked voice identity too low: {sim:.4f}"

    def test_faked_matches_better_than_different_speaker(
        self, speaker_a_real_glyph, speaker_a_faked_glyph, speaker_b_real_glyph
    ):
        """A's faked voice should match A better than B's real voice."""
        sim_faked = score_identity(speaker_a_real_glyph, speaker_a_faked_glyph)
        sim_other = score_identity(speaker_a_real_glyph, speaker_b_real_glyph)
        assert sim_faked > sim_other, (
            f"Faked voice ({sim_faked:.4f}) should match better than "
            f"different speaker ({sim_other:.4f})"
        )

    def test_return_visit_matches_better_than_different_speaker(
        self, speaker_a_real_glyph, speaker_a_return_glyph, speaker_c_real_glyph
    ):
        """Return visit should match original better than a different person."""
        sim_return = score_identity(speaker_a_real_glyph, speaker_a_return_glyph)
        sim_other = score_identity(speaker_a_real_glyph, speaker_c_real_glyph)
        assert sim_return > sim_other, (
            f"Return visit ({sim_return:.4f}) should match better than "
            f"different speaker ({sim_other:.4f})"
        )


# ---------------------------------------------------------------------------
# State drift — "You sound different today"
# ---------------------------------------------------------------------------

class TestStateDrift:
    def test_return_visit_shows_small_drift(
        self, speaker_a_real_glyph, speaker_a_return_glyph
    ):
        """Same person, slightly different day → small state drift."""
        drift = score_state_drift(speaker_a_return_glyph, speaker_a_real_glyph)
        # Identity should be very stable (dim=2000 gives ~0.58)
        assert drift["identity"] > 0.5
        # State layers should show some but not extreme drift
        assert drift["emotional_state"] > 0.2
        assert drift["cadence"] > 0.2

    def test_faked_voice_shows_large_state_drift(
        self, speaker_a_real_glyph, speaker_a_faked_glyph
    ):
        """Faked voice → large emotional/cognitive drift but stable identity."""
        drift = score_state_drift(speaker_a_faked_glyph, speaker_a_real_glyph)
        # Identity stays positive (same vocal tract, dim=2000 gives ~0.47)
        assert drift["identity"] > 0.4
        # Emotional state should show significant shift (pitch + loudness changed)
        assert drift["emotional_state"] < drift["identity"], (
            f"Emotional state ({drift['emotional_state']:.4f}) should drift "
            f"more than identity ({drift['identity']:.4f})"
        )

    def test_drift_returns_all_layers(
        self, speaker_a_real_glyph, speaker_a_return_glyph
    ):
        drift = score_state_drift(speaker_a_return_glyph, speaker_a_real_glyph)
        assert set(drift.keys()) == {"identity", "emotional_state", "cognitive_load", "cadence"}


# ---------------------------------------------------------------------------
# Authenticity scoring — "You can't fake Ada"
# ---------------------------------------------------------------------------

class TestAuthenticity:
    def test_authenticity_score_structure(
        self, speaker_a_real_glyph, speaker_a_faked_glyph
    ):
        result = score_authenticity(speaker_a_real_glyph, speaker_a_faked_glyph)
        assert "identity_match" in result
        assert "total_shift" in result
        assert "per_layer" in result

    def test_faked_voice_has_positive_shift(
        self, speaker_a_real_glyph, speaker_a_faked_glyph
    ):
        """Faking should produce a measurable total_shift > 0."""
        result = score_authenticity(speaker_a_real_glyph, speaker_a_faked_glyph)
        assert result["total_shift"] > 0.05, (
            f"Total shift too small: {result['total_shift']:.4f}"
        )

    def test_faked_voice_identity_still_matches(
        self, speaker_a_real_glyph, speaker_a_faked_glyph
    ):
        """Despite faking, identity match should remain positive."""
        result = score_authenticity(speaker_a_real_glyph, speaker_a_faked_glyph)
        assert result["identity_match"] > 0.4, (
            f"Identity match too low through fake: {result['identity_match']:.4f}"
        )

    def test_authentic_recording_minimal_shift(
        self, speaker_a_real_glyph, speaker_a_return_glyph
    ):
        """Authentic return visit should have less shift than faked voice."""
        result = score_authenticity(speaker_a_real_glyph, speaker_a_return_glyph)
        assert result["total_shift"] < 0.6, (
            f"Authentic return visit shift too large: {result['total_shift']:.4f}"
        )

    def test_fake_shift_greater_than_natural_variation(
        self, speaker_a_real_glyph, speaker_a_faked_glyph, speaker_a_return_glyph
    ):
        """Faked voice shift should exceed natural session-to-session variation."""
        fake_result = score_authenticity(speaker_a_real_glyph, speaker_a_faked_glyph)
        natural_result = score_authenticity(speaker_a_real_glyph, speaker_a_return_glyph)
        assert fake_result["total_shift"] > natural_result["total_shift"], (
            f"Faked shift ({fake_result['total_shift']:.4f}) should exceed "
            f"natural variation ({natural_result['total_shift']:.4f})"
        )


# ---------------------------------------------------------------------------
# Hierarchy-level assertions
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_cortex_similarity_same_speaker(
        self, speaker_a_real_glyph, speaker_a_return_glyph
    ):
        """Global cortex should show positive similarity for same speaker."""
        sim = cosine_similarity(
            speaker_a_real_glyph.global_cortex.data,
            speaker_a_return_glyph.global_cortex.data,
        )
        assert sim > 0.3, f"Same speaker cortex similarity too low: {sim:.4f}"

    def test_cortex_similarity_different_speakers(
        self, speaker_a_real_glyph, speaker_b_real_glyph
    ):
        """Global cortex should show lower similarity for different speakers."""
        sim = cosine_similarity(
            speaker_a_real_glyph.global_cortex.data,
            speaker_b_real_glyph.global_cortex.data,
        )
        assert sim < 0.6, f"Different speaker cortex similarity too high: {sim:.4f}"

    def test_identity_layer_more_stable_than_state_layers(
        self, speaker_a_real_glyph, speaker_a_faked_glyph
    ):
        """When faking, identity layer should be more stable than state layers."""
        id_sim = cosine_similarity(
            speaker_a_real_glyph.layers["identity"].cortex.data,
            speaker_a_faked_glyph.layers["identity"].cortex.data,
        )
        emo_sim = cosine_similarity(
            speaker_a_real_glyph.layers["emotional_state"].cortex.data,
            speaker_a_faked_glyph.layers["emotional_state"].cortex.data,
        )
        # Identity should be more similar than emotional state when faking
        assert id_sim > emo_sim, (
            f"Identity ({id_sim:.4f}) should be more stable than "
            f"emotional state ({emo_sim:.4f}) when faking"
        )
