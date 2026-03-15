"""
Tests for ContinuousProjector and ContinuousConfig.

Validates:
- Determinism: same input → same output
- Dimensionality: correct output shapes
- Similarity preservation: similar embeddings → similar bipolar vectors
- Orthogonality: random embeddings → near-zero cosine
- Spatial projection: 2D feature maps
- Integration with Encoder via ContinuousConfig
"""

import numpy as np
import pytest

from glyphh.encoder.projection import ContinuousProjector
from glyphh.core.config import ContinuousConfig, EncoderConfig, Role
from glyphh.core.config import Layer as LayerConfig, Segment as SegmentConfig
from glyphh.core.ops import cosine_similarity
from glyphh.encoder.base import Encoder


# ============================================================================
# ContinuousProjector Tests
# ============================================================================

class TestContinuousProjector:
    """Tests for the ContinuousProjector class."""

    def test_basic_projection(self):
        """Project a float vector to bipolar space."""
        projector = ContinuousProjector(source_dim=512, target_dim=10000, seed=42)
        embedding = np.random.RandomState(0).randn(512).astype(np.float32)
        result = projector.project(embedding)

        assert result.shape == (10000,)
        assert result.dtype == np.int8
        assert set(np.unique(result)).issubset({-1, 1})

    def test_determinism(self):
        """Same input always produces same output."""
        embedding = np.random.RandomState(0).randn(512).astype(np.float32)

        p1 = ContinuousProjector(source_dim=512, target_dim=10000, seed=42)
        p2 = ContinuousProjector(source_dim=512, target_dim=10000, seed=42)

        r1 = p1.project(embedding)
        r2 = p2.project(embedding)

        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_different_output(self):
        """Different seeds produce different projections."""
        embedding = np.random.RandomState(0).randn(512).astype(np.float32)

        p1 = ContinuousProjector(source_dim=512, target_dim=10000, seed=42)
        p2 = ContinuousProjector(source_dim=512, target_dim=10000, seed=99)

        r1 = p1.project(embedding)
        r2 = p2.project(embedding)

        assert not np.array_equal(r1, r2)

    def test_similarity_preservation(self):
        """Similar embeddings produce similar bipolar vectors."""
        rng = np.random.RandomState(0)
        base = rng.randn(512).astype(np.float32)
        # Add small perturbation → similar embedding
        similar = base + rng.randn(512).astype(np.float32) * 0.1
        # Completely different embedding
        different = rng.randn(512).astype(np.float32)

        projector = ContinuousProjector(source_dim=512, target_dim=10000, seed=42)

        v_base = projector.project(base)
        v_similar = projector.project(similar)
        v_different = projector.project(different)

        sim_close = cosine_similarity(v_base, v_similar)
        sim_far = cosine_similarity(v_base, v_different)

        # Similar should have much higher cosine than random
        assert sim_close > 0.5, f"Similar embeddings should have cosine > 0.5, got {sim_close}"
        assert sim_close > sim_far, "Similar should be more similar than random"

    def test_orthogonality_of_random(self):
        """Random embeddings produce near-orthogonal bipolar vectors."""
        rng = np.random.RandomState(0)
        projector = ContinuousProjector(source_dim=512, target_dim=10000, seed=42)

        sims = []
        for _ in range(20):
            e1 = rng.randn(512).astype(np.float32)
            e2 = rng.randn(512).astype(np.float32)
            v1 = projector.project(e1)
            v2 = projector.project(e2)
            sim = np.dot(v1.astype(float), v2.astype(float)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            sims.append(abs(sim))

        avg_sim = np.mean(sims)
        assert avg_sim < 0.1, f"Random embeddings should be near-orthogonal, got avg |cosine| = {avg_sim}"

    def test_wrong_dimension_raises(self):
        """Wrong input dimension raises ValueError."""
        projector = ContinuousProjector(source_dim=512, target_dim=10000, seed=42)
        wrong = np.random.randn(256).astype(np.float32)

        with pytest.raises(ValueError, match="Expected embedding of shape"):
            projector.project(wrong)

    def test_matrix_caching(self):
        """Projection matrices are cached across instances."""
        ContinuousProjector.clear_cache()

        p1 = ContinuousProjector(source_dim=128, target_dim=5000, seed=7)
        p2 = ContinuousProjector(source_dim=128, target_dim=5000, seed=7)

        assert p1._projection_matrix is p2._projection_matrix

    def test_small_dimensions(self):
        """Works with small source and target dimensions."""
        projector = ContinuousProjector(source_dim=3, target_dim=16, seed=42)
        embedding = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        result = projector.project(embedding)

        assert result.shape == (16,)
        assert set(np.unique(result)).issubset({-1, 1})


# ============================================================================
# Spatial Projection Tests
# ============================================================================

class TestSpatialProjection:
    """Tests for project_spatial with 2D feature maps."""

    def test_2d_feature_map(self):
        """Project a 2D depth map to bipolar space."""
        projector = ContinuousProjector(source_dim=64, target_dim=10000, seed=42)
        depth_map = np.random.RandomState(0).randn(480, 640).astype(np.float32)
        result = projector.project_spatial(depth_map, grid=(8, 8))

        assert result.shape == (10000,)
        assert result.dtype == np.int8

    def test_3d_feature_map(self):
        """Project a 3D feature map (H, W, C) to bipolar space."""
        # 3 channels, pooled to 4x4 = 48 elements
        projector = ContinuousProjector(source_dim=48, target_dim=10000, seed=42)
        feature_map = np.random.RandomState(0).randn(100, 100, 3).astype(np.float32)
        result = projector.project_spatial(feature_map, grid=(4, 4))

        assert result.shape == (10000,)

    def test_spatial_determinism(self):
        """Same feature map → same projection."""
        projector = ContinuousProjector(source_dim=64, target_dim=10000, seed=42)
        depth_map = np.random.RandomState(0).randn(240, 320).astype(np.float32)

        r1 = projector.project_spatial(depth_map, grid=(8, 8))
        r2 = projector.project_spatial(depth_map, grid=(8, 8))

        np.testing.assert_array_equal(r1, r2)

    def test_spatial_dimension_mismatch(self):
        """Wrong grid size for source_dim raises ValueError."""
        projector = ContinuousProjector(source_dim=64, target_dim=10000, seed=42)
        depth_map = np.random.RandomState(0).randn(100, 100).astype(np.float32)

        # 4x4 grid = 16 elements, but projector expects 64
        with pytest.raises(ValueError, match="Pooled feature map has"):
            projector.project_spatial(depth_map, grid=(4, 4))


# ============================================================================
# ContinuousConfig Tests
# ============================================================================

class TestContinuousConfig:
    """Tests for ContinuousConfig dataclass."""

    def test_valid_config(self):
        """Create a valid ContinuousConfig."""
        config = ContinuousConfig(source_dim=512, projection_seed=100)
        assert config.source_dim == 512
        assert config.projection_seed == 100

    def test_default_seed(self):
        """Default projection_seed is 42."""
        config = ContinuousConfig(source_dim=256)
        assert config.projection_seed == 42

    def test_invalid_source_dim(self):
        """Non-positive source_dim raises ConfigurationException."""
        from glyphh.exceptions import ConfigurationException
        with pytest.raises(ConfigurationException):
            ContinuousConfig(source_dim=0)
        with pytest.raises(ConfigurationException):
            ContinuousConfig(source_dim=-1)

    def test_serialization(self):
        """Round-trip to_dict / from_dict."""
        config = ContinuousConfig(source_dim=512, projection_seed=100)
        d = config.to_dict()
        restored = ContinuousConfig.from_dict(d)

        assert restored.source_dim == config.source_dim
        assert restored.projection_seed == config.projection_seed

    def test_repr(self):
        """String representation."""
        config = ContinuousConfig(source_dim=512, projection_seed=100)
        assert "512" in repr(config)
        assert "100" in repr(config)


# ============================================================================
# Encoder Integration Tests
# ============================================================================

class TestEncoderContinuousIntegration:
    """Tests for continuous encoding via Encoder._encode_continuous_value."""

    def setup_method(self):
        """Create encoder with a continuous role."""
        self.config = EncoderConfig(
            dimension=10000,
            seed=42,
            layers=[
                LayerConfig(
                    name="visual",
                    similarity_weight=1.0,
                    segments=[
                        SegmentConfig(
                            name="features",
                            similarity_weight=1.0,
                            roles=[
                                Role(
                                    name="face_embedding",
                                    continuous_config=ContinuousConfig(
                                        source_dim=512,
                                        projection_seed=100,
                                    ),
                                ),
                                Role(
                                    name="label",
                                    text_encoding="bag_of_words",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        self.encoder = Encoder(self.config)

    def test_encode_with_continuous_role(self):
        """Encode a concept with a continuous vector role."""
        from glyphh.core.types import Concept

        embedding = np.random.RandomState(0).randn(512).tolist()
        concept = Concept(
            name="test_image",
            attributes={
                "face_embedding": embedding,
                "label": "portrait headshot studio",
            },
        )
        glyph = self.encoder.encode(concept)

        assert glyph is not None
        assert "visual" in glyph.layers

    def test_continuous_role_similarity(self):
        """Similar embeddings in continuous roles produce similar glyphs."""
        from glyphh.core.types import Concept

        rng = np.random.RandomState(0)
        base_emb = rng.randn(512)
        similar_emb = base_emb + rng.randn(512) * 0.1
        different_emb = rng.randn(512)

        g1 = self.encoder.encode(Concept(
            name="img1",
            attributes={"face_embedding": base_emb.tolist(), "label": "person"},
        ))
        g2 = self.encoder.encode(Concept(
            name="img2",
            attributes={"face_embedding": similar_emb.tolist(), "label": "person"},
        ))
        g3 = self.encoder.encode(Concept(
            name="img3",
            attributes={"face_embedding": different_emb.tolist(), "label": "person"},
        ))

        sim_close = cosine_similarity(g1.global_cortex.data, g2.global_cortex.data)
        sim_far = cosine_similarity(g1.global_cortex.data, g3.global_cortex.data)

        assert sim_close > sim_far, (
            f"Similar embeddings should produce more similar glyphs: "
            f"close={sim_close:.3f}, far={sim_far:.3f}"
        )

    def test_continuous_fallback_on_invalid(self):
        """Invalid continuous value falls back to symbolic encoding."""
        from glyphh.core.types import Concept

        concept = Concept(
            name="bad",
            attributes={
                "face_embedding": "not_a_vector",
                "label": "test",
            },
        )
        # Should not raise — falls back to symbolic
        glyph = self.encoder.encode(concept)
        assert glyph is not None
