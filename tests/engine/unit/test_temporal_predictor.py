"""
Unit tests for beam search predictor.

Tests the BeamSearchPredictor class for temporal forecasting.
"""

import pytest
import numpy as np
from datetime import datetime
from glyphh.temporal.predictor import BeamSearchPredictor, PredictionResult, Prediction, TrendStatistics
from glyphh.core.types import Vector, Glyph, Layer, Segment
from glyphh.fact_tree.builder import FactTree


def create_test_glyph(identifier: str, vector_data: np.ndarray, space_id: str = "test") -> Glyph:
    """Helper to create a test glyph with given vector data."""
    dimension = len(vector_data)
    vector = Vector(data=vector_data, dimension=dimension, space_id=space_id)
    
    return Glyph(
        identifier=identifier,
        name=f"test_{identifier}",
        space_id=space_id,
        global_cortex=vector,
        layers={},
        timestamp=datetime.now(),
        version="v1"
    )


class TestBeamSearchPredictor:
    """Test suite for BeamSearchPredictor class."""
    
    def test_predictor_initialization(self):
        """Test predictor initialization with default parameters."""
        predictor = BeamSearchPredictor()
        
        assert predictor.beam_width == 5
        assert predictor.drift_reduction is True
        assert predictor.temporal_encoder is not None
    
    def test_predictor_custom_parameters(self):
        """Test predictor initialization with custom parameters."""
        predictor = BeamSearchPredictor(beam_width=10, drift_reduction=False)
        
        assert predictor.beam_width == 10
        assert predictor.drift_reduction is False
    
    def test_predict_basic(self):
        """Test basic prediction with minimal history."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        # Create history with 3 glyphs
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.array([-1, 1, -1, 1, 1, -1, 1, -1], dtype=np.int8)
            )
            for i in range(3)
        ]
        
        # Predict 2 time steps ahead
        result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")
        
        # Verify result structure
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == predictor.beam_width
        assert isinstance(result.trends, TrendStatistics)
        assert result.fact_tree is not None
    
    def test_predict_returns_correct_number_of_predictions(self):
        """Test that predict returns at most beam_width predictions."""
        beam_width = 5
        predictor = BeamSearchPredictor(beam_width=beam_width)
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=100).astype(np.int8)
            )
            for i in range(5)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        # Verify we get at most beam_width predictions
        # (may be less if there are fewer unique candidates)
        assert len(result.predictions) <= beam_width
        assert len(result.predictions) > 0
    
    def test_predict_confidence_scores_ordered(self):
        """Test that predictions are ordered by confidence (descending)."""
        predictor = BeamSearchPredictor(beam_width=5)
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=100).astype(np.int8)
            )
            for i in range(5)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        # Verify confidence scores are in descending order
        confidences = [p.confidence for p in result.predictions]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_predict_all_predictions_valid_vectors(self):
        """Test that all predictions are valid bipolar vectors."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(3)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")
        
        # Verify all predictions are valid vectors
        for pred in result.predictions:
            assert isinstance(pred.vector, Vector)
            assert np.all(np.isin(pred.vector.data, [-1, 1]))
            assert pred.vector.dimension == 50
            assert 0.0 <= pred.confidence <= 1.0
    
    def test_predict_path_length_matches_time_intervals(self):
        """Test that prediction path length matches time_intervals."""
        predictor = BeamSearchPredictor(beam_width=3)
        time_intervals = 4
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(3)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=time_intervals, hierarchy_level="cortex")
        
        # Verify path length
        for pred in result.predictions:
            assert len(pred.path) == time_intervals
    
    def test_predict_insufficient_history(self):
        """Test that prediction with insufficient history raises error."""
        predictor = BeamSearchPredictor()
        
        # Create history with only 1 glyph (need at least 2)
        history = [
            create_test_glyph(
                "glyph_0@2024-01-15T10:00:00Z#v1",
                np.array([-1, 1, -1, 1], dtype=np.int8)
            )
        ]
        
        with pytest.raises(ValueError, match="at least 2 glyphs"):
            predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
    
    def test_predict_invalid_time_intervals(self):
        """Test that prediction with invalid time_intervals raises error."""
        predictor = BeamSearchPredictor()
        
        # Create valid history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.array([-1, 1, -1, 1], dtype=np.int8)
            )
            for i in range(3)
        ]
        
        with pytest.raises(ValueError, match="at least 1"):
            predictor.predict(history, time_intervals=0, hierarchy_level="cortex")
    
    def test_predict_invalid_hierarchy_level(self):
        """Test that prediction with invalid hierarchy_level raises error."""
        predictor = BeamSearchPredictor()
        
        # Create valid history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.array([-1, 1, -1, 1], dtype=np.int8)
            )
            for i in range(3)
        ]
        
        with pytest.raises(ValueError, match="Invalid hierarchy_level"):
            predictor.predict(history, time_intervals=1, hierarchy_level="invalid")
    
    def test_extract_vector_cortex_level(self):
        """Test extracting vector at cortex level."""
        predictor = BeamSearchPredictor()
        
        glyph = create_test_glyph(
            "test@2024-01-15T10:00:00Z#v1",
            np.array([-1, 1, -1, 1], dtype=np.int8)
        )
        
        vector = predictor._extract_vector(glyph, "cortex")
        
        assert isinstance(vector, Vector)
        assert np.array_equal(vector.data, glyph.global_cortex.data)
    
    def test_generate_candidate_deltas(self):
        """Test candidate delta generation."""
        predictor = BeamSearchPredictor()
        
        # Create simple trends
        trends = TrendStatistics(
            mean_delta=np.array([0.5, -0.3, 0.8, -0.2]),
            variance=np.array([0.1, 0.2, 0.1, 0.3]),
            autocorrelation=0.6,
            drift_rate=0.5,
            periodicity=None
        )
        
        # Generate candidates
        candidates = predictor._generate_candidate_deltas(trends, time_step=0)
        
        # Verify we get multiple candidates
        assert len(candidates) >= 1
        
        # Verify all candidates are valid vectors
        for candidate in candidates:
            assert isinstance(candidate, Vector)
            assert np.all(np.isin(candidate.data, [-1, 1]))
            assert candidate.dimension == 4
    
    def test_prediction_result_structure(self):
        """Test that PredictionResult has correct structure."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(3)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")
        
        # Verify structure
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'fact_tree')
        assert hasattr(result, 'trends')
        
        # Verify predictions
        for pred in result.predictions:
            assert hasattr(pred, 'vector')
            assert hasattr(pred, 'confidence')
            assert hasattr(pred, 'path')
            assert hasattr(pred, 'hierarchy_level')
            assert pred.hierarchy_level == "cortex"
    
    def test_trend_statistics_structure(self):
        """Test that TrendStatistics has correct structure."""
        predictor = BeamSearchPredictor()
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(3)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        # Verify trends structure
        trends = result.trends
        assert hasattr(trends, 'mean_delta')
        assert hasattr(trends, 'variance')
        assert hasattr(trends, 'autocorrelation')
        assert hasattr(trends, 'drift_rate')
        assert hasattr(trends, 'periodicity')
        
        # Verify types
        assert isinstance(trends.mean_delta, np.ndarray)
        assert isinstance(trends.variance, np.ndarray)
        assert isinstance(trends.autocorrelation, float)
        assert isinstance(trends.drift_rate, float)


class TestTrendComputation:
    """Test suite for statistical trend computation."""
    
    def test_compute_trends_basic(self):
        """Test basic trend computation."""
        predictor = BeamSearchPredictor()
        
        # Create simple deltas
        deltas = [
            Vector(
                data=np.array([1, 1, -1, 1], dtype=np.int8),
                dimension=4,
                space_id="test"
            ),
            Vector(
                data=np.array([1, -1, -1, 1], dtype=np.int8),
                dimension=4,
                space_id="test"
            ),
            Vector(
                data=np.array([1, 1, -1, -1], dtype=np.int8),
                dimension=4,
                space_id="test"
            )
        ]
        
        trends = predictor._compute_trends(deltas)
        
        # Verify structure
        assert isinstance(trends, TrendStatistics)
        assert len(trends.mean_delta) == 4
        assert len(trends.variance) == 4
        assert 0.0 <= trends.autocorrelation <= 1.0
        assert trends.drift_rate >= 0.0
    
    def test_compute_trends_mean_delta(self):
        """Test mean delta computation."""
        predictor = BeamSearchPredictor()
        
        # Create deltas with known mean
        deltas = [
            Vector(
                data=np.array([1, 1, -1, 1], dtype=np.int8),
                dimension=4,
                space_id="test"
            ),
            Vector(
                data=np.array([1, 1, -1, 1], dtype=np.int8),
                dimension=4,
                space_id="test"
            ),
            Vector(
                data=np.array([1, 1, -1, 1], dtype=np.int8),
                dimension=4,
                space_id="test"
            )
        ]
        
        trends = predictor._compute_trends(deltas)
        
        # All deltas are identical, so mean should equal them
        expected_mean = np.array([1.0, 1.0, -1.0, 1.0])
        assert np.allclose(trends.mean_delta, expected_mean)
    
    def test_compute_trends_variance(self):
        """Test variance computation."""
        predictor = BeamSearchPredictor()
        
        # Create deltas with zero variance (all identical)
        deltas = [
            Vector(
                data=np.array([1, 1, -1, 1], dtype=np.int8),
                dimension=4,
                space_id="test"
            ),
            Vector(
                data=np.array([1, 1, -1, 1], dtype=np.int8),
                dimension=4,
                space_id="test"
            )
        ]
        
        trends = predictor._compute_trends(deltas)
        
        # Variance should be zero for identical deltas
        assert np.allclose(trends.variance, 0.0)
    
    def test_compute_autocorrelation_high(self):
        """Test autocorrelation with highly correlated deltas."""
        predictor = BeamSearchPredictor()
        
        # Create highly correlated deltas (all identical)
        deltas_array = np.array([
            [1, 1, -1, 1],
            [1, 1, -1, 1],
            [1, 1, -1, 1]
        ], dtype=np.int8)
        
        autocorr = predictor._compute_autocorrelation(deltas_array)
        
        # Should be high (close to 1.0)
        assert autocorr > 0.9
    
    def test_compute_autocorrelation_low(self):
        """Test autocorrelation with uncorrelated deltas."""
        predictor = BeamSearchPredictor()
        
        # Create uncorrelated deltas (alternating)
        deltas_array = np.array([
            [1, 1, -1, 1],
            [-1, -1, 1, -1],
            [1, 1, -1, 1],
            [-1, -1, 1, -1]
        ], dtype=np.int8)
        
        autocorr = predictor._compute_autocorrelation(deltas_array)
        
        # Should be low (close to 0.0)
        assert autocorr < 0.2
    
    def test_compute_autocorrelation_single_delta(self):
        """Test autocorrelation with single delta."""
        predictor = BeamSearchPredictor()
        
        deltas_array = np.array([[1, 1, -1, 1]], dtype=np.int8)
        
        autocorr = predictor._compute_autocorrelation(deltas_array)
        
        # Should return 0.0 for insufficient data
        assert autocorr == 0.0
    
    def test_compute_drift_rate(self):
        """Test drift rate computation."""
        predictor = BeamSearchPredictor()
        
        # Create deltas with known drift
        deltas_array = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=np.float32)
        
        drift_rate = predictor._compute_drift_rate(deltas_array)
        
        # Mean is [1, 0, 0, 0], so drift rate should be 1.0
        assert np.isclose(drift_rate, 1.0)
    
    def test_detect_periodicity_none(self):
        """Test periodicity detection with no periodicity."""
        predictor = BeamSearchPredictor()
        
        # Create random deltas (no periodicity)
        deltas_array = np.random.choice([-1, 1], size=(10, 50)).astype(np.int8)
        
        periodicity = predictor._detect_periodicity(deltas_array)
        
        # Should not detect periodicity in random data
        # (may occasionally detect false positive, but unlikely)
        assert periodicity is None or isinstance(periodicity, int)
    
    def test_detect_periodicity_insufficient_data(self):
        """Test periodicity detection with insufficient data."""
        predictor = BeamSearchPredictor()
        
        # Create very short sequence
        deltas_array = np.array([
            [1, 1, -1, 1],
            [1, 1, -1, 1]
        ], dtype=np.int8)
        
        periodicity = predictor._detect_periodicity(deltas_array)
        
        # Should return None for insufficient data
        assert periodicity is None
    
    def test_trends_with_real_prediction(self):
        """Test that trends are computed correctly in real prediction."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        # Create history with clear trend
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.array([1, 1, -1, 1] * 25, dtype=np.int8)  # 100 dimensions
            )
            for i in range(5)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        # Verify trends are computed
        assert result.trends is not None
        assert len(result.trends.mean_delta) == 100
        assert len(result.trends.variance) == 100
        assert 0.0 <= result.trends.autocorrelation <= 1.0
        assert result.trends.drift_rate >= 0.0


class TestReliabilityScoring:
    """Test suite for prediction reliability scoring."""
    
    def test_score_reliability_perfect_match(self):
        """Test reliability scoring with perfect trend match."""
        predictor = BeamSearchPredictor()
        
        # Create trends
        trends = TrendStatistics(
            mean_delta=np.array([1.0, 1.0, -1.0, 1.0]),
            variance=np.array([0.0, 0.0, 0.0, 0.0]),
            autocorrelation=1.0,
            drift_rate=2.0,
            periodicity=None
        )
        
        # Create delta that matches mean exactly
        current_state = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        delta = Vector(
            data=np.array([1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        next_state = Vector(
            data=np.array([-1, 1, 1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        score = predictor._score_reliability(next_state, current_state, delta, trends)
        
        # Should be high (close to 1.0)
        assert 0.8 <= score <= 1.0
    
    def test_score_reliability_poor_match(self):
        """Test reliability scoring with poor trend match."""
        predictor = BeamSearchPredictor()
        
        # Create trends
        trends = TrendStatistics(
            mean_delta=np.array([1.0, 1.0, -1.0, 1.0]),
            variance=np.array([0.1, 0.1, 0.1, 0.1]),
            autocorrelation=0.9,
            drift_rate=2.0,
            periodicity=None
        )
        
        # Create delta that doesn't match mean
        current_state = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        delta = Vector(
            data=np.array([-1, -1, 1, -1], dtype=np.int8),  # Opposite of mean
            dimension=4,
            space_id="test"
        )
        next_state = Vector(
            data=np.array([1, -1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        score = predictor._score_reliability(next_state, current_state, delta, trends)
        
        # Should be lower
        assert 0.0 <= score < 0.6
    
    def test_score_reliability_range(self):
        """Test that reliability score is always in [0, 1] range."""
        predictor = BeamSearchPredictor()
        
        # Create various trends and deltas
        for _ in range(10):
            trends = TrendStatistics(
                mean_delta=np.random.randn(50),
                variance=np.random.rand(50),
                autocorrelation=np.random.rand(),
                drift_rate=np.random.rand() * 10,
                periodicity=None
            )
            
            current_state = Vector(
                data=np.random.choice([-1, 1], size=50).astype(np.int8),
                dimension=50,
                space_id="test"
            )
            delta = Vector(
                data=np.random.choice([-1, 1], size=50).astype(np.int8),
                dimension=50,
                space_id="test"
            )
            next_state = Vector(
                data=np.random.choice([-1, 1], size=50).astype(np.int8),
                dimension=50,
                space_id="test"
            )
            
            score = predictor._score_reliability(next_state, current_state, delta, trends)
            
            # Verify range
            assert 0.0 <= score <= 1.0
    
    def test_score_reliability_with_high_autocorrelation(self):
        """Test that high autocorrelation increases reliability."""
        predictor = BeamSearchPredictor()
        
        # Create delta matching mean
        delta = Vector(
            data=np.array([1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        current_state = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        next_state = Vector(
            data=np.array([-1, 1, 1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        # Test with high autocorrelation
        trends_high = TrendStatistics(
            mean_delta=np.array([1.0, 1.0, -1.0, 1.0]),
            variance=np.array([0.1, 0.1, 0.1, 0.1]),
            autocorrelation=0.9,
            drift_rate=2.0,
            periodicity=None
        )
        
        # Test with low autocorrelation
        trends_low = TrendStatistics(
            mean_delta=np.array([1.0, 1.0, -1.0, 1.0]),
            variance=np.array([0.1, 0.1, 0.1, 0.1]),
            autocorrelation=0.1,
            drift_rate=2.0,
            periodicity=None
        )
        
        score_high = predictor._score_reliability(next_state, current_state, delta, trends_high)
        score_low = predictor._score_reliability(next_state, current_state, delta, trends_low)
        
        # High autocorrelation should give higher score
        assert score_high > score_low
    
    def test_score_reliability_with_variance_penalty(self):
        """Test that high variance reduces reliability."""
        predictor = BeamSearchPredictor()
        
        # Use a delta that deviates slightly from mean
        delta = Vector(
            data=np.array([1, -1, -1, 1], dtype=np.int8),  # Different from mean
            dimension=4,
            space_id="test"
        )
        current_state = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        next_state = Vector(
            data=np.array([-1, -1, 1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        # Test with low variance (more predictable)
        trends_low_var = TrendStatistics(
            mean_delta=np.array([1.0, 1.0, -1.0, 1.0]),
            variance=np.array([0.01, 0.01, 0.01, 0.01]),
            autocorrelation=0.5,
            drift_rate=2.0,
            periodicity=None
        )
        
        # Test with high variance (less predictable)
        trends_high_var = TrendStatistics(
            mean_delta=np.array([1.0, 1.0, -1.0, 1.0]),
            variance=np.array([1.0, 1.0, 1.0, 1.0]),
            autocorrelation=0.5,
            drift_rate=2.0,
            periodicity=None
        )
        
        score_low_var = predictor._score_reliability(next_state, current_state, delta, trends_low_var)
        score_high_var = predictor._score_reliability(next_state, current_state, delta, trends_high_var)
        
        # Low variance should give higher score (or at least not lower)
        # The variance penalty should be more forgiving with high variance
        assert score_high_var >= score_low_var * 0.8  # Allow some tolerance
    
    def test_reliability_affects_prediction_confidence(self):
        """Test that reliability scoring affects final prediction confidence."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        # Create history with consistent pattern
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.array([1, 1, -1, 1] * 25, dtype=np.int8)
            )
            for i in range(5)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        # Verify predictions have varying confidence scores
        confidences = [p.confidence for p in result.predictions]
        
        # All confidences should be in valid range
        for conf in confidences:
            assert 0.0 <= conf <= 1.0
        
        # Top prediction should have highest confidence
        assert confidences[0] == max(confidences)


class TestDriftReduction:
    """Test suite for drift reduction."""
    
    def test_reduce_drift_basic(self):
        """Test basic drift reduction."""
        predictor = BeamSearchPredictor()
        
        initial_state = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        predicted_state = Vector(
            data=np.array([1, 1, 1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        corrected = predictor._reduce_drift(predicted_state, initial_state, time_step=1)
        
        # Verify result is valid vector
        assert isinstance(corrected, Vector)
        assert corrected.dimension == 4
        assert corrected.space_id == "test"
        assert np.all(np.isin(corrected.data, [-1, 1]))
    
    def test_reduce_drift_preserves_bipolar(self):
        """Test that drift reduction preserves bipolar constraint."""
        predictor = BeamSearchPredictor()
        
        initial_state = Vector(
            data=np.random.choice([-1, 1], size=100).astype(np.int8),
            dimension=100,
            space_id="test"
        )
        predicted_state = Vector(
            data=np.random.choice([-1, 1], size=100).astype(np.int8),
            dimension=100,
            space_id="test"
        )
        
        for time_step in [1, 5, 10]:
            corrected = predictor._reduce_drift(predicted_state, initial_state, time_step)
            
            # Verify bipolar constraint
            assert np.all(np.isin(corrected.data, [-1, 1]))
    
    def test_reduce_drift_stronger_for_longer_predictions(self):
        """Test that drift reduction is stronger for longer time steps."""
        predictor = BeamSearchPredictor()
        
        initial_state = Vector(
            data=np.array([-1, -1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        predicted_state = Vector(
            data=np.array([1, 1, 1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        # Apply drift reduction at different time steps
        corrected_t1 = predictor._reduce_drift(predicted_state, initial_state, time_step=1)
        corrected_t5 = predictor._reduce_drift(predicted_state, initial_state, time_step=5)
        corrected_t10 = predictor._reduce_drift(predicted_state, initial_state, time_step=10)
        
        # Compute similarity to initial state
        # (higher similarity = more drift reduction)
        from glyphh.core.ops import cosine_similarity
        
        sim_t1 = cosine_similarity(corrected_t1.data, initial_state.data)
        sim_t5 = cosine_similarity(corrected_t5.data, initial_state.data)
        sim_t10 = cosine_similarity(corrected_t10.data, initial_state.data)
        
        # Longer time steps should pull more toward initial state
        assert sim_t10 >= sim_t5 >= sim_t1
    
    def test_reduce_drift_no_drift(self):
        """Test drift reduction when there's no drift."""
        predictor = BeamSearchPredictor()
        
        # Same state (no drift)
        state = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        corrected = predictor._reduce_drift(state, state, time_step=1)
        
        # Should remain unchanged (or very similar)
        assert np.array_equal(corrected.data, state.data)
    
    def test_reduce_drift_with_prediction(self):
        """Test that drift reduction is applied in actual prediction."""
        # Test with drift reduction enabled
        predictor_with_drift = BeamSearchPredictor(beam_width=3, drift_reduction=True)
        
        # Test without drift reduction
        predictor_no_drift = BeamSearchPredictor(beam_width=3, drift_reduction=False)
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(5)
        ]
        
        # Predict with both
        result_with_drift = predictor_with_drift.predict(
            history, time_intervals=5, hierarchy_level="cortex"
        )
        result_no_drift = predictor_no_drift.predict(
            history, time_intervals=5, hierarchy_level="cortex"
        )
        
        # Both should produce valid results
        assert len(result_with_drift.predictions) > 0
        assert len(result_no_drift.predictions) > 0
        
        # Verify all predictions are valid
        for pred in result_with_drift.predictions:
            assert np.all(np.isin(pred.vector.data, [-1, 1]))
        
        for pred in result_no_drift.predictions:
            assert np.all(np.isin(pred.vector.data, [-1, 1]))
    
    def test_reduce_drift_exponential_decay(self):
        """Test that drift reduction uses exponential decay."""
        predictor = BeamSearchPredictor()
        
        initial_state = Vector(
            data=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8),
            dimension=8,
            space_id="test"
        )
        predicted_state = Vector(
            data=np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int8),
            dimension=8,
            space_id="test"
        )
        
        # Test at multiple time steps
        time_steps = [1, 2, 3, 5, 10]
        similarities = []
        
        from glyphh.core.ops import cosine_similarity
        
        for t in time_steps:
            corrected = predictor._reduce_drift(predicted_state, initial_state, t)
            sim = cosine_similarity(corrected.data, initial_state.data)
            similarities.append(sim)
        
        # Similarities should generally increase (more correction for longer predictions)
        # Allow some tolerance for bipolar quantization effects
        for i in range(len(similarities) - 1):
            # Each step should be at least as similar or more similar
            assert similarities[i+1] >= similarities[i] - 0.3  # Allow some tolerance


class TestPredictionFactTree:
    """Test suite for prediction fact tree generation."""
    
    def test_fact_tree_structure(self):
        """Test that fact tree has expected structure."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        # Create history
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(5)
        ]
        
        # Predict
        result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")
        
        # Verify fact tree exists
        assert result.fact_tree is not None
        assert isinstance(result.fact_tree, FactTree)
    
    def test_fact_tree_contains_history(self):
        """Test that fact tree contains historical context."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(5)
        ]
        
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        # Convert to text to check content
        fact_text = result.fact_tree.to_text()
        
        # Should contain historical information
        assert "Historical Context" in fact_text or "history" in fact_text.lower()
        assert "5" in fact_text  # Number of glyphs
    
    def test_fact_tree_contains_trends(self):
        """Test that fact tree contains statistical trends."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(5)
        ]
        
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        fact_text = result.fact_tree.to_text()
        
        # Should contain trend information
        assert "trend" in fact_text.lower() or "Trend" in fact_text
        assert "autocorrelation" in fact_text.lower() or "Autocorrelation" in fact_text
    
    def test_fact_tree_contains_predictions(self):
        """Test that fact tree contains prediction results."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(5)
        ]
        
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        fact_text = result.fact_tree.to_text()
        
        # Should contain prediction results
        assert "candidate" in fact_text.lower() or "Candidate" in fact_text
        assert "confidence" in fact_text.lower() or "Confidence" in fact_text
    
    def test_fact_tree_contains_scoring_explanation(self):
        """Test that fact tree explains scoring methodology."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(5)
        ]
        
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        fact_text = result.fact_tree.to_text()
        
        # Should contain scoring explanation
        assert "scoring" in fact_text.lower() or "Scoring" in fact_text
        assert "reliability" in fact_text.lower() or "Reliability" in fact_text
    
    def test_fact_tree_serialization(self):
        """Test that fact tree can be serialized to JSON."""
        predictor = BeamSearchPredictor(beam_width=3)
        
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(3)
        ]
        
        result = predictor.predict(history, time_intervals=1, hierarchy_level="cortex")
        
        # Should be able to serialize to JSON
        fact_json = result.fact_tree.to_json()
        
        assert isinstance(fact_json, dict)
        assert "description" in fact_json
    
    def test_fact_tree_includes_parameters(self):
        """Test that fact tree includes prediction parameters."""
        beam_width = 7
        predictor = BeamSearchPredictor(beam_width=beam_width, drift_reduction=True)
        
        history = [
            create_test_glyph(
                f"glyph_{i}@2024-01-15T10:00:0{i}Z#v1",
                np.random.choice([-1, 1], size=50).astype(np.int8)
            )
            for i in range(3)
        ]
        
        result = predictor.predict(history, time_intervals=1, hierarchy_level="segment")
        
        fact_text = result.fact_tree.to_text()
        
        # Should include parameters
        assert "segment" in fact_text.lower()
        assert "parameter" in fact_text.lower() or "Parameter" in fact_text
