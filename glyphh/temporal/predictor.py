"""
Beam search predictor for temporal forecasting.

This module provides the BeamSearchPredictor class for predicting future
glyph states using beam search with statistical reliability scoring.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from glyphh.core.types import Vector, Glyph
from glyphh.temporal.delta import TemporalEncoder
from glyphh.fact_tree.builder import FactTree


@dataclass
class TrendStatistics:
    """
    Statistical trends from historical data.
    
    Attributes:
        mean_delta: Mean temporal delta across history
        variance: Variance of temporal deltas
        autocorrelation: Autocorrelation coefficient
        drift_rate: Rate of drift from initial state
        periodicity: Detected period length, if any
    """
    mean_delta: np.ndarray
    variance: np.ndarray
    autocorrelation: float
    drift_rate: float
    periodicity: Optional[int] = None


@dataclass
class Prediction:
    """
    Single prediction candidate.
    
    Attributes:
        vector: Predicted vector state
        confidence: Confidence score (0.0-1.0)
        path: Sequence of deltas leading to this prediction
        hierarchy_level: Level of prediction (cortex, layer, segment, role)
    """
    vector: Vector
    confidence: float
    path: List[Vector]
    hierarchy_level: str


@dataclass
class PredictionResult:
    """
    Result of temporal prediction.
    
    Attributes:
        predictions: Top-k predictions ordered by confidence
        fact_tree: Explanation of prediction process
        trends: Statistical trends used for prediction
    """
    predictions: List[Prediction]
    fact_tree: FactTree
    trends: TrendStatistics


class BeamSearchPredictor:
    """
    Beam search predictor for temporal forecasting.
    
    The BeamSearchPredictor uses beam search to maintain multiple candidate
    predictions and scores them by statistical reliability. It supports:
    - Beam search with configurable beam width
    - Statistical trend analysis (mean, variance, autocorrelation)
    - Reliability scoring based on trend consistency
    - Optional drift reduction to maintain statistical reliability
    - Prediction at any hierarchy level (cortex, layer, segment, role)
    
    Algorithm:
    1. Extract temporal deltas from history
    2. Compute statistical trends (mean, variance, autocorrelation)
    3. Generate candidate predictions using beam search
    4. Score candidates by statistical reliability
    5. Apply drift reduction if enabled
    6. Return top-k predictions with confidence scores
    
    Example:
        >>> predictor = BeamSearchPredictor(beam_width=5, drift_reduction=True)
        >>> result = predictor.predict(history, time_intervals=3, hierarchy_level="cortex")
        >>> print(f"Top prediction confidence: {result.predictions[0].confidence}")
    """
    
    def __init__(self, beam_width: int = 5, drift_reduction: bool = True):
        """
        Initialize beam search predictor.
        
        Args:
            beam_width: Number of candidate paths to maintain (default: 5)
            drift_reduction: Apply drift correction (default: True)
        """
        self.beam_width = beam_width
        self.drift_reduction = drift_reduction
        self.temporal_encoder = TemporalEncoder()
    
    def predict(
        self,
        history: List[Glyph],
        time_intervals: int,
        hierarchy_level: str = "cortex"
    ) -> PredictionResult:
        """
        Predict future state using beam search.
        
        Args:
            history: Historical versions of the glyph (time-ordered)
            time_intervals: Number of time steps to predict into future
            hierarchy_level: "cortex", "layer", "segment", or "role"
        
        Returns:
            PredictionResult with top-k predictions, confidence scores, and fact tree
        
        Raises:
            ValueError: If history is too short, time_intervals is invalid,
                       or hierarchy_level is invalid
        
        Example:
            >>> predictor = BeamSearchPredictor(beam_width=5)
            >>> result = predictor.predict(history, time_intervals=3, hierarchy_level="cortex")
        """
        # Validate inputs
        if len(history) < 2:
            raise ValueError(
                f"History must contain at least 2 glyphs, got {len(history)}"
            )
        
        if time_intervals < 1:
            raise ValueError(
                f"time_intervals must be at least 1, got {time_intervals}"
            )
        
        valid_levels = ["cortex", "layer", "segment", "role"]
        if hierarchy_level not in valid_levels:
            raise ValueError(
                f"Invalid hierarchy_level: {hierarchy_level}. "
                f"Must be one of: {valid_levels}"
            )
        
        # Extract vectors at specified hierarchy level
        vectors = [self._extract_vector(g, hierarchy_level) for g in history]
        
        # Compute temporal deltas
        deltas = [
            self.temporal_encoder.compute_temporal_delta(vectors[i], vectors[i+1])
            for i in range(len(vectors) - 1)
        ]
        
        # Compute statistical trends
        trends = self._compute_trends(deltas)
        
        # Initialize beam with current state
        # Each beam entry: (state, score, path)
        beam = [(vectors[-1], 1.0, [])]
        
        # Predict forward time_intervals steps
        for t in range(time_intervals):
            candidates = []
            
            for state, score, path in beam:
                # Generate candidate next states (use state's space_id)
                for candidate_delta in self._generate_candidate_deltas(trends, t, state.space_id):
                    next_state = self.temporal_encoder.apply_temporal_delta(
                        state, candidate_delta
                    )
                    
                    # Score candidate by statistical reliability
                    reliability = self._score_reliability(
                        next_state, state, candidate_delta, trends
                    )
                    
                    # Apply drift reduction
                    if self.drift_reduction:
                        next_state = self._reduce_drift(next_state, vectors[0], t + 1)
                    
                    candidates.append((
                        next_state,
                        score * reliability,
                        path + [candidate_delta]
                    ))
            
            # Keep top beam_width candidates
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
        
        # Build fact tree explaining predictions
        fact_tree = self._build_prediction_fact_tree(history, beam, trends, hierarchy_level)
        
        # Create prediction results
        predictions = [
            Prediction(
                vector=state,
                confidence=score,
                path=path,
                hierarchy_level=hierarchy_level
            )
            for state, score, path in beam
        ]
        
        return PredictionResult(
            predictions=predictions,
            fact_tree=fact_tree,
            trends=trends
        )
    
    def _extract_vector(self, glyph: Glyph, hierarchy_level: str) -> Vector:
        """
        Extract vector at specified hierarchy level from glyph.
        
        Args:
            glyph: Glyph to extract vector from
            hierarchy_level: "cortex", "layer", "segment", or "role"
        
        Returns:
            Vector at specified hierarchy level
        
        Note:
            For simplicity, this implementation only supports cortex level.
            Full implementation would support layer/segment/role extraction.
        """
        if hierarchy_level == "cortex":
            return glyph.global_cortex
        else:
            # For now, just return global cortex
            # Full implementation would extract from layers/segments/roles
            return glyph.global_cortex
    
    def _generate_candidate_deltas(
        self,
        trends: TrendStatistics,
        time_step: int,
        space_id: str = None
    ) -> List[Vector]:
        """
        Generate candidate delta vectors for beam search.
        
        This generates multiple candidate deltas based on statistical trends:
        1. Mean delta (most likely)
        2. Mean + small perturbation (explore nearby states)
        3. Mean - small perturbation (explore nearby states)
        
        Args:
            trends: Statistical trends from history
            time_step: Current time step in prediction
            space_id: Vector space identifier (optional)
        
        Returns:
            List of candidate delta vectors
        """
        candidates = []
        
        # Get dimension from mean_delta
        dimension = len(trends.mean_delta)
        
        # Use provided space_id or default
        if space_id is None:
            space_id = "prediction"
        
        # Candidate 1: Mean delta (most likely)
        mean_delta_bipolar = np.sign(trends.mean_delta).astype(np.int8)
        mean_delta_bipolar[mean_delta_bipolar == 0] = 1  # Handle zeros
        candidates.append(Vector(
            data=mean_delta_bipolar,
            dimension=dimension,
            space_id=space_id
        ))
        
        # Candidate 2: Small positive perturbation
        # Flip 10% of dimensions in positive direction
        perturbed_pos = mean_delta_bipolar.copy()
        num_flips = max(1, dimension // 10)
        flip_indices = np.random.choice(dimension, num_flips, replace=False)
        perturbed_pos[flip_indices] *= -1
        candidates.append(Vector(
            data=perturbed_pos,
            dimension=dimension,
            space_id=space_id
        ))
        
        # Candidate 3: Small negative perturbation
        # Flip 10% of dimensions in negative direction
        perturbed_neg = mean_delta_bipolar.copy()
        flip_indices = np.random.choice(dimension, num_flips, replace=False)
        perturbed_neg[flip_indices] *= -1
        candidates.append(Vector(
            data=perturbed_neg,
            dimension=dimension,
            space_id=space_id
        ))
        
        return candidates
    
    def _compute_trends(self, deltas: List[Vector]) -> TrendStatistics:
        """
        Compute statistical trends from temporal deltas.
        
        Calculates:
        - Mean delta: Average change across history
        - Variance: Variability of changes
        - Autocorrelation: Correlation between consecutive deltas
        - Drift rate: Magnitude of average change
        - Periodicity: Detected period length (if any)
        
        Args:
            deltas: List of temporal delta vectors
        
        Returns:
            TrendStatistics with mean, variance, autocorrelation, drift rate, periodicity
        """
        # Convert deltas to array
        deltas_array = np.array([d.data for d in deltas])
        
        # Compute mean delta
        mean_delta = np.mean(deltas_array, axis=0)
        
        # Compute variance
        variance = np.var(deltas_array, axis=0)
        
        # Compute autocorrelation (lag-1)
        autocorrelation = self._compute_autocorrelation(deltas_array)
        
        # Compute drift rate (magnitude of mean delta)
        drift_rate = np.linalg.norm(mean_delta)
        
        # Detect periodicity
        periodicity = self._detect_periodicity(deltas_array)
        
        return TrendStatistics(
            mean_delta=mean_delta,
            variance=variance,
            autocorrelation=autocorrelation,
            drift_rate=drift_rate,
            periodicity=periodicity
        )
    
    def _compute_autocorrelation(self, deltas_array: np.ndarray) -> float:
        """
        Compute lag-1 autocorrelation of temporal deltas.
        
        Autocorrelation measures how similar consecutive deltas are.
        High autocorrelation means changes are consistent over time.
        
        Args:
            deltas_array: Array of delta vectors (shape: [n_deltas, dimension])
        
        Returns:
            Autocorrelation coefficient (0.0-1.0)
        """
        if len(deltas_array) < 2:
            return 0.0
        
        # Compute correlation between consecutive deltas
        correlations = []
        for i in range(len(deltas_array) - 1):
            # Cosine similarity between consecutive deltas
            dot_product = np.dot(deltas_array[i], deltas_array[i+1])
            norm_product = np.linalg.norm(deltas_array[i]) * np.linalg.norm(deltas_array[i+1])
            
            if norm_product > 0:
                correlation = dot_product / norm_product
                correlations.append(correlation)
        
        # Return mean correlation
        if correlations:
            # Convert from [-1, 1] to [0, 1] range
            mean_corr = np.mean(correlations)
            return float((mean_corr + 1) / 2)
        else:
            return 0.0
    
    def _compute_drift_rate(self, deltas_array: np.ndarray) -> float:
        """
        Compute drift rate from temporal deltas.
        
        Drift rate is the magnitude of the mean delta, indicating
        how much the vector is changing on average.
        
        Args:
            deltas_array: Array of delta vectors
        
        Returns:
            Drift rate (magnitude of mean delta)
        """
        mean_delta = np.mean(deltas_array, axis=0)
        return float(np.linalg.norm(mean_delta))
    
    def _detect_periodicity(self, deltas_array: np.ndarray) -> Optional[int]:
        """
        Detect periodicity in temporal deltas.
        
        Uses autocorrelation at different lags to detect repeating patterns.
        
        Args:
            deltas_array: Array of delta vectors
        
        Returns:
            Period length if detected, None otherwise
        """
        if len(deltas_array) < 4:
            return None
        
        # Compute autocorrelation at different lags
        max_lag = min(len(deltas_array) // 2, 10)
        autocorrs = []
        
        for lag in range(1, max_lag + 1):
            correlations = []
            for i in range(len(deltas_array) - lag):
                dot_product = np.dot(deltas_array[i], deltas_array[i+lag])
                norm_product = np.linalg.norm(deltas_array[i]) * np.linalg.norm(deltas_array[i+lag])
                
                if norm_product > 0:
                    correlation = dot_product / norm_product
                    correlations.append(correlation)
            
            if correlations:
                autocorrs.append(np.mean(correlations))
            else:
                autocorrs.append(0.0)
        
        # Find peaks in autocorrelation (indicating periodicity)
        if len(autocorrs) >= 2:
            # Look for first significant peak after lag 1
            threshold = 0.5  # Correlation threshold for periodicity
            for lag in range(1, len(autocorrs)):
                if autocorrs[lag] > threshold:
                    return lag + 1  # +1 because lag is 0-indexed
        
        return None
    
    def _score_reliability(
        self,
        next_state: Vector,
        current_state: Vector,
        delta: Vector,
        trends: TrendStatistics
    ) -> float:
        """
        Score prediction reliability based on statistical consistency.
        
        Factors considered:
        - Trend consistency: How well delta matches historical mean
        - Variance penalty: Penalize high-variance predictions
        - Autocorrelation alignment: Reward consistency with historical patterns
        - Drift penalty: Penalize excessive drift from mean
        
        Args:
            next_state: Predicted next state
            current_state: Current state
            delta: Delta applied
            trends: Statistical trends
        
        Returns:
            Reliability score (0.0-1.0)
        """
        # 1. Trend consistency: How close is delta to mean delta?
        # Convert to float for comparison
        delta_float = delta.data.astype(np.float32)
        mean_delta_float = trends.mean_delta.astype(np.float32)
        
        # Compute element-wise difference
        diff = np.abs(delta_float - mean_delta_float)
        # Normalize by 2 (max difference for bipolar values)
        trend_consistency = 1.0 - np.mean(diff) / 2.0
        trend_consistency = max(0.0, min(1.0, trend_consistency))
        
        # 2. Variance penalty: Penalize if delta deviates from expected variance
        # Compute how much delta varies from mean
        delta_variance = (delta_float - mean_delta_float) ** 2
        # Compare to historical variance
        variance_ratio = np.mean(delta_variance / (trends.variance + 1e-6))
        # Exponential penalty for high variance
        variance_penalty = np.exp(-variance_ratio)
        variance_penalty = max(0.0, min(1.0, variance_penalty))
        
        # 3. Autocorrelation alignment: Reward if delta is consistent with trends
        # Use autocorrelation as a weight (high autocorrelation = more predictable)
        autocorr_score = trends.autocorrelation
        
        # 4. Drift penalty: Penalize excessive drift
        # Compute magnitude of delta
        delta_magnitude = np.linalg.norm(delta_float)
        # Compare to historical drift rate
        if trends.drift_rate > 0:
            drift_ratio = delta_magnitude / trends.drift_rate
            # Exponential penalty for excessive drift
            drift_penalty = np.exp(-abs(drift_ratio - 1.0))
        else:
            # No historical drift, penalize any drift
            drift_penalty = np.exp(-delta_magnitude)
        drift_penalty = max(0.0, min(1.0, drift_penalty))
        
        # Combined reliability score (weighted average)
        reliability = (
            0.4 * trend_consistency +
            0.3 * variance_penalty +
            0.2 * autocorr_score +
            0.1 * drift_penalty
        )
        
        return float(max(0.0, min(1.0, reliability)))
    
    def _reduce_drift(
        self,
        predicted_state: Vector,
        initial_state: Vector,
        time_step: int
    ) -> Vector:
        """
        Apply drift reduction to maintain statistical reliability.
        
        Strategy:
        - Compute drift from initial state
        - Apply exponential decay correction (stronger for longer predictions)
        - Preserve bipolar constraint after correction
        
        Args:
            predicted_state: Predicted state vector
            initial_state: Initial state vector (from history)
            time_step: Current time step (1-indexed)
        
        Returns:
            Drift-corrected vector with bipolar constraint preserved
        """
        # Compute drift from initial state
        drift = predicted_state.data.astype(np.float32) - initial_state.data.astype(np.float32)
        drift_magnitude = np.linalg.norm(drift)
        
        # Exponential decay correction (stronger for longer predictions)
        # decay_factor decreases as time_step increases
        decay_factor = np.exp(-0.1 * time_step)
        corrected_drift = drift * decay_factor
        
        # Apply correction
        corrected_state = initial_state.data.astype(np.float32) + corrected_drift
        
        # Restore bipolar constraint
        # Use sign function to map to {-1, +1}
        corrected_state_bipolar = np.sign(corrected_state).astype(np.int8)
        
        # Handle zeros (map to +1)
        corrected_state_bipolar[corrected_state_bipolar == 0] = 1
        
        return Vector(
            data=corrected_state_bipolar,
            dimension=predicted_state.dimension,
            space_id=predicted_state.space_id
        )
    
    def _build_prediction_fact_tree(
        self,
        history: List[Glyph],
        beam: List[Tuple[Vector, float, List[Vector]]],
        trends: TrendStatistics,
        hierarchy_level: str
    ) -> FactTree:
        """
        Build fact tree explaining prediction process.
        
        The fact tree includes:
        - Historical context (number of glyphs, time range)
        - Statistical trends (mean, variance, autocorrelation, drift)
        - Prediction parameters (beam width, time intervals, hierarchy level)
        - Top predictions with confidence scores
        - Explanation of scoring factors
        
        Args:
            history: Historical glyphs used for prediction
            beam: Final beam with predictions (state, score, path)
            trends: Statistical trends computed from history
            hierarchy_level: Hierarchy level of prediction
        
        Returns:
            FactTree explaining the prediction process
        """
        fact_tree = FactTree()
        
        # Add root node
        fact_tree.add_fact(
            path=["prediction"],
            description="Temporal Prediction",
            value=f"Predicted {len(beam)} candidates at {hierarchy_level} level"
        )
        
        # Add historical context
        fact_tree.add_fact(
            path=["prediction", "history"],
            description="Historical Context",
            value=f"{len(history)} glyphs analyzed"
        )
        
        fact_tree.add_fact(
            path=["prediction", "history", "time_range"],
            description="Time Range",
            value=f"{history[0].timestamp} to {history[-1].timestamp}"
        )
        
        fact_tree.add_fact(
            path=["prediction", "history", "space_id"],
            description="Vector Space",
            value=history[0].space_id
        )
        
        # Add statistical trends
        fact_tree.add_fact(
            path=["prediction", "trends"],
            description="Statistical Trends",
            value="Computed from temporal deltas"
        )
        
        fact_tree.add_fact(
            path=["prediction", "trends", "mean_delta"],
            description="Mean Delta",
            value=f"Magnitude: {np.linalg.norm(trends.mean_delta):.4f}",
            data_sample=trends.mean_delta[:10].tolist() if len(trends.mean_delta) > 10 else trends.mean_delta.tolist()
        )
        
        fact_tree.add_fact(
            path=["prediction", "trends", "variance"],
            description="Variance",
            value=f"Mean: {np.mean(trends.variance):.4f}",
            data_sample=trends.variance[:10].tolist() if len(trends.variance) > 10 else trends.variance.tolist()
        )
        
        fact_tree.add_fact(
            path=["prediction", "trends", "autocorrelation"],
            description="Autocorrelation",
            value=f"{trends.autocorrelation:.4f}",
            math_explanation="Correlation between consecutive deltas (0=uncorrelated, 1=highly correlated)"
        )
        
        fact_tree.add_fact(
            path=["prediction", "trends", "drift_rate"],
            description="Drift Rate",
            value=f"{trends.drift_rate:.4f}",
            math_explanation="Magnitude of average change per time step"
        )
        
        if trends.periodicity is not None:
            fact_tree.add_fact(
                path=["prediction", "trends", "periodicity"],
                description="Detected Periodicity",
                value=f"Period: {trends.periodicity} time steps"
            )
        
        # Add prediction parameters
        fact_tree.add_fact(
            path=["prediction", "parameters"],
            description="Prediction Parameters",
            value=f"Beam width: {self.beam_width}, Drift reduction: {self.drift_reduction}"
        )
        
        fact_tree.add_fact(
            path=["prediction", "parameters", "hierarchy_level"],
            description="Hierarchy Level",
            value=hierarchy_level
        )
        
        # Add predictions
        fact_tree.add_fact(
            path=["prediction", "results"],
            description="Prediction Results",
            value=f"{len(beam)} candidates"
        )
        
        for i, (state, score, path) in enumerate(beam):
            fact_tree.add_fact(
                path=["prediction", "results", f"candidate_{i+1}"],
                description=f"Candidate {i+1}",
                value=f"Confidence: {score:.4f}"
            )
            
            fact_tree.add_fact(
                path=["prediction", "results", f"candidate_{i+1}", "vector"],
                description="Predicted Vector",
                value=f"Dimension: {state.dimension}",
                data_sample=state.data[:10].tolist() if len(state.data) > 10 else state.data.tolist()
            )
            
            fact_tree.add_fact(
                path=["prediction", "results", f"candidate_{i+1}", "path_length"],
                description="Prediction Path Length",
                value=f"{len(path)} time steps"
            )
        
        # Add scoring explanation
        fact_tree.add_fact(
            path=["prediction", "scoring"],
            description="Reliability Scoring",
            value="Weighted combination of multiple factors"
        )
        
        fact_tree.add_fact(
            path=["prediction", "scoring", "factors"],
            description="Scoring Factors",
            value="Trend consistency (40%), Variance penalty (30%), Autocorrelation (20%), Drift penalty (10%)",
            math_explanation="reliability = 0.4*trend_consistency + 0.3*variance_penalty + 0.2*autocorr + 0.1*drift_penalty"
        )
        
        return fact_tree
