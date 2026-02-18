"""
Custom encoder configuration for the assistant model.

If this file exists in a model directory, the runtime imports
ENCODER_CONFIG from it instead of building one from config.yaml.

The assistant uses a router-style encoder with intent/action/context
segments — a different shape than the default flat encoder.
"""

from glyphh.core.config import EncoderConfig, Layer, Segment, Role, NumericConfig, EncodingStrategy

# Numeric binning for context_type: followup=0, standalone=1
CONTEXT_TYPE_NUMERIC = NumericConfig(
    bin_width=1.0,
    encoding_strategy=EncodingStrategy.THERMOMETER,
    min_value=0.0,
    max_value=2.0,
)

CONTEXT_TYPE_MAP = {"followup": 0.0, "standalone": 1.0}

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    layers=[
        Layer(
            name="router",
            similarity_weight=1.0,
            segments=[
                Segment(
                    name="intent",
                    roles=[
                        Role(name="verb", similarity_weight=1.0),
                        Role(name="object", similarity_weight=0.9),
                        Role(name="domain", similarity_weight=1.0),
                        Role(name="keywords", similarity_weight=0.7),
                    ],
                ),
                Segment(
                    name="action",
                    roles=[
                        Role(name="action_type", similarity_weight=0.8),
                        Role(name="action_id", similarity_weight=0.6),
                    ],
                ),
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="context_type",
                            similarity_weight=1.0,
                            numeric_config=CONTEXT_TYPE_NUMERIC,
                        ),
                    ],
                ),
            ],
        )
    ],
)
