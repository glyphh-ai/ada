"""
Router-style encoder config for the Glyphh assistant model.

The assistant is a router: it takes natural language input and routes
to actions (commands, knowledge responses, quick actions, query help).
This config defines the vector space structure for that routing.

The intent segment captures what the user wants.
The action segment captures what to route to.
"""

from glyphh.core.config import EncoderConfig, Layer, Segment, Role, NumericConfig, EncodingStrategy

# Numeric binning for context_type: followup=0, standalone=1
# Thermometer encoding with 2 bins produces maximally-separated vectors:
#   bin 0 (followup):   [0, 0]  → distinct symbol
#   bin 1 (standalone): [1, 0]  → distinct symbol
# This gives the context segment real discriminating power.
CONTEXT_TYPE_NUMERIC = NumericConfig(
    bin_width=1.0,
    encoding_strategy=EncodingStrategy.THERMOMETER,
    min_value=0.0,
    max_value=2.0,
)

# Mapping from string labels to numeric bin values
CONTEXT_TYPE_MAP = {"followup": 0.0, "standalone": 1.0}

ASSISTANT_ENCODER_CONFIG = EncoderConfig(
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
