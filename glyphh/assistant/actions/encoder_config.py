"""
Router-style encoder config for the Glyphh actions model.

Same architecture as the assistant model but tuned for action routing:
the intent segment captures what the user wants to do, and the action
segment captures which API endpoint to call.
"""

from glyphh.core.config import EncoderConfig, Layer, Segment, Role

ACTIONS_ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=43,  # Different seed from assistant (42) for independent vector space
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
                        Role(name="action_name", similarity_weight=0.6),
                    ],
                ),
            ],
        )
    ],
)
