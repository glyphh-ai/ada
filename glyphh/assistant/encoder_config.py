"""
Router-style encoder config for the Glyphh assistant model.

CANONICAL SOURCE: models/assistant/encoder.py
This file re-exports for backwards compatibility with scripts
that import from glyphh.assistant.encoder_config.
"""

import sys
from pathlib import Path

# Ensure the model directory's encoder.py is importable
_model_dir = Path(__file__).parent.parent.parent / "models" / "assistant"
if str(_model_dir) not in sys.path:
    sys.path.insert(0, str(_model_dir))

from encoder import ENCODER_CONFIG as ASSISTANT_ENCODER_CONFIG  # noqa: E402, F401
from encoder import CONTEXT_TYPE_NUMERIC, CONTEXT_TYPE_MAP  # noqa: E402, F401
