from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from ..core import models
from ..core.vector_space import build_vector_space_metadata

logger = logging.getLogger(__name__)

TEMPORAL_SIDECAR_MODEL_ID = "glyphh_temporal_metrics_v0.1"

TEMPORAL_SIDECAR_ROLES_CONFIG = {
    "model_id": TEMPORAL_SIDECAR_MODEL_ID,
    "vector_dim": 1536,
    "layers": [
        {
            "name": "metrics",
            "segments": [
                {
                    "name": "similarity",
                    "roles": [
                        {"role": "model_id", "type": "string"},
                        {"role": "test_name", "type": "string"},
                        {"role": "role", "type": "string"},
                        {"role": "segment", "type": "string"},
                        {"role": "timestamp", "type": "datetime"},
                        {"role": "cortex_similarity", "type": "number"},
                        {"role": "layer_similarity", "type": "number"},
                        {"role": "segment_similarity", "type": "number"},
                        {"role": "intent_name", "type": "string"},
                        {"role": "notes", "type": "string"},
                    ],
                }
            ],
        }
    ],
}

TEMPORAL_SIDECAR_NL_CONFIG = {
    "intents": [
        {
            "name": "similarity_trend",
            "type": "aggregate_sum",
            "amount_field": "cortex_similarity",
            "date_field": "timestamp",
            "time_filters": {"date_field": "timestamp", "range": "last_30_days"},
            "patterns": ["similarity trend", "drift trend"],
        },
        {
            "name": "recent_regressions",
            "type": "find",
            "role": "test_name",
            "filters": [{"role": "cortex_similarity", "value": "<0.6"}],
            "time_filters": {"date_field": "timestamp", "range": "last_7_days"},
            "patterns": ["recent regressions", "what regressed"],
        },
    ]
}


def ensure_temporal_sidecar_model(db: Session) -> models.Model:
    existing = db.get(models.Model, TEMPORAL_SIDECAR_MODEL_ID)
    if existing:
        return existing
    space_meta = build_vector_space_metadata(roles_config=TEMPORAL_SIDECAR_ROLES_CONFIG)
    model = models.Model(
        id=TEMPORAL_SIDECAR_MODEL_ID,
        name="Temporal Similarity Metrics",
        description="Sidecar model for temporal similarity tracking.",
        version=1,
        roles_config=TEMPORAL_SIDECAR_ROLES_CONFIG,
        vector_dim=space_meta["vector_dim"],
        encoder_seed=space_meta["encoder_seed"],
        space_id=space_meta["space_id"],
        space_version=space_meta["space_version"],
        data_mode="live",
        status="active",
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    nl_cfg = models.ModelNLConfig(
        id=TEMPORAL_SIDECAR_MODEL_ID,
        model_id=model.id,
        config={"config": TEMPORAL_SIDECAR_NL_CONFIG},
        version=1,
    )
    db.add(nl_cfg)
    runtime_cfg = models.ModelRuntimeConfig(
        id=TEMPORAL_SIDECAR_MODEL_ID,
        model_id=model.id,
        config={"sidecar": "temporal"},
        version=1,
    )
    db.add(runtime_cfg)
    db.commit()
    logger.info("temporal sidecar model registered")
    return model
