from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Iterable

from glyphh.encoder import Encoder
from glyphh.vector import bind
from sqlalchemy.orm import Session

from ..core import models
from ..core.db import SessionLocal
from ..core.schemas import ConceptInput

logger = logging.getLogger("glyphh.runtime.history")


def persist_history_vectors(
    model: models.Model,
    concepts: Iterable[ConceptInput],
    listener_id: str | None = None,
) -> None:
    encoder = Encoder(
        config=model.roles_config,
        dim=model.vector_dim,
        seed=model.encoder_seed,
    )
    db: Session = SessionLocal()
    saved = 0
    try:
        for concept in concepts:
            for role_name, raw_value in concept.attributes.items():
                slot = encoder.role_to_slot.get(role_name)
                if slot is None:
                    continue
                layer_idx, seg_idx = slot
                if raw_value is None:
                    value_token = "null"
                elif isinstance(raw_value, (str, bool, int, float)):
                    value_token = str(raw_value)
                else:
                    value_token = json.dumps(raw_value, sort_keys=True)
                bound_vec = bind(encoder.role_vec(role_name), encoder.value_vec(value_token))
                vector = [float(v) for v in bound_vec]
                db.add(
                    models.GlyphHistoryVector(
                        model_id=model.id,
                        listener_id=listener_id,
                        role=role_name,
                        layer=layer_idx,
                        segment_index=seg_idx,
                        vector=vector,
                        attribute_value=raw_value,
                        vector_metadata={
                            "concept": concept.name,
                            "timestamp": dt.datetime.utcnow().isoformat(),
                        },
                    )
                )
                saved += 1
        if saved:
            db.commit()
    except Exception:
        db.rollback()
        logger.exception("failed to persist history vectors (listener=%s)", listener_id)
    finally:
        db.close()
