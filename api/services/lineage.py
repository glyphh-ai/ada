from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..core import models
from .similarity import compute_pair_metrics


def _base_name(name: str) -> str:
    return name.split("@", 1)[0]


def _parse_observed_at(name: str, semantic: dict) -> datetime | None:
    raw = semantic.get("observed_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    if "@" in name:
        _, suffix = name.split("@", 1)
        try:
            return datetime.fromisoformat(suffix.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _diff_semantic(prev: dict, curr: dict) -> dict:
    ignore = {"observed_at", "_space_id"}
    prev_keys = {k for k in prev.keys() if k not in ignore}
    curr_keys = {k for k in curr.keys() if k not in ignore}
    added = {k: curr[k] for k in sorted(curr_keys - prev_keys)}
    removed = {k: prev[k] for k in sorted(prev_keys - curr_keys)}
    changed = {}
    for key in sorted(prev_keys & curr_keys):
        if prev.get(key) != curr.get(key):
            changed[key] = {"from": prev.get(key), "to": curr.get(key)}
    return {"added": added, "removed": removed, "changed": changed}


def _row_payload(row: dict) -> dict:
    return {
        "name": row["name"],
        "observed_at": row["observed_at"].isoformat() if row["observed_at"] else None,
        "attributes": row["semantic"],
    }


def get_lineage_window(
    db: Session,
    *,
    model_id: str,
    name: str,
    observed_at: str | None = None,
) -> dict:
    base_name = _base_name(name)
    glyphs = (
        db.query(models.Glyph)
        .filter(models.Glyph.model_id == model_id)
        .filter(or_(models.Glyph.name == base_name, models.Glyph.name.like(f"{base_name}@%")))
        .all()
    )
    rows: list[dict[str, Any]] = []
    for glyph in glyphs:
        semantic = glyph.semantic or {}
        rows.append(
            {
                "name": glyph.name,
                "observed_at": _parse_observed_at(glyph.name, semantic),
                "semantic": semantic,
                "cortex": glyph.cortex,
            }
        )
    rows.sort(key=lambda row: row["observed_at"] or datetime.min)
    if not rows:
        return {"name": base_name, "previous": None, "current": None, "next": None, "deltas": []}

    target_observed = None
    if observed_at:
        try:
            target_observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError:
            target_observed = None

    if target_observed is None:
        idx = len(rows) - 1
    else:
        idx = next(
            (i for i, row in enumerate(rows) if row["observed_at"] == target_observed),
            len(rows) - 1,
        )

    previous = rows[idx - 1] if idx > 0 else None
    current = rows[idx]
    next_row = rows[idx + 1] if idx + 1 < len(rows) else None

    deltas: list[dict] = []
    if previous is not None:
        metrics = compute_pair_metrics(
            np.frombuffer(previous["cortex"], dtype=np.int8),
            np.frombuffer(current["cortex"], dtype=np.int8),
        )
        deltas.append(
            {
                "from": previous["name"],
                "to": current["name"],
                "observed_at_from": previous["observed_at"].isoformat() if previous["observed_at"] else None,
                "observed_at_to": current["observed_at"].isoformat() if current["observed_at"] else None,
                "field_diff": _diff_semantic(previous["semantic"], current["semantic"]),
                "similarity": {"cortex": metrics.shifted_cosine01, "layer": None, "segment": None},
            }
        )
    if next_row is not None:
        metrics = compute_pair_metrics(
            np.frombuffer(current["cortex"], dtype=np.int8),
            np.frombuffer(next_row["cortex"], dtype=np.int8),
        )
        deltas.append(
            {
                "from": current["name"],
                "to": next_row["name"],
                "observed_at_from": current["observed_at"].isoformat() if current["observed_at"] else None,
                "observed_at_to": next_row["observed_at"].isoformat() if next_row["observed_at"] else None,
                "field_diff": _diff_semantic(current["semantic"], next_row["semantic"]),
                "similarity": {"cortex": metrics.shifted_cosine01, "layer": None, "segment": None},
            }
        )

    return {
        "name": base_name,
        "previous": _row_payload(previous) if previous else None,
        "current": _row_payload(current) if current else None,
        "next": _row_payload(next_row) if next_row else None,
        "deltas": deltas,
    }
