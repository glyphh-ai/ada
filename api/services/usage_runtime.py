from __future__ import annotations

from fastapi import FastAPI


def record_usage(app: FastAPI, event_type: str, count: int = 1) -> None:
    tracker = getattr(app.state, "usage_tracker", None)
    if tracker:
        tracker.record(event_type, count)
