"""
Decide API Routes — precedent-grounded decisions with fact-tree receipts.

Jev-shaped interface, glyphh-shaped answer: POST a situation (flat
attributes, plus an optional ordered event list where order matters) and
get back the outcome distribution over your OWN recorded precedents, a
confidence with defined semantics (consensus among sufficient
precedent), and the full fact tree — every precedent cited with its
similarity, situation, facts, and glyph hash. When precedent is too
thin, the answer says so instead of guessing.

Record decided cases via /v1/decide/record: one-shot, no training run.
State is in-memory; persistence/tenancy is deployment infrastructure.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from glyphh.decision import DecisionSpace

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/decide", tags=["decide"])

DIMENSION = 4096


class DecideRequest(BaseModel):
    situation: Dict[str, str] = Field(..., min_length=1)
    events: Optional[List[Dict[str, str]]] = None
    k: Optional[int] = Field(None, ge=1, le=100)


class RecordRequest(BaseModel):
    situation: Dict[str, str] = Field(..., min_length=1)
    outcome: str = Field(..., min_length=1, max_length=256)
    facts: Optional[Dict[str, Any]] = None
    events: Optional[List[Dict[str, str]]] = None


_space: Optional[DecisionSpace] = None
_lock = threading.Lock()


def get_space() -> DecisionSpace:
    global _space
    if _space is None:
        _space = DecisionSpace(dimension=DIMENSION)
    return _space


@router.post("")
async def decide(request: DecideRequest) -> Dict[str, Any]:
    """Sim across recorded decisions; fact tree + confidence back."""
    space = get_space()
    with _lock:
        answer = space.query(request.situation, request.events, request.k)
    return answer.to_json()


@router.post("/record", status_code=201)
async def record(request: RecordRequest) -> Dict[str, Any]:
    """Store a decided case as precedent (one-shot, no training)."""
    space = get_space()
    with _lock:
        decision_id = space.record(
            request.situation, request.outcome,
            facts=request.facts, events=request.events,
        )
    return {"decision_id": decision_id, "precedents": len(space)}


@router.get("/stats")
async def stats() -> Dict[str, Any]:
    space = get_space()
    return {"precedents": len(space), "dimension": space.dimension,
            "k": space.k, "min_precedent_sim": space.min_precedent_sim}
