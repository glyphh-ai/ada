"""
Strand API Routes — trajectory intelligence as a service.

Domain-agnostic: callers stream generic events (actor + action +
attributes), one session per conversation/workflow/user-journey. Every
posted event returns, in one response:

  predicted_next  what sessions like this one do next, with confidence
                  (pre-warm tools, prefetch data, suggest the next step)
  event_margin    how expected THIS event was given the session's strand
  drift           the session's running trajectory-anomaly score
                  (hijack / injection / workflow-drift signal)

The predictor is shared across sessions within a deployment, and every
observed event teaches it (one-shot Hebbian update, no training jobs):
the service gets better at a tenant's workflows just by watching them.
State is in-memory; persistence/sharding is deployment infrastructure.
"""

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from glyphh.core.ops import bundle, generate_symbol
from glyphh.strand.anticipator import Anticipator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/strand", tags=["strand"])

DIMENSION = 4096
SEED = 42
MAX_SESSIONS = 10_000


class StrandEvent(BaseModel):
    """One step of a trajectory: who did what, with what shape."""

    actor: str = Field(..., min_length=1, max_length=128)
    action: str = Field(..., min_length=1, max_length=128)
    attributes: Dict[str, str] = Field(default_factory=dict)


class EventResponse(BaseModel):
    session_id: str
    turn: int
    event_margin: float
    drift: float
    predicted_next: List[Dict[str, Any]]


class SessionSummary(BaseModel):
    session_id: str
    turns: int
    drift: float


class _StrandService:
    """One shared predictor; per-session strands."""

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self._symbols: Dict[str, Any] = {}
        self._anticipator = Anticipator(dimension=dimension)
        self._sessions: Dict[str, Anticipator] = {}
        self._lock = threading.Lock()

    def _sym(self, key: str):
        if key not in self._symbols:
            self._symbols[key] = generate_symbol(SEED, key, self.dimension)
        return self._symbols[key]

    def _codon(self, event: StrandEvent):
        parts = [self._sym("actor") * self._sym(f"actor:{event.actor}")]
        action = self._sym(f"action:{event.action}")
        keys = sorted(event.attributes) or ["none"]
        for k in keys:
            parts.append((action * self._sym(f"attr:{k}")).astype("int8"))
        return bundle(parts)

    @staticmethod
    def _label(event: StrandEvent) -> tuple:
        return (event.actor, event.action, tuple(sorted(event.attributes)))

    def create_session(self) -> str:
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                raise HTTPException(429, "Session limit reached")
            session_id = str(uuid.uuid4())[:12]
            session = Anticipator(dimension=self.dimension)
            # Sessions share the deployment-wide predictor.
            session.predictor = self._anticipator.predictor
            self._sessions[session_id] = session
            return session_id

    def get(self, session_id: str) -> Anticipator:
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(404, f"Unknown session {session_id}")
        return session

    def record(self, session_id: str, event: StrandEvent) -> EventResponse:
        session = self.get(session_id)
        with self._lock:
            margin = session.observe_turn(self._codon(event), self._label(event))
            predicted = session.anticipate(top_k=3)
        return EventResponse(
            session_id=session_id,
            turn=session.turns,
            event_margin=round(margin, 4),
            drift=round(session.drift(), 4),
            predicted_next=[
                {
                    "actor": label[0],
                    "action": label[1],
                    "attributes": list(label[2]),
                    "confidence": round(confidence, 4),
                }
                for label, confidence in predicted
            ],
        )

    def end_session(self, session_id: str) -> SessionSummary:
        session = self.get(session_id)
        with self._lock:
            summary = SessionSummary(
                session_id=session_id,
                turns=session.turns,
                drift=round(session.drift(), 4),
            )
            del self._sessions[session_id]
        return summary


_service: Optional[_StrandService] = None


def get_service() -> _StrandService:
    global _service
    if _service is None:
        _service = _StrandService()
    return _service


@router.post("/sessions", status_code=201)
async def create_session() -> Dict[str, str]:
    """Open a trajectory session."""
    return {"session_id": get_service().create_session()}


@router.post("/sessions/{session_id}/events", response_model=EventResponse)
async def record_event(session_id: str, event: StrandEvent) -> EventResponse:
    """Record one event; get next-step prediction and drift back."""
    return get_service().record(session_id, event)


@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def session_summary(session_id: str) -> SessionSummary:
    """Current turn count and drift for a session."""
    session = get_service().get(session_id)
    return SessionSummary(
        session_id=session_id,
        turns=session.turns,
        drift=round(session.drift(), 4),
    )


@router.delete("/sessions/{session_id}", response_model=SessionSummary)
async def end_session(session_id: str) -> SessionSummary:
    """Close a session; its trajectory stops, the learning is kept."""
    return get_service().end_session(session_id)
