"""
Per-Model Chat API — conversational Glyphh match + LLM formatting.

Every deployed model gets this endpoint:
  POST /{org_id}/{model_id}/chat/message

Flow: User query → Glyphh HDC match → if confident: pure Glyphh answer
                                     → if low confidence: LLM synthesis
                                     → if follow-up: LLM with history

The runtime owns LLM selection (local or external). When no LLM is
configured (air-gapped), returns raw Glyphh output directly.

This replaces the platform's /assistant/chat — the platform no longer
makes LLM calls.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from domains.chat.service import ModelChatService, ChatTurn, ChatResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/chat", tags=["model-chat"])


class ModelChatRequest(BaseModel):
    """Request body for per-model chat."""
    message: str = Field(..., description="User message", min_length=1)
    history: Optional[List[Dict[str, str]]] = Field(
        None, description='Conversation history: [{"role": "user", "content": "..."}]'
    )
    session_id: Optional[str] = Field(None, description="Session ID for context")
    is_followup: bool = Field(False, description="Whether this is a conversational follow-up")


class ModelChatResponse(BaseModel):
    """Response from per-model chat."""
    state: str
    content: str
    command: Optional[str] = None
    code: Optional[str] = None
    confidence: float = 0.0
    match_method: str = "glyphh"
    fact_tree: Optional[Dict[str, Any]] = None
    provider: str = "glyphh"
    usage: Dict[str, Any] = {}
    trace_id: str = ""
    action_name: Optional[str] = None
    missing_slots: List[str] = []


@router.post("/message", response_model=ModelChatResponse)
async def model_chat(
    org_id: str,
    model_id: str,
    request: ModelChatRequest,
) -> ModelChatResponse:
    """Chat with any deployed model.

    The runtime queries the Glyphh model, then optionally formats
    the result through an LLM. Pure Glyphh answers are returned
    directly when the model has a confident match.
    """
    service = ModelChatService(org_id=org_id, model_id=model_id)

    history = []
    if request.history:
        for h in request.history:
            history.append(ChatTurn(
                role=h.get("role", "user"),
                content=h.get("content", ""),
            ))

    result = await service.chat(
        message=request.message,
        history=history,
        session_id=request.session_id,
        is_followup=request.is_followup,
    )

    return ModelChatResponse(
        state=result.state,
        content=result.content,
        command=result.command,
        code=result.code,
        confidence=result.confidence,
        match_method=result.match_method,
        fact_tree=result.fact_tree,
        provider=result.provider,
        usage=result.usage,
        trace_id=result.trace_id,
        action_name=result.action_name,
        missing_slots=result.missing_slots,
    )
